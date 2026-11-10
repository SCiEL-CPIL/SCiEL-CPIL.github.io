# EVOLVE-BLOCK-START
import time
import numpy as np


_DEADLINE = 630.0
_SAFETY = 15.0  # return this many seconds before the hard kill


def _time_left() -> float:
    return _DEADLINE - time.time() - _SAFETY


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot(yaw, pitch, roll):
    return _rx(roll) @ _ry(pitch) @ _rz(yaw)


def _build_path(params, n=120):
    """
    params: np.array of 10 floats
      0: yaw_mid   (deg)  angle at which we start the 0->90 yaw sweep
      1: yaw_span  (deg)  total yaw sweep (should be ~90)
      2: pitch_mid (deg)
      3: pitch_span(deg)
      4: roll_mid  (deg)
      5: roll_span (deg)
      6: x_amp     (amplitude of x oscillation in the first turn)
      7: y_amp     (amplitude of y oscillation in the second turn)
      8: z_lift    (how high we lift in the second turn)
      9: corner_y  (where the y trajectory kinks, ~5.0)
    """
    yaw_mid, yaw_span = params[0], params[1]
    pitch_mid, pitch_span = params[2], params[3]
    roll_mid, roll_span = params[4], params[5]
    x_amp, y_amp, z_lift, corner_y = params[6], params[7], params[8], params[9]

    t = np.linspace(0.0, 1.0, n)

    # smoothstep for a C1 trajectory
    def smooth(u):
        u = np.clip(u, 0, 1)
        return u * u * (3 - 2 * u)

    # yaw: 0 -> 90, centered around yaw_mid fraction of the path
    yaw_frac = np.clip((t - (yaw_mid / 90.0)) / (yaw_span / 90.0), 0, 1)
    yaw = 90.0 * smooth(yaw_frac)

    # pitch: 0 -> 90, centered around pitch_mid
    pitch_frac = np.clip((t - (pitch_mid / 90.0)) / (pitch_span / 90.0), 0, 1)
    pitch = 90.0 * smooth(pitch_frac)

    # roll: 0 -> 90, centered around roll_mid
    roll_frac = np.clip((t - (roll_mid / 90.0)) / (roll_span / 90.0), 0, 1)
    roll = 90.0 * smooth(roll_frac)

    # y: go from 0 to 6, with a kink at corner_y
    y = 6.0 * t
    # add a gentle S‑curve around the corner
    y = y + y_amp * np.sin(np.pi * t) * 0.1  # small wiggle to help clear the corner

    # x: start at ~1, dip toward 0 near the first turn, then back to ~0 at end
    # First turn happens when yaw is small (t < 0.3)
    # We want x to be ~1 when yaw~0, and ~0 when yaw~90
    x = 1.0 - x_amp * smooth(np.clip(t / 0.5, 0, 1))
    # add a small oscillation to help the block pivot around the corner
    x = x + 0.05 * np.sin(2 * np.pi * t)

    # z: start at 0, lift to ~1 in the second turn (when pitch is large)
    z = z_lift * smooth(np.clip((t - 0.4) / 0.6, 0, 1))

    poses = np.column_stack([x, y, z, yaw, pitch, roll])
    return poses


def _densify(poses, max_step=0.05, max_ang=2.0):
    """Insert linearly interpolated poses so each step moves <= max_step and
    changes each angle by <= max_ang degrees."""
    out = [poses[0]]
    for i in range(1, len(poses)):
        p0, p1 = poses[i - 1], poses[i]
        dpos = np.linalg.norm(p1[:3] - p0[:3])
        dang = np.max(np.abs(p1[3:] - p0[3:]))
        n = max(1, int(np.ceil(max(dpos / max_step, dang / max_ang, 1))))
        for j in range(1, n + 1):
            alpha = j / n
            out.append(p0 * (1 - alpha) + p1 * alpha)
    return np.array(out)


def _in_corridor(pts):
    """pts: (N,3) array. Returns boolean mask of points inside the corridor."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Region 1: x<=0, 0<=y<=1, 0<=z<=1
    r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    # Region 2: 0<=x<=1, 0<=y<=5, 0<=z<=1
    r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    # Region 3: 0<=x<=1, 5<=y<=6, z>=0
    r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return r1 | r2 | r3


def _evaluate(poses, grid_x, grid_y, grid_z, n_steps=30):
    """
    Fast vectorized evaluation: for each of n_steps sampled poses (linearly
    interpolated along the dense path), check if all grid points survive.
    Returns the number of 4x4x4 blocks that survive.
    """
    # Sample n_steps poses from the dense path
    idx = np.linspace(0, len(poses) - 1, n_steps).astype(int)
    sampled = poses[idx]

    # Grid points
    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    N = pts.shape[0]

    # Precompute base grid shape
    nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)

    # For each sampled pose, transform all grid points and check corridor
    # We'll accumulate a boolean array of shape (n_steps, N)
    survive = np.ones((n_steps, N), dtype=bool)

    for i, pose in enumerate(sampled):
        x, y, z, yaw_deg, pitch_deg, roll_deg = pose
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        R = _rot(yaw, pitch, roll)
        # Transform: p_world = R @ p + t
        transformed = pts @ R.T + np.array([x, y, z])
        survive[i] = _in_corridor(transformed)

    # A grid point survives if it survives in ALL sampled poses
    all_survive = np.all(survive, axis=0).reshape(nx, ny, nz)

    # Count 4x4x4 blocks where all points survive
    # Use a simple sliding window sum
    block_sum = np.zeros((nx - 3, ny - 3, nz - 3), dtype=np.int32)
    for dx in range(4):
        for dy in range(4):
            for dz in range(4):
                block_sum += all_survive[dx:dx + nx - 3, dy:dy + ny - 3, dz:dz + nz - 3].astype(np.int32)

    return int(np.sum(block_sum == 64))


def _evaluate_full(poses, grid_x, grid_y, grid_z):
    """Full evaluation using all dense poses (slower but more accurate)."""
    idx = np.arange(len(poses))
    sampled = poses[idx]

    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    N = pts.shape[0]

    survive = np.ones((len(sampled), N), dtype=bool)

    for i, pose in enumerate(sampled):
        x, y, z, yaw_deg, pitch_deg, roll_deg = pose
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        R = _rot(yaw, pitch, roll)
        transformed = pts @ R.T + np.array([x, y, z])
        survive[i] = _in_corridor(transformed)

    all_survive = np.all(survive, axis=0)
    nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)
    all_survive = all_survive.reshape(nx, ny, nz)

    block_sum = np.zeros((nx - 3, ny - 3, nz - 3), dtype=np.int32)
    for dx in range(4):
        for dy in range(4):
            for dz in range(4):
                block_sum += all_survive[dx:dx + nx - 3, dy:dy + ny - 3, dz:dz + nz - 3].astype(np.int32)

    return int(np.sum(block_sum == 64))


def moving_sofa_3d_path() -> np.ndarray:
    t0 = time.time()

    # Grid as specified
    grid_x = np.linspace(-4, 0, 81)
    grid_y = np.linspace(0, 1, 21)
    grid_z = np.linspace(0, 1, 21)

    # Initial parameter guess
    best_params = np.array([
        10.0,   # yaw_mid: start yaw sweep early
        80.0,   # yaw_span
        40.0,   # pitch_mid: pitch sweep in the middle
        50.0,   # pitch_span
        60.0,   # roll_mid: roll sweep later
        30.0,   # roll_span
        0.8,    # x_amp
        0.3,    # y_amp
        0.8,    # z_lift
        5.0,    # corner_y
    ])

    # Evaluate initial guess
    poses = _densify(_build_path(best_params))
    best_score = _evaluate(poses, grid_x, grid_y, grid_z, n_steps=30)

    # Coordinate descent
    n_params = 10
    step_sizes = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.1, 0.05, 0.1, 0.2]
    bounds = [
        (0, 50),    # yaw_mid
        (30, 90),   # yaw_span
        (10, 70),   # pitch_mid
        (30, 90),   # pitch_span
        (30, 80),   # roll_mid
        (20, 60),   # roll_span
        (0.3, 1.2), # x_amp
        (0.0, 0.5), # y_amp
        (0.3, 1.2), # z_lift
        (4.0, 6.0), # corner_y
    ]

    # Number of poses for fast evaluation
    n_steps_fast = 30

    while _time_left() > 30.0:
        improved = False
        for i in range(n_params):
            if _time_left() < 30.0:
                break
            for sign in [-1, 1]:
                trial = best_params.copy()
                trial[i] += sign * step_sizes[i]
                # Clip to bounds
                trial[i] = np.clip(trial[i], bounds[i][0], bounds[i][1])

                poses = _densify(_build_path(trial))
                score = _evaluate(poses, grid_x, grid_y, grid_z, n_steps=n_steps_fast)

                if score > best_score:
                    best_params = trial
                    best_score = score
                    improved = True
                    # Shrink step for this parameter
                    step_sizes[i] *= 0.8
                else:
                    # If no improvement, shrink step
                    step_sizes[i] *= 0.9

            # If no improvement in either direction, shrink step more
            if not improved:
                step_sizes[i] *= 0.7

        # If all steps are very small, do a final full evaluation
        if max(step_sizes) < 0.01:
            break

    # Final full evaluation to get the true score
    if _time_left() > 10.0:
        poses = _densify(_build_path(best_params))
        final_score = _evaluate_full(poses, grid_x, grid_y, grid_z)
        # If the full score is worse, fall back to the fast-evaluated path
        # (but we still return the path; the evaluator will compute the true score)

    return _densify(_build_path(best_params))


# EVOLVE-BLOCK-END
