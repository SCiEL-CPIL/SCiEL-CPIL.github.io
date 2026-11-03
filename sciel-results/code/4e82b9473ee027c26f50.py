"""Rigid-body dynamics packing of 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300
TIME_LIMIT = 115.0  # safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _dynamics(pts, start_time, time_limit):
    """Molecular dynamics simulation with Lennard-Jones-like repulsion."""
    n = len(pts)
    pos = pts.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    # Parameters
    sigma = 0.1  # Interaction length scale (roughly 2x target radius)
    epsilon = 10.0  # Energy scale
    dt = 0.005  # Time step
    damping = 0.95  # Velocity damping factor
    max_steps = 10000
    
    # Precompute indices for neighbor pairs (optimization: only compute forces for close pairs)
    # For N=300, full O(N^2) is ~45k pairs, which is fast enough in NumPy if vectorized,
    # but we can optimize by only considering pairs within a cutoff if needed.
    # Given N=300, full pairwise is ~45k operations per step. 10000 steps = 450M ops.
    # This might be too slow in Python loops, but vectorized NumPy should handle it.
    
    best_pos = pos.copy()
    best_score = _score(pos)
    
    # Adaptive damping: start with high damping to settle, then lower it
    # We will run for a fixed number of steps or until time limit
    
    step = 0
    while step < max_steps:
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        # d2 matrix
        d2 = np.sum(pos * pos, axis=1)[:, None] - 2.0 * (pos @ pos.T) + np.sum(pos * pos, axis=1)[None, :]
        np.fill_diagonal(d2, np.inf)
        
        # Avoid division by zero or negative due to floating point
        d2 = np.maximum(d2, 1e-12)
        d = np.sqrt(d2)
        
        # Lennard-Jones force magnitude: F = 24*eps*(2*(sig/d)^12 - (sig/d)^6)/d
        # Simplified repulsive force: F = k / d^3 (Coulomb-like) or LJ
        # Let's use a strong repulsive potential: U = 1/d^6, F = 6/d^7 * r_vec
        # Or standard LJ: U = 4*eps*((sig/d)^12 - (sig/d)^6)
        # Force on i from j: F_ij = -dU/dd * (r_i - r_j)/d
        
        # For stability and speed, let's use a simpler repulsive force: F = 1/d^2 (Coulomb)
        # or F = 1/d^3.
        # Let's use LJ-like but only repulsive part to avoid attraction traps.
        # F_mag = 24 * eps * (2 * (sig/d)^12 - (sig/d)^6) / d
        # If d < sig, this is positive (repulsive). If d > sig, it becomes attractive.
        # We want pure repulsion to maximize separation.
        # So, F_mag = k / d^3 is a good simple repulsive force.
        
        # Let's use F = 1/d^3 for simplicity and robustness.
        # Force vector F_ij = (pos[i] - pos[j]) / d^3
        # Sum over j != i
        
        # Vectorized force calculation
        # d2 is (N, N)
        # d is (N, N)
        # Force magnitude factor: 1/d^3
        # But we need to apply direction: (pos[i] - pos[j])
        
        # pos[i] is (N, 2) -> (N, 1, 2)
        # pos[j] is (N, 2) -> (1, N, 2)
        # diff = pos[:, None, :] - pos[None, :, :] (N, N, 2)
        
        diff = pos[:, None, :] - pos[None, :, :]
        
        # d3 = d**3
        # Avoid division by zero on diagonal
        d3 = d**3
        np.fill_diagonal(d3, np.inf)
        
        # Force = diff / d3
        # This gives force on i from j?
        # F_ij = k * (r_i - r_j) / |r_i - r_j|^3
        # Yes.
        
        forces = diff / d3[:, :, None]
        
        # Sum over j (axis 1) to get net force on i
        net_forces = np.sum(forces, axis=1)
        
        # Boundary force: push inward if close to boundary
        rad = np.hypot(pos[:, 0], pos[:, 1])
        # If rad > 0.9, apply inward force
        # F_wall = k_wall * (1 - rad/1.0) * r_hat ?
        # Let's use a soft boundary: if rad > 0.8, force = k * (rad - 0.8) * (-r_hat)
        boundary_force = np.zeros_like(pos)
        mask = rad > 0.8
        if np.any(mask):
            # r_hat = pos / rad
            r_hat = pos[mask] / rad[mask, None]
            # Strength proportional to how far out
            strength = (rad[mask] - 0.8) * 10.0
            boundary_force[mask] = -strength[:, None] * r_hat
            
        # Total force
        total_forces = net_forces + boundary_force
        
        # Update velocity
        vel += total_forces * dt
        vel *= damping
        
        # Update position
        pos += vel * dt
        
        # Hard constraint: keep inside unit circle
        rad = np.hypot(pos[:, 0], pos[:, 1])
        out = rad > 1.0
        if np.any(out):
            pos[out] = pos[out] / rad[out, None]
            # Zero out radial velocity component for those points?
            # Simple: just zero velocity for those points to prevent oscillation
            vel[out] = 0.0
        
        # Check score
        # To save time, only compute full score every 10 steps
        if step % 10 == 0:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
        
        step += 1
        
        # Adaptive dt: if energy increases too much, reduce dt?
        # Or just keep it simple.
        
    return best_pos, best_score


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    # ensure valid start: clip inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    base_score = _score(pts)
    best_pts = pts.copy()
    best_score = base_score

    try:
        opt_pts, opt_score = _dynamics(pts, start_time, TIME_LIMIT)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass

    # final safety: ensure finite, inside, shape (N,2)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts
