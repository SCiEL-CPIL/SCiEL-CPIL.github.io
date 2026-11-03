"""Pack 200 equal circles in a unit square using force-directed optimization with cooling."""
import numpy as np
import time
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 1] if False else pts[:, 0])  # keep consistent
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance using KDTree for efficiency
    tree = KDTree(pts)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
        dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(dists)

    return min(wall_dist, min_pair_dist / 2.0)


def _compute_forces(pts, r_target=0.035):
    """
    Compute repulsive forces between circles and attractive forces toward center.
    Uses a soft potential to avoid singularities.
    """
    n = len(pts)
    forces = np.zeros_like(pts)

    # Pairwise repulsion: F = k * (1/d^2 - 1/(2r)^2) * direction if d < 2r
    # Actually, we want to maximize min distance. Use a potential that penalizes close pairs.
    # V(d) = 1/d^alpha for d < d_cut, else 0
    # Force = -dV/dd * direction = alpha/d^(alpha+1) * direction (repulsive)
    
    # Use a softer approach: for each pair within 2*r_target, push apart
    # Also add wall forces
    
    # Pairwise
    tree = KDTree(pts)
    # Query pairs within a reasonable distance
    query_dist = 2 * r_target * 1.5
    pairs = tree.query_pairs(query_dist, output_type='ndarray')
    
    if len(pairs) > 0:
        diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero
        mask = dists > 1e-10
        if np.any(mask):
            diffs_valid = diffs[mask]
            dists_valid = dists[mask]
            # Force magnitude: strong repulsion when close
            # F = k / d^2
            k = 0.1
            mags = k / (dists_valid ** 2)
            # Normalize direction
            dirs = diffs_valid / dists_valid[:, np.newaxis]
            f = mags[:, np.newaxis] * dirs
            # Accumulate forces
            for i in range(len(f)):
                forces[pairs[mask][i, 0]] += f[i]
                forces[pairs[mask][i, 1]] -= f[i]

    # Wall forces: push away from boundaries
    # Left wall: if x < -0.5 + r, push right
    r_eff = r_target
    forces[:, 0] += np.where(pts[:, 0] < -0.5 + r_eff, (-0.5 + r_eff - pts[:, 0]) * 10, 0)
    forces[:, 0] += np.where(pts[:, 0] > 0.5 - r_eff, (0.5 - r_eff - pts[:, 0]) * 10, 0)
    forces[:, 1] += np.where(pts[:, 1] < -0.5 + r_eff, (-0.5 + r_eff - pts[:, 1]) * 10, 0)
    forces[:, 1] += np.where(pts[:, 1] > 0.5 - r_eff, (0.5 - r_eff - pts[:, 1]) * 10, 0)

    return forces


def _hex_grid_initial(n=200):
    """Generate a hexagonal grid initial configuration."""
    # Target spacing for ~200 circles in unit square
    # Area per circle ~ 1/200 = 0.005
    # Hex spacing: s * s * sqrt(3)/2 = 0.005 => s^2 = 0.005 * 2 / sqrt(3) = 0.00577 => s = 0.076
    s = 0.076
    row_h = s * np.sqrt(3) / 2
    
    pts = []
    y = -0.5 + s / 2
    row_idx = 0
    while y <= 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s / 2 + shift
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
            x += s
        y += row_h
        row_idx += 1
    
    pts = np.array(pts)
    
    # If we have more than n, select best n
    if len(pts) > n:
        # Use greedy selection
        selected = []
        remaining = list(range(len(pts)))
        for _ in range(n):
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                if len(selected) == 0:
                    min_d = 1.0
                else:
                    diffs = pts[idx] - pts[selected]
                    min_d = np.min(np.linalg.norm(diffs, axis=1))
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        pts = pts[selected]
    elif len(pts) < n:
        # Pad with random points
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    return pts[:n]


def _select_best_subset(pts, n):
    """Select n points from pts to maximize minimum distance."""
    if len(pts) <= n:
        return pts
    
    selected = []
    remaining = list(range(len(pts)))
    
    for _ in range(n):
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            if len(selected) == 0:
                min_d = 1.0
            else:
                diffs = pts[idx] - pts[selected]
                min_d = np.min(np.linalg.norm(diffs, axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return pts[selected]


def _optimize_from_initial(pts, t0, time_limit):
    """Run force-directed optimization with cooling schedule."""
    pts = pts.copy()
    n = len(pts)
    
    # Cooling schedule
    temp_start = 0.05
    temp_end = 0.001
    num_steps = 5000
    
    best_pts = pts.copy()
    best_r = _get_radius(pts)
    
    for step in range(num_steps):
        if time.time() - t0 > time_limit - 5:
            break
        
        # Current temperature
        progress = step / num_steps
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Compute forces
        forces = _compute_forces(pts, r_target=best_r)
        
        # Random perturbation
        noise = np.random.normal(0, temp, size=pts.shape)
        
        # Update positions
        pts_new = pts + forces * 0.01 + noise
        
        # Keep within bounds (with small margin)
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.499, 0.499)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.499, 0.499)
        
        # Evaluate
        r_new = _get_radius(pts_new)
        
        # Accept if better or with probability exp(-(r_old - r_new)/temp)
        if r_new >= best_r or np.random.random() < np.exp(-(best_r - r_new) / max(temp, 1e-10)):
            pts = pts_new
            if r_new > best_r:
                best_r = r_new
                best_pts = pts_new.copy()
    
    return best_pts, best_r


def construct():
    t0 = time.time()
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple initial configurations
    num_trials = 8
    
    for trial in range(num_trials):
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
        
        if trial == 0:
            # Hex grid
            pts = _hex_grid_initial(N)
        elif trial == 1:
            # Random
            pts = np.random.uniform(-0.45, 0.45, size=(N, 2))
        elif trial == 2:
            # Grid
            side = int(np.sqrt(N))
            extra = N - side * side
            pts = []
            for i in range(side):
                for j in range(side):
                    pts.append((-0.5 + (i + 0.5) / side, -0.5 + (j + 0.5) / side))
            # Add extra points
            for k in range(extra):
                pts.append((-0.5 + (k + 0.5) / (side + 1), 0.5 - 0.01))
            pts = np.array(pts)
        else:
            # Perturbed hex
            pts = _hex_grid_initial(N)
            pts += np.random.normal(0, 0.02, size=pts.shape)
            pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
            pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        
        # Optimize
        opt_pts, opt_r = _optimize_from_initial(pts, t0, _TIME_LIMIT)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _hex_grid_initial(N)
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
