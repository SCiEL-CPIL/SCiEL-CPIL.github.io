"""Vectorized force-directed packing for 300 circles in a unit disk."""

import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def _pack_score(pts):
    """Compute packing radius: min of boundary and pairwise distances."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    r_bound = 1.0 - np.linalg.norm(pts, axis=1)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return float(min(np.min(r_bound), min_pair))


def _init_points(n, seed=0):
    """Hexagonal lattice initialization with centering."""
    rng = np.random.RandomState(seed)
    spacing = 2.0 / (np.sqrt(n) * 0.98)
    pts = []
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                pts.append([x, y])
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < n:
        pts = rng.uniform(-0.95, 0.95, size=(n, 2))
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.95
        pts[mask] *= 0.95 / norms[mask, np.newaxis]
        return pts
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind='stable')[:n]
    pts = pts[idx].copy()
    pts += rng.normal(0, spacing * 0.01, size=pts.shape)
    return pts


def _force_directed_relax(pts, deadline):
    """Spatial hashing force-directed optimization."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_score = _pack_score(pts)
    
    # Estimate current radius for grid size
    # We use the current best score as a proxy for radius
    r_est = max(best_score, 0.01)
    
    lr = 0.005
    it = 0
    no_improve = 0
    center_attraction = 0.1
    
    while time.time() < deadline - 15:
        # Build spatial hash
        # Grid cell size should be at least 2 * r_est to cover all interactions
        cell_size = 2.0 * r_est
        if cell_size < 0.01:
            cell_size = 0.01
            
        grid = {}
        # Map points to grid cells
        for i in range(n):
            x, y = pts[i]
            cx = int(np.floor(x / cell_size))
            cy = int(np.floor(y / cell_size))
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        forces = np.zeros_like(pts)
        
        # Calculate forces for each point
        for i in range(n):
            x, y = pts[i]
            cx = int(np.floor(x / cell_size))
            cy = int(np.floor(y / cell_size))
            
            # Check 3x3 neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if i == j:
                                continue
                            dx_vec = x - pts[j][0]
                            dy_vec = y - pts[j][1]
                            dist_sq = dx_vec**2 + dy_vec**2
                            dist = np.sqrt(dist_sq)
                            
                            if dist < 1e-9:
                                # Random nudge if coincident
                                forces[i] += np.random.uniform(-0.01, 0.01, size=2)
                                continue
                            
                            # Repulsion if overlapping (dist < 2*r_est)
                            # We use a soft repulsion: F = 1/d^2
                            # This drives them apart until they are far apart
                            # But we want them to be close. 
                            # Standard packing: Minimize overlaps.
                            # If dist < 2*r_est, there is overlap.
                            # Force should be proportional to overlap amount?
                            # Let's use F = (2*r_est - dist) / dist * (dx_vec, dy_vec)
                            # This is a spring force that pushes them to distance 2*r_est.
                            
                            target_dist = 2.0 * r_est
                            if dist < target_dist:
                                # Spring constant
                                k = 1.0
                                force_mag = k * (target_dist - dist)
                                forces[i] += (force_mag / dist) * np.array([dx_vec, dy_vec])
            
            # Center attraction to keep them in
            forces[i] -= center_attraction * pts[i]
            
            # Boundary force: push inward if too close to edge
            norm = np.linalg.norm(pts[i])
            if norm > (1.0 - r_est):
                # Push inward proportional to violation
                violation = norm - (1.0 - r_est)
                forces[i] -= (violation * 10.0 / norm) * pts[i]
        
        # Update positions
        pts_new = pts + lr * forces
        
        # Hard constraint: keep inside unit circle
        norms_new = np.linalg.norm(pts_new, axis=1)
        mask_out = norms_new > (1.0 - r_est)
        if np.any(mask_out):
            pts_new[mask_out] *= (1.0 - r_est) / norms_new[mask_out, np.newaxis]
            
        # Update best
        score = _pack_score(pts_new)
        if score > best_score:
            best_score = score
            best_pts = pts_new.copy()
            no_improve = 0
            # Update r_est to current best to refine grid
            r_est = max(score, 0.01)
        else:
            no_improve += 1
            if no_improve > 50:
                lr *= 0.9
                if lr < 1e-6:
                    break
            # Also update r_est slightly to allow grid to adapt if score is stuck
            # but only if we are improving? No, r_est should reflect the target.
            # If score is not improving, maybe r_est is too high?
            # Let's keep r_est as best_score.
        
        pts = pts_new
        it += 1
        
        # Periodically re-evaluate r_est from best_pts to ensure grid is appropriate
        if it % 100 == 0:
             r_est = max(best_score, 0.01)

    return best_pts, best_score


def _local_search(pts, deadline):
    """Refine with small random perturbations around best point."""
    rng = np.random.RandomState(123)
    n = pts.shape[0]
    best_pts = pts.copy()
    best_score = _pack_score(pts)
    step = 0.005

    while time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            direction = rng.normal(0, 1, size=(n, 2))
            direction /= (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9)
            pts_try = best_pts + step * direction
            norms = np.linalg.norm(pts_try, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                pts_try[mask] *= 0.999 / norms[mask, np.newaxis]
            s = _pack_score(pts_try)
            if s > best_score:
                best_score = s
                best_pts = pts_try.copy()
        step *= 0.85
        if step < 1e-5:
            break

    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    best_pts = None
    best_score = -np.inf

    for seed in range(8):
        if time.time() > deadline - 40:
            break

        pts = _init_points(N, seed)
        pts, score = _force_directed_relax(pts, deadline)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    if best_pts is not None and time.time() < deadline - 20:
        best_pts, best_score = _local_search(best_pts, deadline)

    if best_pts is None:
        best_pts = _init_points(N, 0)

    return best_pts
# EVOLVE-BLOCK-END
