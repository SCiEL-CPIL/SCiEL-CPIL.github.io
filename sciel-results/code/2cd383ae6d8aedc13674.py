# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 140.0
    best_points = None
    best_score = 0.0
    
    # Vectorized score calculation
    def calc_score(points):
        # points: (N, 2)
        # 1. Check boundary
        dists_origin = np.linalg.norm(points, axis=1)
        if np.any(dists_origin > 1.0):
            return 0.0
        min_boundary = 1.0 - np.max(dists_origin)
        
        # 2. Pairwise distances
        # Compute all pairwise differences
        # diff[i, j] = points[i] - points[j]
        # We only need upper triangle for min
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_pair / 2.0, min_boundary)

    # Ring configurations
    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 16, 15],
        [1, 6, 10, 14, 19],
        [1, 5, 10, 15, 19],
        [1, 6, 11, 16, 16],
        [1, 6, 12, 17, 14],
    ]
    
    for config in ring_configs:
        if time.time() > deadline - 20:
            break
        
        n_rings = len(config)
        counts = np.array(config)
        
        # Initial radii
        init_radii = np.linspace(0.2, 0.8, n_rings)
        init_offsets = np.zeros(n_rings)
        x0 = np.concatenate([init_radii, init_offsets])
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            
            radii = x[:n_rings]
            offsets = x[n_rings:]
            
            if np.any(radii < 0) or np.any(radii > 1.0):
                return 1e10
            
            points = []
            for i, (count, ring_r) in enumerate(zip(counts, radii)):
                offset = offsets[i]
                angles = offset + 2 * np.pi * np.arange(count) / count
                xs = ring_r * np.cos(angles)
                ys = ring_r * np.sin(angles)
                points.append(np.column_stack((xs, ys)))
            
            points = np.vstack(points)
            
            score = calc_score(points)
            
            # Update global best
            nonlocal best_score
            nonlocal best_points
            if score > best_score:
                best_score = score
                best_points = points.copy()
            
            return -score

        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < -best_score + 1e-9:
                # Reconstruct and verify
                radii = result.x[:n_rings]
                offsets = result.x[n_rings:]
                points = []
                for i, (count, ring_r) in enumerate(zip(counts, radii)):
                    offset = offsets[i]
                    angles = offset + 2 * np.pi * np.arange(count) / count
                    xs = ring_r * np.cos(angles)
                    ys = ring_r * np.sin(angles)
                    points.append(np.column_stack((xs, ys)))
                points = np.vstack(points)
                
                score = calc_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass

    # Final Polish: Simple iterative relaxation
    if best_points is not None:
        points = best_points.copy()
        # Run a few iterations of repulsion + boundary constraint
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            
            # Compute forces
            forces = np.zeros_like(points)
            dists_origin = np.linalg.norm(points, axis=1)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-6:
                        d = np.random.normal(size=2) * 1e-6
                        dist = np.linalg.norm(d)
                    # Strong repulsion if close
                    force_mag = 1.0 / (dist * dist) if dist < 0.5 else 0.0
                    forces[i] += d / dist * force_mag
                    forces[j] -= d / dist * force_mag
            
            # Boundary attraction/repulsion
            for i in range(N):
                if dists_origin[i] > 0.9:
                    # Push in if too close to boundary? No, we want to maximize r, so we want them as far from boundary as possible?
                    # Actually, the score is limited by min(pair_dist/2, 1-dist).
                    # If 1-dist is the bottleneck, we should move points inward.
                    # If pair_dist is the bottleneck, we should move them apart.
                    # A simple heuristic: move towards center if very close to boundary.
                    forces[i] -= points[i] * 0.1 * (dists_origin[i] - 0.8)
            
            # Update positions
            points += forces * 0.01
            
            # Project back inside unit circle if they went out
            new_dists = np.linalg.norm(points, axis=1)
            mask = new_dists > 1.0
            if np.any(mask):
                points[mask] = points[mask] / new_dists[mask] * 0.999
            
            score = calc_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
            else:
                # Revert if it got worse significantly? Or just keep exploring.
                pass

    if best_points is not None:
        return best_points

    # Fallback
    return np.zeros((N, 2))
# EVOLVE-BLOCK-END
