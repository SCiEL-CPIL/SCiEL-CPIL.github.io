import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s buffer for safety


def _calc_score(points):
    """Calculate the minimum radius for a given configuration."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _ring_init(r_est=0.125):
    """Generate ring-based initial configuration matching known optimal structure.
    
    Structure: 1 center + 5 ring1 + 13 ring2 + 31 ring3 = 50
    """
    centers = np.zeros((N, 2))
    
    # Center point
    centers[0] = [0.0, 0.0]
    
    # Ring 1: 5 points
    # Adjacent points in ring should be ~2*r_est apart
    # Chord = 2*R1*sin(pi/5) = 2*r_est => R1 = r_est/sin(pi/5)
    R1 = r_est / np.sin(np.pi / 5)
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        centers[1 + i] = [R1 * np.cos(angle), R1 * np.sin(angle)]
    
    # Ring 2: 13 points
    # These should be between ring1 and ring3
    # Estimate R2 based on touching ring1 points
    # Distance from ring2 point to nearest ring1 point ~ 2*r_est
    # Let's estimate R2 ~ 2 * R1 * 0.6 or use iterative approach
    # For 13 points, chord = 2*R2*sin(pi/13) should be ~2*r_est
    R2 = r_est / np.sin(np.pi / 13)
    # But R2 should be larger than R1. Let's scale it
    # Actually, let's just place them at a radius that makes sense
    # The gap between ring1 and ring2 should be ~2*r_est
    # So R2 ~ R1 + 2*r_est roughly, but need to check adjacency
    # Let's use a more careful estimate
    R2 = R1 + 1.5 * r_est  # rough estimate
    for i in range(13):
        angle = 2 * np.pi * i / 13
        centers[6 + i] = [R2 * np.cos(angle), R2 * np.sin(angle)]
    
    # Ring 3: 31 points
    # Outermost ring, should be near boundary
    # 31 points on circle R3, chord = 2*R3*sin(pi/31) ~ 2*r_est
    R3 = r_est / np.sin(np.pi / 31)
    # But also need to fit in unit circle: R3 + r_est <= 1
    R3 = min(R3, 1.0 - r_est)
    for i in range(31):
        angle = 2 * np.pi * i / 31 + np.pi / 31  # offset
        centers[19 + i] = [R3 * np.cos(angle), R3 * np.sin(angle)]
    
    return centers


def _hex_init(perturb=0.0):
    """Generate hexagonal lattice configuration as fallback."""
    best_points = None
    best_score = -1
    
    for scale in np.linspace(0.22, 0.28, 8):
        k = int(1.0 / scale) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * scale
                y = j * scale * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    rows.append([x, y])
        points = np.array(rows)
        if len(points) < N:
            continue
        
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
        if perturb > 0:
            points = points + np.random.randn(N, 2) * perturb
        
        score = _calc_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    return best_points


def _optimize(points, max_steps=5000, initial_step=0.02):
    """Gradient ascent with random restarts to escape local minima."""
    step = initial_step
    best_score = _calc_score(points)
    best_points = points.copy()
    no_improve_count = 0
    
    for s in range(max_steps):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Compute pairwise distances and boundary distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dist = 1.0 - radii
        
        min_pair_dist = np.min(dist)
        min_boundary_dist = np.min(boundary_dist)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score + 1e-10:
            best_score = current_score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 200:
                break
        
        # Determine active constraint and apply gradient step
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Inter-circle constraint is active
            idx_i, idx_j = np.unravel_index(np.argmin(dist), dist.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            
            vec = points[idx_i] - points[idx_j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
            points[idx_j] -= step * vec
        else:
            # Boundary constraint is active
            idx_i = np.argmin(boundary_dist)
            vec = -points[idx_i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
        
        # Occasionally add small random perturbation to escape local minima
        if s % 50 == 0 and s > 0:
            points += np.random.randn(N, 2) * step * 0.1
        
        # Decay step size
        step *= 0.995
        
    return best_points


def construct():
    best_points = None
    best_score = -1
    
    # Strategy 1: Ring-based initialization (matches known optimal structure)
    for r_est in [0.125, 0.126, 0.124, 0.127, 0.123]:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        points = _ring_init(r_est)
        score_init = _calc_score(points)
        
        if score_init > best_score:
            best_score = score_init
            best_points = points.copy()
        
        optimized = _optimize(points.copy(), max_steps=3000, initial_step=0.015)
        score_opt = _calc_score(optimized)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = optimized.copy()
    
    # Strategy 2: Hex lattice with perturbation (diversity)
    for restart in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        points = _hex_init(perturb=0.005 * (restart + 1))
        if points is None:
            continue
        
        optimized = _optimize(points.copy(), max_steps=2000, initial_step=0.01)
        score = _calc_score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    # Final polish: small steps on best configuration
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 5:
        polished = _optimize(best_points.copy(), max_steps=1000, initial_step=0.003)
        score = _calc_score(polished)
        if score > best_score:
            best_points = polished
    
    # Ensure validity: clip points to be strictly inside unit circle
    if best_points is not None:
        radii = np.hypot(best_points[:, 0], best_points[:, 1])
        for i in range(N):
            if radii[i] >= 1.0:
                # Push inward
                scale = 0.999 / radii[i] if radii[i] > 0 else 1.0
                best_points[i] *= scale
    
    return best_points
