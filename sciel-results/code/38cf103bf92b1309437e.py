# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _score_polygon(polygon):
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Check convexity: no repeated points, all turns same sign, total turn = ±2π
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(centered[i] - centered[j]) < 1e-10:
                return 0.0
    
    # Compute cross products for turn direction
    cross_products = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_products.append(cross)
    
    # All must be nonzero and same sign
    nonzero = [c for c in cross_products if abs(c) > 1e-12]
    if len(nonzero) != n:
        return 0.0
    if any(c > 0 for c in nonzero) and any(c < 0 for c in nonzero):
        return 0.0
    
    # Check total turning angle
    angles = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        if norm_e1 < 1e-12 or norm_e2 < 1e-12:
            return 0.0
        cos_angle = np.dot(e1, e2) / (norm_e1 * norm_e2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if cross_products[i] > 0:
            angle = angle
        else:
            angle = -angle
        angles.append(angle)
    
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def _score_symmetric(offsets):
    """
    Given 5 angular offsets, construct a 20-gon with 4-fold rotational symmetry.
    The 20 vertices are at angles: offsets[i % 5] + (i // 5) * pi/2 for i in 0..19.
    Returns the minimum vertex score.
    """
    n = 20
    # Construct angles
    base_offsets = np.sort(offsets)
    # Ensure they are within a 90-degree sector [0, pi/2) to avoid overlap issues
    # Actually, let's just use them as is, but the symmetry repeats them.
    # To maintain convexity and proper ordering, we need the 5 offsets to be sorted
    # and the resulting 20 angles to be sorted.
    
    angles = np.zeros(20)
    for i in range(20):
        idx = i % 5
        block = i // 5
        angles[i] = base_offsets[idx] + block * (np.pi / 2.0)
    
    # Sort angles to ensure cyclic order
    angles = np.sort(angles)
    # Normalize to [0, 2*pi)
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return _score_polygon(polygon)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Optimize 5 angular offsets for a 20-gon with 4-fold rotational symmetry.
    This reduces the search space from 20D to 5D, allowing for much deeper search.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    n_offsets = 5
    base_angles = np.linspace(0.0, np.pi/2.0, n_offsets, endpoint=False)
    
    # Get baseline score from regular polygon (offsets are equally spaced in 90 deg)
    best_score = _score_symmetric(base_angles)
    best_offsets = base_angles.copy()
    
    # Objective for differential evolution: maximize min vertex score
    def objective(offsets_flat):
        if time.time() > deadline:
            return 0.0
        s = _score_symmetric(offsets_flat)
        return -s
    
    # Bounds: each offset in [0, pi/2)
    bounds = [(0.0, np.pi/2.0)] * n_offsets
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=500,
            popsize=20,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < 0:  # valid result (negative of score)
            opt_offsets = result.x
            opt_score = _score_symmetric(opt_offsets)
            if opt_score > best_score:
                best_score = opt_score
                best_offsets = opt_offsets.copy()
    except Exception:
        pass
    
    # If time remains, do local refinement around best
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        # Simulated annealing style local search
        current_offsets = best_offsets.copy()
        current_score = best_score
        T = 0.01
        T_min = 1e-6
        
        for _ in range(100000):
            if time.time() > deadline - 5.0:
                break
            if T < T_min:
                break
            
            # Perturb one or two offsets
            num_perturb = rng.choice([1, 2], p=[0.7, 0.3])
            indices = rng.choice(n_offsets, size=num_perturb, replace=False)
            new_offsets = current_offsets.copy()
            for idx in indices:
                new_offsets[idx] += rng.normal(0, T)
                new_offsets[idx] = np.clip(new_offsets[idx], 0.0, np.pi/2.0)
            
            new_score = _score_symmetric(new_offsets)
            
            if new_score > current_score:
                current_offsets = new_offsets
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_offsets = current_offsets.copy()
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_offsets = new_offsets
                        current_score = new_score
            
            T *= 0.9999
    
    # Construct final polygon from best offsets
    base_offsets = np.sort(best_offsets)
    angles = np.zeros(20)
    for i in range(20):
        idx = i % 5
        block = i // 5
        angles[i] = base_offsets[idx] + block * (np.pi / 2.0)
    
    angles = np.sort(angles)
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon

# EVOLVE-BLOCK-END
