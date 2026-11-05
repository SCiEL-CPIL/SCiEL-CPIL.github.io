import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _score_from_params(r2, phi):
    """
    Calculate the score for a configuration defined by r2 and phi,
    assuming r1 = 1.
    Points are two squares, one of radius 1, one of radius r2,
    rotated by phi relative to each other.
    """
    # Square 1 (radius 1)
    # Angles: 0, pi/2, pi, 3pi/2
    p1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0]
    ])
    
    # Square 2 (radius r2, rotated by phi)
    # Angles: phi, phi+pi/2, phi+pi, phi+3pi/2
    angles = np.array([phi, phi + np.pi/2, phi + np.pi, phi + 3*np.pi/2])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    p2 = r2 * np.column_stack((cos_a, sin_a))
    
    points = np.vstack((p1, p2))
    
    # Calculate pairwise distances
    # Using broadcasting for distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Find max distance (diameter)
    max_dist = np.max(dists)
    
    if max_dist == 0:
        return -np.inf
        
    # Scale to diameter 2
    scale = 2.0 / max_dist
    points_scaled = points * scale
    
    # Recalculate distances after scaling (or just scale the product)
    # Product of distances scales as scale^(n*(n-1))
    # n=8, so exponent is 56.
    # But it's safer to just recompute or scale the dists.
    # dists_scaled = dists * scale
    # product = np.prod(dists_scaled[dists_scaled > 0]) # Avoid 0s? No, distinct points.
    
    # Let's just use the scaled distances directly for product
    # We only need upper triangle
    i, j = np.triu_indices(8, k=1)
    d_scaled = dists[i, j] * scale
    product = np.prod(d_scaled)
    
    score = product / (8.0 ** 8)
    return score

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Regular octagon as baseline
    # r1 = 1/sqrt(2), r2 = 1/sqrt(2), phi = pi/4
    # In our parameterization r1=1, so r2=1, phi=pi/4
    baseline_score = _score_from_params(1.0, np.pi/4)
    if baseline_score > best_score:
        best_score = baseline_score
        best_params = [1.0, np.pi/4]
        # Generate baseline points for return if needed
        p1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        p2 = 1.0 * np.column_stack((np.cos(angles), np.sin(angles)))
        best_points = np.vstack((p1, p2))
        
    # Optimization
    # Parameters: r2, phi
    # Bounds: r2 in [0.1, 10], phi in [0, pi/4]
    # Due to symmetry, phi in [0, pi/4] is sufficient.
    
    def objective(params):
        r2, phi = params
        # Ensure positive r2
        if r2 <= 0:
            return 0.0
        s = _score_from_params(r2, phi)
        return -s  # Minimize negative score
        
    # Differential Evolution
    try:
        # Check time
        if time.time() - start_time < deadline - 10:
            res = differential_evolution(
                objective, 
                bounds=[(0.1, 10.0), (0.0, np.pi/4)], 
                seed=42, 
                maxiter=1000, 
                popsize=20,
                tol=1e-8
            )
            if res.success:
                r2_opt, phi_opt = res.x
                score_opt = -res.fun
                if score_opt > best_score:
                    best_score = score_opt
                    best_params = [r2_opt, phi_opt]
                    # Generate points
                    p1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
                    angles = np.array([phi_opt, phi_opt + np.pi/2, phi_opt + np.pi, phi_opt + 3*np.pi/2])
                    p2 = r2_opt * np.column_stack((np.cos(angles), np.sin(angles)))
                    best_points = np.vstack((p1, p2))
                    
    except Exception as e:
        pass

    # Local Refinement with L-BFGS-B
    if best_params is not None and time.time() - start_time < deadline - 5:
        try:
            res_ref = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B',
                bounds=[(0.1, 10.0), (0.0, np.pi/4)],
                options={'maxiter': 1000}
            )
            if res_ref.success:
                r2_ref, phi_ref = res_ref.x
                score_ref = -res_ref.fun
                if score_ref > best_score:
                    best_score = score_ref
                    best_params = [r2_ref, phi_ref]
                    p1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
                    angles = np.array([phi_ref, phi_ref + np.pi/2, phi_ref + np.pi, phi_ref + 3*np.pi/2])
                    p2 = r2_ref * np.column_stack((np.cos(angles), np.sin(angles)))
                    best_points = np.vstack((p1, p2))
        except Exception:
            pass

    # Final Fallback: Regular Octagon
    if best_points is None:
        from math import cos, sin, pi
        best_points = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])

    # Ensure distinct points and valid format
    # The points are already distinct for r2 > 0 and phi not making them coincide.
    # Convert to list of lists of floats
    return best_points.tolist()

# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to ensure it's the entry point.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program has `def construct():` inside the EVOLVE-BLOCK.
# I will place the helper functions outside or inside as appropriate, but `construct` must be the main entry.
# To be safe and follow the structure, I'll put everything inside the block or ensure `construct` is accessible.
# Standard practice for these tasks: Define helpers outside, `construct` inside or outside.
# The seed had `construct` inside the block. I will do the same, but I can define helpers globally.
# Actually, the prompt says "Rewrite the program from scratch". I will provide the full code.
# The EVOLVE-BLOCK markers usually imply the content between them is what gets swapped.
# I will put the helper functions and the construct function inside the block to be self-contained if possible,
# or just ensure `construct` is defined.
# Let's put helpers outside to avoid re-definition overhead if called multiple times?
# No, `construct` is called once per evaluation.
# I will define helpers outside the block to keep the block clean, or just put everything in.
# The seed had only `construct` in the block. I will put helpers outside.
# EVOLVE-BLOCK-END
