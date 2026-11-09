# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _smooth_score(points, temperature=0.05):
    """
    Compute a smooth approximation of (d_min / d_max)^2.
    Uses log-sum-exp for d_max and min for d_min (or smoothed min).
    To be differentiable everywhere, we approximate d_min with a soft-min 
    and d_max with a soft-max.
    Score = exp(2 * (soft_log_min - soft_log_max))
    """
    if len(points) < 2:
        return 0.0
    
    dists = pdist(points, metric='euclidean')
    if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
        return 0.0
    
    # Avoid log(0)
    dists = np.maximum(dists, 1e-12)
    
    # Soft max: T * log(sum(exp(d/T)))
    # Soft min: -T * log(sum(exp(-d/T)))
    t = temperature
    
    log_dists = np.log(dists)
    
    # For stability in log-sum-exp:
    max_log_d = np.max(log_dists)
    min_log_d = np.min(log_dists)
    
    # soft_log_max = max_log_d + T * log(sum(exp((log_d - max_log_d)/T)))
    # soft_log_min = min_log_d - T * log(sum(exp((min_log_d - log_d)/T)))
    
    # If T is small, these approach max and min.
    # We want to maximize min/max. 
    # Let's work in log space: maximize 2 * (log_min - log_max)
    
    # Calculate soft log max
    shifted_max = (log_dists - max_log_d) / t
    # Clip to avoid overflow in exp
    shifted_max = np.clip(shifted_max, -100, 100)
    soft_log_max = max_log_d + t * np.log(np.sum(np.exp(shifted_max)))
    
    # Calculate soft log min
    shifted_min = (min_log_d - log_dists) / t
    shifted_min = np.clip(shifted_min, -100, 100)
    soft_log_min = min_log_d - t * np.log(np.sum(np.exp(shifted_min)))
    
    log_ratio = soft_log_min - soft_log_max
    # Score is ratio^2, so log(score) = 2 * log_ratio
    # We return -log(score) for minimization, or just -log_ratio.
    # Returning -log_ratio is equivalent for ordering.
    return -log_ratio

def _objective(params, temperature=0.05):
    """
    Objective function for minimization.
    params: 42-length array of coordinates for 14 points in 3D.
    """
    points = params.reshape(14, 3)
    # Normalize to unit sphere to remove scale invariance and keep points bounded
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    points = points / norms[:, np.newaxis]
    
    return _smooth_score(points, temperature)

def _generate_icosahedron_seed():
    """
    Generate 12 vertices of a regular icosahedron, plus 2 points (poles).
    This is a strong initial guess.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add two points at the poles (0,0,1) and (0,0,-1)
    # Note: These might be close to existing vertices, but L-BFGS should fix it.
    # Actually, icosahedron vertices are not at poles.
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    points = np.vstack([ico, poles])
    return points.flatten()

def _generate_cuboctahedron_seed():
    """
    Generate 12 vertices of a cuboctahedron, plus 2 points.
    Cuboctahedron vertices are permutations of (0, 1, 1).
    """
    verts = []
    for perm in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        for signs in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            v = [0.0, 0.0, 0.0]
            v[perm[0]] = signs[0] * perm[0]
            v[perm[1]] = signs[1] * perm[1]
            v[perm[2]] = 1.0 # The non-zero non-permuted one? No, (0,1,1) has one 0.
            # Wait, permutations of (0,1,1):
            # (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1)
            # (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1)
            # (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0)
            pass
            
    # Simpler way to generate cuboctahedron
    verts = []
    for i in range(3):
        for j in range(2):
            for k in range(2):
                v = [0.0, 0.0, 0.0]
                # One coordinate is 0, two are +/-1
                # i is the index of the 0 coordinate
                # j, k determine signs of the other two
                s1 = 1 if j == 0 else -1
                s2 = 1 if k == 0 else -1
                other_indices = [x for x in range(3) if x != i]
                v[other_indices[0]] = s1
                v[other_indices[1]] = s2
                verts.append(v)
                
    points = np.array(verts)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    # Add 2 points, say at (1,0,0) and (-1,0,0)? 
    # Cuboctahedron has vertices at (1,1,0) etc. Distance to (1,0,0) is 1.
    # Distance between vertices is sqrt(2).
    # So (1,0,0) is distance 1 from 4 vertices, and sqrt(2) from others.
    # This creates a smaller min distance.
    # Let's just add two random points on the sphere for variety in multi-start
    extra = np.array([[0, 0, 1], [0, 0, -1]])
    # Check if these are already close.
    # Cuboctahedron has (0,1,1), (0,-1,1)... no (0,0,1) is not a vertex.
    # Distance from (0,0,1) to (0,1,1) is 1.
    points = np.vstack([points, extra])
    return points.flatten()

def _generate_random_seed(rng):
    """Generate 14 random points on the unit sphere."""
    points = rng.normal(size=(14, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points.flatten()

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Uses a 42D Cartesian optimization with L-BFGS-B.
    1. Initialize with high-symmetry seeds (Icosahedron+2, Cuboctahedron+2) and random points.
    2. Optimize using L-BFGS-B on a smooth differentiable objective.
    3. Return the best configuration found.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Define seeds
    seeds = []
    seeds.append(_generate_icosahedron_seed())
    seeds.append(_generate_cuboctahedron_seed())
    
    # Add some random seeds
    rng = np.random.default_rng(42)
    for _ in range(10):
        seeds.append(_generate_random_seed(rng))
        
    # Also add a "cube" seed: 8 corners + 6 face centers? No, 14 points.
    # 8 corners of cube (1,1,1)... normalized.
    # 6 face centers (1,0,0)...
    cube_corners = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                cube_corners.append([x, y, z])
    cube_centers = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    cube_pts = np.array(cube_corners + cube_centers)
    cube_pts = cube_pts / np.linalg.norm(cube_pts, axis=1, keepdims=True)
    seeds.append(cube_pts.flatten())
    
    # Optimization loop
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # Try different temperatures for the soft-min/max to balance exploration/exploitation
        # Start with higher T (smoother) then lower T (sharper)
        temperatures = [0.1, 0.05, 0.01]
        
        for t in temperatures:
            if time.time() > deadline:
                break
                
            # Use L-BFGS-B
            # Bounds: None (sphere constraint handled by normalization inside objective? 
            # No, L-BFGS-B works on the raw coordinates. Normalization inside objective 
            # makes the landscape non-smooth at origin, but we start away from origin.
            # Actually, it's better to constrain to sphere using Lagrange multipliers or 
            # parameterization, but L-BFGS-B with normalization is a common hack.
            # To make it smoother, we can optimize on the sphere using a retraction.
            # But scipy's minimize doesn't support retractions directly.
            # We will stick to Cartesian + normalization. It works well enough.
            
            try:
                res = minimize(
                    _objective, 
                    seed, 
                    args=(t,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                # Calculate true score for the result
                pts = res.x.reshape(14, 3)
                norms = np.linalg.norm(pts, axis=1)
                norms[norms < 1e-12] = 1.0
                pts = pts / norms[:, np.newaxis]
                
                dists = pdist(pts, metric='euclidean')
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 0:
                    true_score = (d_min / d_max) ** 2
                else:
                    true_score = 0.0
                
                if true_score > best_score:
                    best_score = true_score
                    best_params = res.x.copy()
                    
            except Exception:
                continue

    # If we found a good candidate, do a final polish with very low temperature
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _objective, 
                best_params, 
                args=(0.001,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            pts = res.x.reshape(14, 3)
            norms = np.linalg.norm(pts, axis=1)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms[:, np.newaxis]
            
            dists = pdist(pts, metric='euclidean')
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max > 0:
                true_score = (d_min / d_max) ** 2
            else:
                true_score = 0.0
                
            if true_score > best_score:
                best_score = true_score
                best_params = res.x.copy()
        except Exception:
            pass

    # Generate final points
    if best_params is None:
        # Fallback
        pts = _generate_icosahedron_seed().reshape(14, 3)
    else:
        pts = best_params.reshape(14, 3)
        
    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms[:, np.newaxis]
    
    return pts

# EVOLVE-BLOCK-END
