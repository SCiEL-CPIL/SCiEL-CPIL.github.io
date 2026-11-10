# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

# Precompute constants for unit hexagons (angle 0)
_HEX_ANGLES = np.deg2rad(60.0 * np.arange(6))
_HEX_COS = np.cos(_HEX_ANGLES)
_HEX_SIN = np.sin(_HEX_ANGLES)
_HEX_VERTS = np.column_stack([_HEX_COS, _HEX_SIN]) # (6, 2)

# Outward normals for the container hexagon (angle 0)
# Vertices at 0, 60, 120... => Normals at 30, 90, 150...
_NORM_ANGLES = np.deg2rad(30.0 + 60.0 * np.arange(6))
_NORM_COS = np.cos(_NORM_ANGLES)
_NORM_SIN = np.sin(_NORM_ANGLES)
_NORMS = np.column_stack([_NORM_COS, _NORM_SIN]) # (6, 2)

# Factor to convert max projection to side length s
# Projection of hexagon with side s onto normal: s * sqrt(3)
# We want s * sqrt(3) >= max_proj => s = max_proj / sqrt(3)
_SQRT3 = np.sqrt(3.0)

def _min_containing_hex_side(centers):
    """
    Compute minimum side length s of a regular hexagon centered at origin, angle 0,
    containing all unit hexagons with given centers.
    """
    # Centers shape: (12, 2)
    # Verts shape: (12, 6, 2)
    verts = centers[:, np.newaxis, :] + _HEX_VERTS[np.newaxis, :, :]
    
    # Project onto 6 normals
    # verts_flat: (72, 2)
    # projections: (6, 72)
    max_proj_per_normal = np.max(verts @ _NORMS.T, axis=1)
    max_proj = np.max(max_proj_per_normal)
    
    return max_proj / _SQRT3

def _separation_violation(centers):
    """
    Compute total separation violation (how much pairs overlap).
    Returns 0 if all pairs are separated, positive value otherwise.
    Vectorized for speed.
    """
    n = len(centers)
    # centers: (12, 2)
    # _HEX_VERTS: (6, 2)
    
    total_violation = 0.0
    # To save time in tight loops, we can vectorize pair checks
    # But 12*11/2 = 66 pairs. Vectorizing all 66 at once is possible.
    
    # Get all pairs i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    ci = centers[i_idx] # (66, 2)
    cj = centers[j_idx] # (66, 2)
    
    # Verts for i: (66, 6, 2)
    vi = ci[:, np.newaxis, :] + _HEX_VERTS[np.newaxis, :, :]
    vj = cj[:, np.newaxis, :] + _HEX_VERTS[np.newaxis, :, :]
    
    # Project onto normals
    # pi: (66, 6, 6) -> proj of vi onto 6 normals
    # Actually, normals are (6,2). 
    # vi @ normals.T -> (66, 6, 6) where last dim is normal index
    pi = vi @ _NORMS.T # (66, 6, 6)
    pj = vj @ _NORMS.T # (66, 6, 6)
    
    min_i = np.min(pi, axis=1) # (66, 6)
    max_i = np.max(pi, axis=1) # (66, 6)
    min_j = np.min(pj, axis=1) # (66, 6)
    max_j = np.max(pj, axis=1) # (66, 6)
    
    # Overlap on each of the 6 axes for each pair
    # overlap_k = min(max_i_k, max_j_k) - max(min_i_k, min_j_k)
    overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j) # (66, 6)
    
    # A pair is separated if ANY axis has overlap < -tol (gap)
    # Here we check for violation: if max overlap across axes is > 0, they intersect.
    # Wait, SAT says they are disjoint if there is a separating axis.
    # If for ALL axes, the intervals overlap (overlap >= -tol), then they intersect.
    # We want to penalize intersection.
    # Intersection depth is roughly the minimum gap? No.
    # If they intersect, all overlaps are >= 0 (roughly).
    # We want to push them apart.
    # A good penalty is the square of the maximum overlap on any axis?
    # Actually, if they intersect, the "violation" is how much they are pushed in.
    # Let's use the sum of positive overlaps on all axes? No, that's not right.
    # If they intersect, all 6 axes show overlap > 0.
    # If they are separated by gap g on axis k, overlap_k is negative.
    # We want to maximize the minimum gap?
    # Let's define violation for a pair as:
    # If separated (min overlap < -1e-6), violation = 0.
    # If intersecting (max overlap > 0), violation = (max_overlap)^2.
    # This encourages separation.
    
    max_overlap_per_pair = np.max(overlaps, axis=1) # (66,)
    
    # If max_overlap > 0, they intersect.
    # Penalty: sum of squares of positive max_overlaps
    violations = np.where(max_overlap_per_pair > 0, max_overlap_per_pair**2, 0.0)
    
    return np.sum(violations)

def _objective(params):
    """
    Objective: minimize the containing hexagon side length with penalty for overlaps.
    """
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    
    # Penalty weight needs to be large enough to prevent overlap
    # s is around 3-4. Violation is sum of squares of overlaps.
    # If overlap is 0.1, violation is 0.01. 100 * 0.01 = 1.
    # So penalty 100 is reasonable.
    penalty = 100.0 * violation
    return s + penalty

def _make_initial_hex_lattice():
    """
    Create a hexagonal lattice arrangement.
    """
    # Center at (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 5 hexagons (to make 12 total)
    
    # Flat-top hexagon geometry:
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Offset for odd columns: sqrt(3)/2
    
    init = np.array([
        [0.0, 0.0],
        # Ring 1
        [1.5, _SQRT3/2],
        [1.5, -_SQRT3/2],
        [0.0, _SQRT3],
        [0.0, -_SQRT3],
        [-1.5, _SQRT3/2],
        [-1.5, -_SQRT3/2],
        # Ring 2 (partial)
        [3.0, 0.0],
        [2.25, 1.5*_SQRT3], # 2.25, 2.598
        [0.75, 2.0*_SQRT3], # 0.75, 3.464
        [-1.5, 1.5*_SQRT3], # -1.5, 2.598
        [-3.0, 0.0],
    ])
    return init

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Bounds for differential evolution
    # The container side s is at least sqrt(12) ~ 3.46.
    # Centers should be within roughly [-s, s]
    # Let's use generous bounds to allow global search
    bounds = [(-5.0, 5.0)] * 24
    
    # Initial point from hex lattice
    x0 = _make_initial_hex_lattice().flatten()
    
    best_centers = x0.reshape(12, 2)
    best_s = _min_containing_hex_side(best_centers)
    
    # Run Differential Evolution
    # DE is good for global optimization and handles non-smooth penalties better than NM
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True, # Run L-BFGS-B at the end for precision
            init='sobol', # Better space filling
            x0=x0 # Use our good initial point as one of the population
        )
        
        if result.success or result.fun < best_s:
            candidate = result.x.reshape(12, 2)
            # Verify separation
            if _separation_violation(candidate) < 1e-12:
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
                    
    except Exception:
        pass
    
    # Check time and maybe do a local refinement if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        # Try a few random restarts of DE with smaller popsize
        rng = np.random.default_rng(123)
        for r in range(5):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            # Perturb initial point
            noise = rng.normal(0, 0.5, size=(12, 2))
            x0_pert = (best_centers + noise).flatten()
            
            try:
                res_p = differential_evolution(
                    _objective,
                    bounds,
                    seed=100+r,
                    maxiter=50,
                    popsize=10,
                    tol=1e-8,
                    polish=True,
                    x0=x0_pert
                )
                if res_p.fun < best_s:
                    candidate = res_p.x.reshape(12, 2)
                    if _separation_violation(candidate) < 1e-12:
                        s_cand = _min_containing_hex_side(candidate)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_centers = candidate.copy()
            except Exception:
                continue

    # Final validation and output
    # Ensure no overlaps
    if _separation_violation(best_centers) > 1e-12:
        # Fallback to initial lattice if optimization failed
        best_centers = _make_initial_hex_lattice()
        best_s = _min_containing_hex_side(best_centers)
        
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
