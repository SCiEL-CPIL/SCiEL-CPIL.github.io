"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

# Precompute all triple indices for fast area calculation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

def _compute_areas(points):
    """Compute all triangle areas efficiently using precomputed cross products."""
    p = points  # (N, 2)
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[j_idx, k_idx] + C[i_idx, j_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull_area = _hull_area(points)
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temp=0.02):
    """
    Differentiable approximation of the score using soft-min.
    This allows gradient-based optimization.
    """
    if len(points) < 3:
        return -1e9
    
    try:
        hull_area = _hull_area(points)
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
        
    areas = _compute_areas(points)
    
    # Soft min: -log(sum(exp(-area/temp))) / temp
    # To avoid overflow, subtract max
    min_a = np.min(areas)
    exp_areas = np.exp(-(areas - min_a) / temp)
    soft_min_val = min_a - temp * np.log(np.sum(exp_areas))
    
    return soft_min_val / hull_area

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on unit circle (outer), 7 on circle of radius r (inner).
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _generate_jittered_polygon(amplitude, noise_vector):
    """
    Generate 14 points as a regular 14-gon with radial jitter.
    noise_vector is of length 14, representing radial displacements.
    """
    base = _generate_regular_polygon(N)
    # Normalize base to unit radius (it already is)
    # Add radial jitter
    # base is (N, 2). We want to move each point along its radial direction.
    # Radial direction is just the point itself (since it's on unit circle).
    # So pts = base * (1 + noise_vector)
    pts = base * (1.0 + noise_vector[:, np.newaxis])
    return pts

def _optimize_general(deadline):
    """
    Optimize the general 14-point configuration.
    1. Try Two-Ring structures (fast, low-dim).
    2. Try Jittered Regular Polygon (breaks symmetry, low-dim).
    3. Full 28-dim coordinate optimization on best candidate.
    """
    best_score = -float('inf')
    best_pts = None
    
    # --- Strategy 1: Two-Ring Optimization ---
    # This is often a strong baseline.
    # We optimize r and theta.
    def two_ring_obj(params):
        r, theta = params
        if r < 0.0 or r > 1.5:
            return 1e6
        pts = _generate_two_ring_points(r, theta)
        return -_score(pts)
        
    # Try a few initial guesses for two-ring
    r_starts = [0.3, 0.5, 0.7, 0.9]
    theta_starts = [0.0, np.pi/7, np.pi/14, 2*np.pi/7]
    
    for r0 in r_starts:
        for th0 in theta_starts:
            if time.time() > deadline - 30.0:
                break
            try:
                res = minimize(
                    two_ring_obj, 
                    [r0, th0], 
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True}
                )
                if res.fun < 1e5:
                    cand_pts = _generate_two_ring_points(res.x[0], res.x[1])
                    cand_score = _score(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
            except Exception:
                pass

    # --- Strategy 2: Jittered Regular Polygon ---
    # The "jittered" regular 14-gon where each vertex is displaced radially.
    # We optimize the 14 radial displacements.
    # To keep it low-dimensional and structured, we can use a small number of Fourier modes
    # or just optimize the 14 values directly with Nelder-Mead (14D is manageable).
    # Let's use 14D Nelder-Mead.
    
    def jitter_obj(params):
        # params: 14 radial displacements
        # Constrain displacements to be small, e.g., [-0.2, 0.2]
        # If a point goes too far in, it might create a small triangle with neighbors.
        # If it goes too far out, it might not hurt but increases hull area.
        pts = _generate_jittered_polygon(1.0, params)
        return -_score(pts)
        
    # Initial guesses for jitter
    # 1. All zeros (regular polygon)
    # 2. Small random noise
    jitter_seeds = [
        np.zeros(N),
        np.random.normal(0, 0.05, N),
        np.random.normal(0, 0.1, N),
        # Symmetric jitter: alternate +d, -d
        np.array([0.05 if i % 2 == 0 else -0.05 for i in range(N)]),
        # Fourier-like jitter
        0.1 * np.cos(2 * np.linspace(0, 2*np.pi, N, endpoint=False)),
        0.1 * np.sin(2 * np.linspace(0, 2*np.pi, N, endpoint=False))
    ]
    
    for seed_j in jitter_seeds:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                jitter_obj, 
                seed_j, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300, 'adaptive': True}
            )
            if res.fun < 1e5:
                cand_pts = _generate_jittered_polygon(1.0, res.x)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # --- Strategy 3: Full Coordinate Optimization ---
    # Use the best candidate so far as the starting point for full 28D optimization.
    # This allows the structure to break free from the "jittered polygon" or "two-ring" constraints.
    
    if best_pts is None:
        best_pts = _generate_regular_polygon(N)
        
    # Use soft-min score for smoother landscape in full optimization
    def full_obj(params):
        pts = params.reshape(N, 2)
        # Add a penalty if points get too close or hull collapses
        # _soft_min_score handles hull collapse with -1e9
        return -_soft_min_score(pts, temp=0.02)
        
    # Nelder-Mead on 28D
    try:
        res = minimize(
            full_obj, 
            best_pts.flatten(), 
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 500, 'adaptive': True}
        )
        if res.fun < 1e5:
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
    except Exception:
        pass

    # --- Strategy 4: Coordinate Descent Polish ---
    # Final refinement of individual points.
    if time.time() < deadline - 5.0:
        best_pts = _polish_points(best_pts, deadline)
        final_score = _score(best_pts)
        if final_score > best_score:
            best_score = final_score

    return best_pts, best_score

def _polish_points(pts, deadline):
    """
    Refine points using coordinate descent with Nelder-Mead on small neighborhoods.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Try small perturbations for each point
    for _ in range(5): # Limited iterations to save time
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
                
            # Define a function to optimize point i
            def obj_i(x):
                trial_pts = current_pts.copy()
                trial_pts[i] = x
                return -_score(trial_pts)
                
            try:
                res = minimize(
                    obj_i, 
                    current_pts[i], 
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
                )
                
                new_score = -res.fun
                if new_score > current_score + 1e-10:
                    current_pts[i] = res.x
                    current_score = new_score
                    improved = True
            except Exception:
                pass
                
        if not improved:
            break
            
    return current_pts

def construct():
    start_time = time.time()
    # 150s limit. Leave 10s buffer.
    deadline = start_time + 140.0
    
    # Run the general optimization
    best_pts, best_score = _optimize_general(deadline)
    
    # Fallback if optimization failed
    if best_pts is None:
        best_pts = _generate_regular_polygon(N)
        
    return best_pts

# EVOLVE-BLOCK-END
