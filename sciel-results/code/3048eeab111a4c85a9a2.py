"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    """
    Vectorized calculation of areas for all triples of points.
    points: (N, 2) array
    returns: array of shape (C(N,3),) containing areas
    """
    n = len(points)
    # Create indices for all triples i < j < k
    i, j, k = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    i = i[i < j]
    j = j[i < j]
    k = k[i < j]
    k = k[j < k]
    
    # Extract points
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # Cross product (2D)
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    vec1 = pj - pi
    vec2 = pk - pi
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is the area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    areas = _triangle_areas(points)
    min_tri = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_tri / hull_area

def _soft_min(areas, T):
    """
    Soft minimum of areas.
    As T -> 0, soft_min -> min(areas).
    """
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(areas)
    # If T is very small, this is just min_val
    if T < 1e-10:
        return min_val
    # -T * ln(sum(exp(-areas/T)))
    # To avoid overflow in exp, subtract min:
    # exp(-areas/T) = exp(-(areas - min_val)/T) * exp(-min_val/T)
    # sum = exp(-min_val/T) * sum(exp(-(areas - min_val)/T))
    # ln(sum) = -min_val/T + ln(sum(exp(-(areas - min_val)/T)))
    # -T * ln(sum) = min_val - T * ln(sum(exp(-(areas - min_val)/T)))
    
    shifted = (areas - min_val) / T
    # Clip to avoid overflow in exp, though shifted >= 0 so exp(-shifted) <= 1
    # Actually, we want sum(exp(-shifted)). Since shifted >= 0, exp(-shifted) is in (0, 1].
    log_sum = np.log(np.sum(np.exp(-shifted)))
    return min_val - T * log_sum

def _objective_smooth(params, T):
    """
    Objective function for differential_evolution.
    Minimizes -soft_min(areas) / hull_area.
    params: flattened array of 2*N coordinates.
    """
    points = params.reshape(N, 2)
    
    # Constrain to unit disk (soft penalty or hard clip? 
    # DE bounds will handle this if we set bounds correctly, 
    # but let's ensure we don't go out of bounds if we use Cartesian bounds [-1, 1])
    # The DE bounds will be set to [-1, 1] for x and y.
    
    areas = _triangle_areas(points)
    min_area_soft = _soft_min(areas, T)
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0 # Invalid, return 0 (maximize 0) -> minimize 0
    
    score = min_area_soft / hull_area
    return -score

def _simulated_annealing(points, start_temp=0.01, end_temp=1e-6, steps=2000, seed=42):
    """
    Local refinement using Simulated Annealing on the exact score.
    """
    rng = np.random.default_rng(seed)
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for i in range(steps):
        if time.time() - start_time > 10: # Check time periodically
            break
            
        # Annealing schedule
        T = start_temp * (end_temp / start_temp) ** (i / steps)
        
        # Generate perturbation
        # Gaussian noise, scaled by temperature? Or fixed step size?
        # Fixed step size is often better for geometric problems, 
        # but SA accepts worse moves based on T.
        step_size = 0.01 * (1.0 + T) # Slightly larger steps at higher T
        perturbation = rng.normal(0, step_size, size=current_points.shape)
        new_points = current_points + perturbation
        
        # Project back to unit disk if necessary? 
        # The problem says "normalized to unit area", but the evaluator normalizes by hull area.
        # However, the points should likely stay within a reasonable convex region.
        # The seed used a disk. Let's keep points within the unit disk to be safe, 
        # or just let them be, as the score is relative to hull area.
        # If points go to infinity, hull area grows, but so do triangles.
        # However, DE bounds are [-1, 1]. Let's clip to [-1, 1] to match DE space.
        new_points = np.clip(new_points, -1.0, 1.0)
        
        new_score = _score(new_points)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0:
            # Better, always accept
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Worse, accept with probability exp(delta / T)
            if T > 1e-12:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: 130s out of 150s
    
    # Initial guess: Regular 14-gon on unit circle
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    best_score = _score(points_init)
    best_points = points_init.copy()
    
    # Differential Evolution
    # Bounds: x, y in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Use a moderate temperature for soft-min
    T_soft = 0.05
    
    try:
        result = differential_evolution(
            _objective_smooth,
            bounds,
            args=(T_soft,),
            maxiter=100,
            popsize=15,
            tol=1e-5,
            seed=N,
            polish=True, # Let DE do its own polish
            update_interval=10 # Check time more often? DE doesn't have a direct time limit callback, 
                               # but we can wrap the objective if needed. 
                               # However, 100 iterations * 15 pop * 28 dims is manageable.
        )
        
        if result.fun < 0:
            points_de = result.x.reshape(N, 2)
            score_de = _score(points_de)
            if score_de > best_score:
                best_score = score_de
                best_points = points_de
    except Exception:
        pass
    
    # Local Refinement with Simulated Annealing
    if time.time() < deadline - 5:
        # Try SA from the best DE point
        sa_points, sa_score = _simulated_annealing(best_points, start_temp=0.02, end_temp=1e-8, steps=5000, seed=100)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
            
        # Try SA from a few random restarts if time permits
        rng = np.random.default_rng(42)
        restarts = 5
        for i in range(restarts):
            if time.time() > deadline - 5:
                break
            # Random points in unit disk
            r = rng.random(N) * 1.0
            theta = rng.random(N) * 2 * np.pi
            points_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            # Quick SA refinement
            sa_points, sa_score = _simulated_annealing(points_rand, start_temp=0.05, end_temp=1e-8, steps=2000, seed=100+i)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points

    return best_points
# EVOLVE-BLOCK-END
