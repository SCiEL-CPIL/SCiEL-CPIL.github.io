# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Calculate hull area using shoelace formula
    # Assumes points are ordered around the perimeter (which they are in our parametrization)
    x = points[:, 0]
    y = points[:, 1]
    
    x_shifted = np.roll(x, -1)
    y_shifted = np.roll(y, -1)
    
    area_sum = np.sum(x * y_shifted - x_shifted * y)
    A_hull = 0.5 * np.abs(area_sum)
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    
    Strategy:
    1. Parametrize points as a regular 13-gon with radial perturbations:
       P_i = (1 + eps * u_i) * (cos(theta_i), sin(theta_i))
       where theta_i = 2*pi*i/13.
    2. This guarantees convexity (for small enough eps) and fixes the angular structure.
    3. The convex hull area is determined by the radii.
    4. We use a deterministic coordinate ascent / local search on the 13 radial offsets u_i.
       We aim to maximize A_min / A_hull.
    5. Since the regular polygon is a strong local optimum for "uniform" distribution,
       we expect small perturbations to help equalize the smallest triangle areas.
    
    The key insight from the directive is to change the representation. 
    The current program searches angular gaps. This program searches radial deviations 
    from a regular polygon. This is a different manifold.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    n_points = 13
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Initial state: Regular polygon (all radii 1)
    # We represent the state as the vector of radii R = [1, 1, ..., 1]
    # Or better, offsets from 1: U = [0, 0, ..., 0]
    # R_i = 1 + U_i
    
    # To ensure convexity, we need R_i > 0 and the points to be in convex position.
    # For a regular polygon with small radial perturbations, convexity is preserved.
    # Let's constrain U_i to be in [-0.1, 0.1] to be safe.
    
    best_u = np.zeros(n_points)
    best_score = _compute_score(np.column_stack([
        (1 + best_u) * np.cos(angles), 
        (1 + best_u) * np.sin(angles)
    ]))
    
    current_u = best_u.copy()
    current_score = best_score
    
    # Optimization: Coordinate Ascent with Adaptive Step Size
    # We will iterate through each coordinate (radius) and try to adjust it 
    # to improve the score.
    
    # Step size for radial perturbation
    # Radii are around 1. A change of 0.01 is ~1% change in distance.
    step_size = 0.01
    
    # We will run this for a fixed number of iterations or until time limit.
    # Coordinate ascent can be slow if step size is fixed. 
    # Let's use a simple local search: 
    # 1. For each i, try u_i + step, u_i - step.
    # 2. Accept if better.
    # 3. If no improvement in a full pass, reduce step size.
    # 4. If step size too small, try a larger random jump (restart-like).
    
    max_passes = 1000
    step_reduction = 0.9
    min_step = 1e-5
    
    for pass_num in range(max_passes):
        if time.time() - start_time > deadline - safety_margin:
            break
            
        improved = False
        for i in range(n_points):
            if time.time() - start_time > deadline - safety_margin:
                break
                
            # Try positive step
            trial_u = current_u.copy()
            trial_u[i] += step_size
            # Clamp to valid range to ensure convexity/stability
            # If radius becomes <= 0, it's invalid.
            if trial_u[i] > -0.9: 
                trial_points = np.column_stack([
                    (1 + trial_u) * np.cos(angles), 
                    (1 + trial_u) * np.sin(angles)
                ])
                trial_score = _compute_score(trial_points)
                
                if trial_score > current_score:
                    current_u = trial_u
                    current_score = trial_score
                    improved = True
                    # Optionally, keep going in the same direction?
                    # Simple coordinate ascent usually does one step then moves to next.
                    # But we can do a few greedy steps.
                    # Let's stick to one step for simplicity and stability.
                    continue # Skip negative step if positive worked? No, maybe negative is better?
                    # Actually, if we moved in +, maybe - is not better than current. 
                    # But it's possible the optimum is further +. 
                    # Standard CA: try +, if better accept. Try -, if better accept (from new state).
                    
            # Try negative step
            trial_u = current_u.copy()
            trial_u[i] -= step_size
            if trial_u[i] > -0.9:
                trial_points = np.column_stack([
                    (1 + trial_u) * np.cos(angles), 
                    (1 + trial_u) * np.sin(angles)
                ])
                trial_score = _compute_score(trial_points)
                
                if trial_score > current_score:
                    current_u = trial_u
                    current_score = trial_score
                    improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_u = current_u.copy()
            
        if not improved:
            # No improvement in this pass, reduce step size
            step_size *= step_reduction
            if step_size < min_step:
                # Try a random restart with larger step?
                # Or just break if we've converged.
                # Let's try a few random restarts to escape local optima.
                # Randomize current_u within a small range
                current_u = np.random.normal(0, 0.05, n_points)
                current_u = np.clip(current_u, -0.5, 0.5)
                current_score = _compute_score(np.column_stack([
                    (1 + current_u) * np.cos(angles), 
                    (1 + current_u) * np.sin(angles)
                ]))
                if current_score > best_score:
                    best_score = current_score
                    best_u = current_u.copy()
                
                step_size = 0.05 # Reset step size for the new region
                # Limit restarts to avoid infinite loop of bad restarts
                # We rely on the pass counter and time limit.
                # To prevent infinite loop, we can just break if step_size is tiny and no improvement
                # But the restart resets step_size. 
                # Let's add a restart counter.
                # For simplicity, we will just let the time limit handle it.
                # However, if we restart too often, we waste time.
                # Let's limit restarts.
                if pass_num > 50:
                     break

    # Final points
    final_u = best_u
    points = np.column_stack([
        (1 + final_u) * np.cos(angles), 
        (1 + final_u) * np.sin(angles)
    ])
    
    # Final sanity check
    if not np.all(np.isfinite(points)):
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return points

# EVOLVE-BLOCK-END
