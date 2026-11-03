"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _compute_radii(centers):
    """
    Compute radii for given centers.
    Radius of circle i is the minimum of:
    1. Distance to the four boundaries of the unit square.
    2. Half the distance to the nearest other center (Voronoi edge distance).
    
    This ensures circles are disjoint and inside the square.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries
    dist_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Distance to nearest neighbor (approximated via Voronoi)
    radii = dist_bound.copy()
    
    try:
        vor = Voronoi(centers)
        
        # For each point, find the minimum distance to its Voronoi vertices
        # Or more simply, for each point, find the nearest other point
        # Using Voronoi regions is more robust for "packing" logic, but 
        # nearest neighbor distance / 2 is the strict constraint for disjoint circles.
        # Actually, for disjoint circles, the constraint is dist(i,j) >= r_i + r_j.
        # If we set r_i = min(dist_bound_i, min_j dist(i,j)/2), does that work?
        # Yes, if r_i = min_j dist(i,j)/2, then r_i + r_j <= dist(i,j)/2 + dist(i,j)/2 = dist(i,j).
        # So circles are disjoint.
        
        # Efficient nearest neighbor search using Voronoi or just brute force for N=26
        # N=26 is small, brute force is fast enough and less prone to Voronoi edge cases.
        for i in range(N):
            min_dist = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    min_dist = d
            # The radius is limited by the distance to the nearest neighbor / 2
            # AND the distance to the boundary
            radii[i] = min(dist_bound[i], min_dist / 2.0)
            
    except Exception:
        # Fallback to brute force if Voronoi fails (unlikely for 26 points)
        for i in range(N):
            min_dist = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    min_dist = d
            radii[i] = min(dist_bound[i], min_dist / 2.0)
            
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    return radii


def _objective(params):
    """
    Objective function: Negative sum of radii.
    params: flattened array of shape (2*N,) containing x, y coordinates.
    """
    centers = params.reshape(N, 2)
    radii = _compute_radii(centers)
    return -np.sum(radii)


def _objective_grad(params):
    """
    Analytic gradient of the objective function.
    This is tricky because _compute_radii involves a min() operation which is non-differentiable.
    However, we can use numerical gradients or approximate analytic gradients.
    Given the time limit, numerical gradient via finite differences is safer and likely fast enough
    for 52 dimensions if we use a good step size, or we can use the fact that the active constraint
    (boundary or specific neighbor) determines the gradient.
    
    Let's implement an analytic gradient by determining the active constraint for each circle.
    """
    centers = params.reshape(N, 2)
    x = centers[:, 0]
    y = centers[:, 1]
    
    radii = _compute_radii(centers)
    
    grad = np.zeros_like(params)
    
    # For each circle, determine which constraint is active
    # Constraint 1: Boundary (x, y, 1-x, 1-y)
    # Constraint 2: Nearest neighbor j (dist(i,j)/2)
    
    for i in range(N):
        r_i = radii[i]
        
        # Check boundary distances
        d_left = x[i]
        d_right = 1.0 - x[i]
        d_bottom = y[i]
        d_top = 1.0 - y[i]
        
        # Find nearest neighbor
        min_dist = np.inf
        nearest_j = -1
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
                nearest_j = j
        
        d_nn = min_dist / 2.0
        
        # Determine active constraint
        # If boundary is tighter, gradient is towards center of square (away from boundary)
        # If neighbor is tighter, gradient is away from neighbor
        
        if d_left <= d_right and d_left <= d_bottom and d_left <= d_top and d_left <= d_nn:
            # Active: Left boundary
            # r_i = x_i
            # dr_i/dx_i = 1, dr_i/dy_i = 0
            # Objective is -sum(r_i), so grad_obj = -grad_r
            # d(-r_i)/dx_i = -1
            grad[2*i] = -1.0
            grad[2*i+1] = 0.0
        elif d_right <= d_left and d_right <= d_bottom and d_right <= d_top and d_right <= d_nn:
            # Active: Right boundary
            # r_i = 1 - x_i
            # dr_i/dx_i = -1
            grad[2*i] = 1.0
            grad[2*i+1] = 0.0
        elif d_bottom <= d_left and d_bottom <= d_right and d_bottom <= d_top and d_bottom <= d_nn:
            # Active: Bottom boundary
            # r_i = y_i
            grad[2*i] = 0.0
            grad[2*i+1] = -1.0
        elif d_top <= d_left and d_top <= d_right and d_top <= d_bottom and d_top <= d_nn:
            # Active: Top boundary
            # r_i = 1 - y_i
            grad[2*i] = 0.0
            grad[2*i+1] = 1.0
        elif d_nn < d_left and d_nn < d_right and d_nn < d_bottom and d_nn < d_top:
            # Active: Nearest neighbor
            # r_i = 0.5 * dist(i, j)
            # dist = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
            # dr_i/dx_i = 0.5 * (x_i - x_j) / dist
            # dr_i/dy_i = 0.5 * (y_i - y_j) / dist
            # d(-r_i)/dx_i = -0.5 * (x_i - x_j) / dist
            # d(-r_i)/dy_i = -0.5 * (y_i - y_j) / dist
            
            dx = x[i] - x[nearest_j]
            dy = y[i] - y[nearest_j]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 1e-12:
                factor = -0.5 / dist
                grad[2*i] = factor * dx
                grad[2*i+1] = factor * dy
            else:
                # Degenerate case, push apart slightly
                grad[2*i] = -0.1
                grad[2*i+1] = -0.1
        else:
            # Degenerate or tie, use numerical gradient fallback for this point? 
            # Or just assume boundary if close.
            # For robustness, if no single clear winner, we might be at a "corner" of the constraint space.
            # In such cases, the gradient is not unique. We can pick any subgradient.
            # Let's pick the boundary one as it's usually smoother.
            grad[2*i] = -1.0 if d_left < d_right else 1.0
            grad[2*i+1] = -1.0 if d_bottom < d_top else 1.0
            
    return grad


def _polish_analytic(centers, time_budget=10.0):
    """
    Refine centers using L-BFGS-B with analytic gradient.
    """
    start_time = time.time()
    params = centers.flatten().copy()
    best_params = params.copy()
    best_val = _objective(params)
    
    bounds = [(0, 1) for _ in range(2 * N)]
    
    # L-BFGS-B with analytic gradient
    try:
        result = minimize(
            _objective, 
            best_params, 
            jac=_objective_grad, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        val = _objective(result.x)
        if val < best_val:
            best_params = result.x
            best_val = val
    except Exception:
        pass
    
    # Second pass with tighter tolerance if time permits
    if time.time() - start_time < time_budget:
        try:
            result = minimize(
                _objective, 
                best_params, 
                jac=_objective_grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = _objective(result.x)
            if val < best_val:
                best_params = result.x
                best_val = val
        except Exception:
            pass
            
    return best_params.reshape(N, 2)


def _initial_hex():
    """Hexagonal packing initial config for centers."""
    cols = 5
    rows = 6
    positions = []
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config for centers."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _initial_random(seed):
    """Random initial config for centers."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.1, 0.9, size=(N, 2))
    return positions


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses the Voronoi-derived radius strategy to reduce dimensionality,
    followed by analytic gradient polishing.
    """
    deadline = _start_time + _TIME_LIMIT
    
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(999),
    ]
    
    best_overall = None
    best_score_overall = -np.inf
    
    for i, centers0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time for polishing
        alloc = min(remaining - 2, max(5, 10 - i * 2))
        
        # Phase 1: Analytic Gradient Polish
        centers_polished = _polish_analytic(centers0, time_budget=alloc)
        
        # Calculate radii and score
        radii = _compute_radii(centers_polished)
        score = np.sum(radii)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = centers_polished.copy()
    
    # Final result construction
    if best_overall is not None:
        radii = _compute_radii(best_overall)
        result = np.column_stack([best_overall, radii])
        return result
    else:
        # Fallback: simple grid
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result

# EVOLVE-BLOCK-END
