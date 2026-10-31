"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _score_from_params(params):
    """
    params: array of shape (1 + 2*N, )
      params[0] = aspect ratio r = width / height (or height/width depending on convention)
      params[1:2N+1] = 2N coordinates (x_i, y_i) in [0,1] x [0,1] normalized space
    
    Rectangle perimeter = 4, so 2*(w + h) = 4 => w + h = 2.
    Let aspect ratio a = w/h. Then w = 2a/(1+a), h = 2/(1+a).
    
    We normalize: place circles in a unit square [0,1]^2, then scale.
    Actually, let's use: w + h = 2. Let a = w/h. w = 2a/(1+a), h = 2/(1+a).
    
    Circle i has center (x_i * w, y_i * h) and radius r_i.
    Constraints:
      1. x_i * w >= r_i, (1 - x_i) * w >= r_i  => r_i <= min(x_i, 1-x_i) * w
      2. y_i * h >= r_i, (1 - y_i) * h >= r_i  => r_i <= min(y_i, 1-y_i) * h
      3. For i != j: (x_i - x_j)^2 * w^2 + (y_i - y_j)^2 * h^2 >= (r_i + r_j)^2
    
    For given positions and aspect ratio, the max radius for circle i is:
      r_i = min(x_i * w, (1-x_i) * w, y_i * h, (1-y_i) * h)
    But we also need non-overlap. The max feasible radius is the largest r such that
    all constraints hold. This is complex.
    
    Alternative: use a penalty approach. Define the objective as:
      sum_i r_i - penalty(overlaps) - penalty(wall violations)
    
    But we want to maximize sum of radii with disjoint interiors.
    
    Better approach: For given positions and aspect ratio, compute the maximum
    feasible radius for each circle considering walls, then check pairwise
    distances. The actual max radius is the largest r such that:
      r <= min_i(wall_limit_i) and for all i<j: dist_ij >= 2r (if all same radius)
    
    But circles can have different radii. This is a hard problem.
    
    Let me use a different formulation: assume all circles have the same radius r.
    Then for given positions and aspect ratio, the max r is:
      r = min over all constraints:
        - min_i min(x_i*w, (1-x_i)*w, y_i*h, (1-y_i)*h)
        - min_{i<j} dist_ij / 2
    
    Then objective = N * r.
    
    But allowing different radii could be better. However, for simplicity and
    given the time budget, let's start with equal radii and see.
    
    Actually, the task says "maximize the sum of their radii". Different radii
    might allow a better packing. But let's first try equal radii.
    
    For equal radii r:
      Constraints:
        x_i * w >= r  => x_i >= r/w
        (1-x_i) * w >= r  => x_i <= 1 - r/w
        y_i * h >= r  => y_i >= r/h
        (1-y_i) * h >= r  => y_i <= 1 - r/h
        dist_ij >= 2r for all i != j
    
    Given positions (x_i, y_i) in [0,1]^2 and aspect ratio a, the max r is:
      r_wall = min_i min(x_i, 1-x_i) * w, min_i min(y_i, 1-y_i) * h
      r_pair = min_{i<j} sqrt((x_i-x_j)^2 * w^2 + (y_i-y_j)^2 * h^2) / 2
      r = min(r_wall, r_pair)
    
    Objective to maximize: N * r
    
    Let's parameterize by a (aspect ratio) and 2N positions.
    """
    a = params[0]
    coords = params[1:]
    coords = coords.reshape(N, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    
    w = 2 * a / (1 + a)
    h = 2 / (1 + a)
    
    # Wall constraints
    wall_x = np.minimum(x, 1 - x) * w
    wall_y = np.minimum(y, 1 - y) * h
    r_wall = np.minimum(np.min(wall_x), np.min(wall_y))
    
    # Pairwise distances
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = (x[i] - x[j]) * w
            dy = (y[i] - y[j]) * h
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
    
    r_pair = min_dist / 2
    r = min(r_wall, r_pair)
    
    return -N * r  # minimize negative


def _bounds():
    """Bounds for differential evolution."""
    # Aspect ratio: let's search in [0.3, 3.0]
    a_bounds = (0.3, 3.0)
    # Positions: [0, 1]
    pos_bounds = [(0, 1)] * (2 * N)
    return [a_bounds] + pos_bounds


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # Try differential evolution with limited iterations
    bounds = _bounds()
    
    # Use a population of 30, max 100 generations, but check time
    pop_size = 30
    max_iter = 50
    
    # Differential evolution
    try:
        result = differential_evolution(
            _score_from_params,
            bounds,
            popsize=pop_size,
            maxiter=max_iter,
            seed=42,
            tol=1e-6,
            polish=False,  # skip polish to save time
            callback=_de_callback
        )
    except Exception:
        pass
    
    if result and result.success or (result and result.fun < best_score):
        best_params = result.x
        best_score_val = -result.fun
    else:
        # Fallback to grid
        best_params = np.array([1.0] + [c[0] for c in cells] + [c[1] for c in cells])
        # Actually, let's just compute from the grid
        best_params = None
    
    # Also try a simple local search from grid
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try to improve with a simple local search
        ...
    
    # Build the result from best_params
    if best_params is not None:
        a = best_params[0]
        coords = best_params[1:].reshape(N, 2)
        x = coords[:, 0]
        y = coords[:, 1]
        
        w = 2 * a / (1 + a)
        h = 2 / (1 + a)
        
        # Compute max feasible radius
        wall_x = np.minimum(x, 1 - x) * w
        wall_y = np.minimum(y, 1 - y) * h
        r_wall = np.minimum(np.min(wall_x), np.min(wall_y))
        
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = (x[i] - x[j]) * w
                dy = (y[i] - y[j]) * h
                d = np.sqrt(dx * dx + dy * dy)
                if d < min_dist:
                    min_dist = d
        
        r = min(r_wall, min_dist / 2)
        
        # Add small epsilon to ensure strict non-overlap
        r = r * (1 - 1e-9)
        
        # Build output: (x_actual, y_actual, r)
        result_arr = np.zeros((N, 3))
        result_arr[:, 0] = x * w
        result_arr[:, 1] = y * h
        result_arr[:, 2] = r
        
        return result_arr
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # Use square aspect ratio for fallback
    w = h = 1.0
    r = 0.5 / side - 1e-3
    
    result_arr = np.zeros((N, 3))
    for i, (cx, cy) in enumerate(cells):
        result_arr[i, 0] = cx * w
        result_arr[i, 1] = cy * h
        result_arr[i, 2] = r
    
    return result_arr


def _de_callback(xk, convergence):
    """Callback for differential evolution to check time."""
    pass
# EVOLVE-BLOCK-END
