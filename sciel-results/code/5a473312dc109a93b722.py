"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START

def _pairwise_indices(n):
    """Return indices for pairwise constraints."""
    i_idx = []
    j_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx, dtype=int), np.array(j_idx, dtype=int)

def _compute_score(circles):
    """Compute the objective: sum of radii, with validity check."""
    if len(circles) != N:
        return -np.inf
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r < 0):
        return -np.inf
    
    W = max(x) + max(r) - min(x) - min(r)
    H = max(y) + max(r) - min(y) - min(r)
    
    if W + H > 2.0 + 1e-9:
        return -np.inf
    
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf
    
    return np.sum(r)

def _objective_and_grad(circles):
    """
    Objective: minimize -sum(r)
    With constraints handled by the optimizer.
    """
    r = circles[:, 2]
    return -np.sum(r), -np.concatenate([np.zeros(2 * N), np.ones(N)])

def _make_constraints():
    """
    Constraints:
    1. Perimeter: W + H <= 2 (using auxiliary variables W, H)
       We handle this by projecting or using a different formulation.
       
    Alternative: Direct constraints on x, y, r.
    To make it tractable, we'll use a fixed bounding box approach first,
    or use the fact that we can shift/scale.
    
    Let's use a simpler approach: 
    Fix the bounding box to [0, W] x [0, H] where W + H <= 2.
    But W and H are determined by the circles.
    
    Better approach for optimization:
    Use variables: x_i, y_i, r_i for i in 1..N.
    Constraints:
    - r_i >= 0
    - x_i + r_i <= x_j + r_j + (x_j - x_i) ... no, this is getting complicated.
    
    Let's use a different strategy:
    1. Start with a good initial guess (grid-based).
    2. Use scipy.optimize with constraints:
       - Disjointness: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
       - Perimeter: (max(x) + max(r) - min(x) - min(r)) + (max(y) + max(r) - min(y) - min(r)) <= 2
    
    The non-smooth max/min makes this hard.
    
    Alternative: Assume the bounding box is [0, W] x [0, H] with W + H <= 2.
    Then constraints are:
    - r_i <= x_i <= W - r_i
    - r_i <= y_i <= H - r_i
    - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    We can optimize over x, y, r, W, H with W + H <= 2.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: grid-based
    side = int(np.ceil(np.sqrt(N)))
    # For N=21, side=5, so 5x5 grid, use 21 cells
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([0.5, 0.5, 0.4])  # placeholder
    # Better initial guess: uniform grid in [0,1]x[0,1]
    # But perimeter of [0,1]x[0,1] is 4, so W=1, H=1, W+H=2. OK.
    # Grid spacing: 1/side
    # Radius: 0.5/side - epsilon
    eps = 1e-4
    x0 = np.zeros(3 * N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < N:
                x0[idx * 3] = (i + 0.5) / side
                x0[idx * 3 + 1] = (j + 0.5) / side
                x0[idx * 3 + 2] = 0.5 / side - eps
                idx += 1
    # Fill remaining with small circles at corners
    while idx < N:
        x0[idx * 3] = 0.1
        x0[idx * 3 + 1] = 0.1
        x0[idx * 3 + 2] = 0.01
        idx += 1
    
    best_circles = x0.reshape(N, 3).copy()
    best_score = _compute_score(best_circles)
    
    # Try multiple random restarts with perturbation
    rng = np.random.default_rng(42)
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb initial guess
        x_init = best_circles.copy()
        if restart > 0:
            # Add random perturbation
            x_init[:, 0] += rng.normal(0, 0.05, N)
            x_init[:, 1] += rng.normal(0, 0.05, N)
            x_init[:, 2] *= (1 + rng.normal(0, 0.1, N))
            x_init[:, 2] = np.maximum(x_init[:, 2], 0.01)
        
        # Use scipy.optimize with SLSQP
        # Variables: x_1, y_1, r_1, ..., x_N, y_N, r_N
        # Objective: minimize -sum(r)
        
        def obj_and_grad(v):
            v = v.reshape(N, 3)
            r = v[:, 2]
            val = -np.sum(r)
            grad = np.zeros_like(v)
            grad[:, 2] = -1.0
            return val, grad.flatten()
        
        # Constraints:
        # 1. Disjointness: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        #    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # 2. Perimeter: W + H <= 2 where W = max(x) + max(r) - min(x) - min(r), etc.
        #    This is non-smooth. Approximate by assuming bounding box is fixed?
        
        # Alternative: Use a penalty method or just optimize positions given a fixed box.
        # Let's fix the bounding box to [0, 1] x [0, 1] for now (perimeter = 4, so W+H=2).
        # Constraints:
        # - r_i <= x_i <= 1 - r_i
        # - r_i <= y_i <= 1 - r_i
        # - Disjointness
        
        # This is a non-convex problem due to disjointness.
        # Use SLSQP with these constraints.
        
        i_idx, j_idx = _pairwise_indices(N)
        n_pairs = len(i_idx)
        
        def constraints(v):
            v = v.reshape(N, 3)
            x = v[:, 0]
            y = v[:, 1]
            r = v[:, 2]
            
            # Disjointness constraints
            cons = []
            for k in range(n_pairs):
                i = i_idx[k]
                j = j_idx[k]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                sum_r = r[i] + r[j]
                cons.append(dist_sq - sum_r * sum_r)
            
            # Box constraints: r_i <= x_i <= 1 - r_i
            # => x_i - r_i >= 0 and 1 - x_i - r_i >= 0
            for i in range(N):
                cons.append(x[i] - r[i])
                cons.append(1.0 - x[i] - r[i])
                cons.append(y[i] - r[i])
                cons.append(1.0 - y[i] - r[i])
            
            # r_i >= 0
            for i in range(N):
                cons.append(r[i])
            
            return np.array(cons)
        
        try:
            result = minimize(
                obj_and_grad,
                x_init.flatten(),
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': constraints},
                bounds=[(0, 1)] * 2 * N + [(0, 0.5)] * N,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                new_circles = result.x.reshape(N, 3)
                new_score = _compute_score(new_circles)
                if new_score > best_score:
                    best_score = new_score
                    best_circles = new_circles.copy()
        except Exception as e:
            pass
    
    # Final polish: try to adjust the bounding box
    # Compute current W, H
    x = best_circles[:, 0]
    y = best_circles[:, 1]
    r = best_circles[:, 2]
    W = max(x) + max(r) - min(x) - min(r)
    H = max(y) + max(r) - min(y) - min(r)
    
    if W + H < 2.0 - 1e-6:
        # We can scale up
        scale = 2.0 / (W + H)
        # But scaling changes perimeter to 4, which is allowed (<= 4)
        # Actually, perimeter = 2*(W+H), so if W+H < 2, perimeter < 4.
        # We can scale to make W+H = 2.
        # But scaling also scales radii, so sum(r) increases.
        cx = (max(x) + min(x)) / 2
        cy = (max(y) + min(y)) / 2
        best_circles[:, 0] = (best_circles[:, 0] - cx) * scale + cx
        best_circles[:, 1] = (best_circles[:, 1] - cy) * scale + cy
        best_circles[:, 2] = best_circles[:, 2] * scale
    
    # Verify and return
    final_score = _compute_score(best_circles)
    if final_score < 0:
        # Fallback to grid
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)
    
    return best_circles

# EVOLVE-BLOCK-END
