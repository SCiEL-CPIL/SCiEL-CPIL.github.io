"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def _initial_positions(n):
    """Generate initial positions using random + repulsion relaxation."""
    rng = np.random.default_rng(42)
    # Start with a grid-like perturbation for better initial spread
    side = int(np.ceil(np.sqrt(n)))
    pts = np.zeros((n, 2))
    for i in range(n):
        row = i // side
        col = i % side
        pts[i] = [(col + 0.5) / side, (row + 0.5) / side]
    # Add small random perturbation
    pts += rng.uniform(-0.1, 0.1, size=(n, 2))
    pts = np.clip(pts, 0.05, 0.95)
    
    # Repulsion relaxation: push points apart
    for _ in range(200):
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-6:
                    d = rng.uniform(-0.01, 0.01, 2)
                    dist = np.linalg.norm(d)
                # Repulsive force inversely proportional to distance
                force = 0.01 / (dist ** 2 + 1e-4)
                pts[i] += force * d / dist
                pts[j] -= force * d / dist
        pts = np.clip(pts, 0.05, 0.95)
    
    return pts


def _objective_and_grad(vars_vec, n):
    """
    Variables: x[n], y[n], r[n] -> total 3n
    Objective: maximize sum(r) -> minimize -sum(r)
    Constraints (linearized):
      r_i >= 0
      x_i - r_i >= 0
      x_i + r_i <= 1
      y_i - r_i >= 0
      y_i + r_i <= 1
      For each pair (i,j): (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        Linearize as: 2(x_i-x_j)(x_i-x_j_0) + 2(y_i-y_j)(y_i-y_j_0) >= (r_i+r_j)^2 - (x_i-x_j_0)^2 - (y_i-y_j_0)^2
        This is a tangent plane approximation.
    """
    x = vars_vec[:n]
    y = vars_vec[n:2*n]
    r = vars_vec[2*n:]
    
    # Objective: minimize -sum(r)
    obj = -np.sum(r)
    grad = np.zeros(3 * n)
    grad[2*n:] = -1.0
    
    return obj, grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = N
    best_result = None
    best_score = -np.inf
    
    # Initial positions
    pts = _initial_positions(n)
    
    # Initial radii: small
    r0 = np.full(n, 0.02)
    
    # Build variable vector
    vars_vec = np.concatenate([pts[:, 0], pts[:, 1], r0])
    
    # Precompute pair indices
    pair_i, pair_j = np.triu_indices(n, k=1)
    npairs = len(pair_i)
    
    # We'll use a sequential linearization approach (like SOCP relaxation iterated)
    # Each iteration: linearize the non-convex constraints at current point, solve LP, update
    
    max_iter = 200
    current_vars = vars_vec.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        x = current_vars[:n]
        y = current_vars[n:2*n]
        r = current_vars[2*n:]
        
        # Build linear constraints: A @ vars <= b (scipy uses <= for LinearConstraint)
        # Actually scipy LinearConstraint uses A @ x >= lb and A @ x <= ub
        # Let's use: minimize c^T x subject to lb <= A @ x <= ub
        
        # Objective: minimize -sum(r)
        c = np.zeros(3 * n)
        c[2*n:] = -1.0
        
        # Constraints:
        # 1) r_i >= 0: A_row = [0,...,1,...,0], lb=0, ub=inf
        # 2) x_i - r_i >= 0
        # 3) x_i + r_i <= 1 -> -x_i - r_i >= -1
        # 4) y_i - r_i >= 0
        # 5) y_i + r_i <= 1 -> -y_i - r_i >= -1
        # 6) Pair constraints (linearized)
        
        # Count constraints:
        # n (r>=0) + n (x-r>=0) + n (-x-r>=-1) + n (y-r>=0) + n (-y-r>=-1) + npairs
        n_constraints = 5 * n + npairs
        
        A = np.zeros((n_constraints, 3 * n))
        lb = np.full(n_constraints, -np.inf)
        ub = np.full(n_constraints, np.inf)
        
        row = 0
        # r_i >= 0
        for i in range(n):
            A[row, 2*n + i] = 1.0
            lb[row] = 0.0
            row += 1
        
        # x_i - r_i >= 0
        for i in range(n):
            A[row, i] = 1.0
            A[row, 2*n + i] = -1.0
            lb[row] = 0.0
            row += 1
        
        # -x_i - r_i >= -1
        for i in range(n):
            A[row, i] = -1.0
            A[row, 2*n + i] = -1.0
            lb[row] = -1.0
            row += 1
        
        # y_i - r_i >= 0
        for i in range(n):
            A[row, n + i] = 1.0
            A[row, 2*n + i] = -1.0
            lb[row] = 0.0
            row += 1
        
        # -y_i - r_i >= -1
        for i in range(n):
            A[row, n + i] = -1.0
            A[row, 2*n + i] = -1.0
            lb[row] = -1.0
            row += 1
        
        # Pair constraints (linearized at current point)
        for k in range(npairs):
            i = pair_i[k]
            j = pair_j[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-12:
                dx = 1.0
                dy = 0.0
                dist_sq = 1.0
            dist = np.sqrt(dist_sq)
            
            # Linearized constraint: 2*dx*(x_i - x_j) + 2*dy*(y_i - y_j) >= (r_i + r_j)^2 - dist_sq + 2*dist*(r_i + r_j - dist)
            # Simpler: tangent plane to the cone. 
            # The constraint (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
            # At point (x0, y0, r0), the linearization is:
            # 2*dx0*(x_i - x_j) + 2*dy0*(y_i - y_j) - 2*dist0*(r_i + r_j) >= dist0^2 - 2*dist0^2 = -dist0^2
            # Wait, let me redo. Let f = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
            # grad_x = 2*(x_i-x_j), grad_y = 2*(y_i-y_j), grad_r_i = -2*(r_i+r_j), grad_r_j = -2*(r_i+r_j)
            # Linearization: 2*dx0*(x_i-x_j) + 2*dy0*(y_i-y_j) - 2*dist0*(r_i+r_j) >= f(x0) + 2*dx0*dx0 + 2*dy0*dy0 - 2*dist0*dist0
            # = 2*dist0^2 - 2*dist0^2 = 0... hmm that's not right.
            
            # Let me use a simpler approach: just enforce dist >= r_i + r_j using linearization
            # 2*dx*(x_i - x_j) + 2*dy*(y_i - y_j) >= 2*dist*(r_i + r_j)
            # This is the support function of the second-order cone.
            
            A[row, i] = 2.0 * dx
            A[row, n + i] = 2.0 * dy
            A[row, 2*n + i] = -2.0 * dist
            A[row, j] = -2.0 * dx
            A[row, n + j] = -2.0 * dy
            A[row, 2*n + j] = -2.0 * dist
            lb[row] = 0.0
            row += 1
        
        # Also bound variables to keep them in reasonable range
        # x, y in [0, 1], r in [0, 0.5]
        # Add bounds via constraints
        
        # Solve LP
        try:
            res = minimize(
                fun=lambda v: c @ v,
                x0=current_vars,
                jac=lambda v: c,
                method='SLSQP',
                bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                constraints=[
                    LinearConstraint(A, lb, ub)
                ],
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if res.success:
                new_vars = res.x
                # Clip to valid range
                new_vars[:2*n] = np.clip(new_vars[:2*n], 0, 1)
                new_vars[2*n:] = np.clip(new_vars[2*n:], 0, 0.5)
                
                # Check feasibility of original (nonlinear) constraints
                new_x = new_vars[:n]
                new_y = new_vars[n:2*n]
                new_r = new_vars[2*n:]
                
                feasible = True
                # Check boundaries
                if np.any(new_x < new_r - 1e-9) or np.any(new_x + new_r > 1 + 1e-9) or \
                   np.any(new_y < new_r - 1e-9) or np.any(new_y + new_r > 1 + 1e-9):
                    feasible = False
                # Check pairwise
                for k in range(npairs):
                    i = pair_i[k]
                    j = pair_j[k]
                    dx = new_x[i] - new_x[j]
                    dy = new_y[i] - new_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_r[i] + new_r[j] - 1e-7:
                        feasible = False
                        break
                
                if feasible:
                    score = np.sum(new_r)
                    if score > best_score:
                        best_score = score
                        best_result = new_vars.copy()
                
                # Damping for stability
                current_vars = 0.3 * current_vars + 0.7 * new_vars
            else:
                # Try a small step
                break
                
        except Exception:
            break
    
    # Final result
    if best_result is not None:
        x = best_result[:n]
        y = best_result[n:2*n]
        r = best_result[2*n:]
        return np.column_stack([x, y, r])
    else:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
