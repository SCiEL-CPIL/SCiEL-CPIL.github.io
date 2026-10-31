"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s hard limit (evaluator kills at 150s)
    
    # Initial guess: grid-like arrangement
    side = int(np.ceil(np.sqrt(N)))
    # Better initial positions: jittered grid
    rng = np.random.default_rng(42)
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            y = (j + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            positions.append([x, y])
    positions = np.array(positions[:N])
    
    # Initial radii: small
    radii = np.full(N, 0.03)
    
    best_x = np.column_stack([positions, radii])
    best_val = np.sum(radii)
    
    def objective(x):
        """Minimize negative sum of radii."""
        r = x[N:]
        return -np.sum(r)
    
    def get_constraints(x, margin=1e-9):
        """
        Constraints:
        1. Circles inside unit square: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
        2. Non-overlap: ||p_i - p_j|| >= r_i + r_j for all i != j
        3. r_i > 0
        """
        pos = x[:2*N].reshape(N, 2)
        r = x[N:]
        
        # Boundary constraints (linear in x)
        # x_i - r_i >= 0
        # -x_i + r_i + 1 >= 0  => x_i - r_i <= 1
        # y_i - r_i >= 0
        # -y_i + r_i + 1 >= 0  => y_i - r_i <= 1
        
        # For each circle i:
        # x_i - r_i >= 0  -> -x_i + r_i <= 0
        # x_i - r_i <= 1  -> x_i - r_i >= -1  -> -x_i + r_i <= 1
        # y_i - r_i >= 0  -> -y_i + r_i <= 0
        # y_i - r_i <= 1  -> y_i - r_i >= -1  -> -y_i + r_i <= 1
        
        # Let's use LinearConstraint for L @ x <= ub and L @ x >= lb
        
        # Boundary: for each i
        # x_i - r_i >= 0  =>  L row: [1, 0, 0, ..., -1, ...]  >= 0
        # x_i - r_i <= 1  =>  L row: [1, 0, 0, ..., -1, ...]  <= 1
        # y_i - r_i >= 0  =>  L row: [0, 1, 0, ..., -1, ...]  >= 0
        # y_i - r_i <= 1  =>  L row: [0, 1, 0, ..., -1, ...]  <= 1
        
        lb_bound = np.zeros(4 * N)
        ub_bound = np.zeros(4 * N)
        L_bound = np.zeros((4 * N, 2 * N + N))
        
        for i in range(N):
            xi = i
            yi = N + i
            ri = 2 * N + i
            # x_i - r_i >= 0
            L_bound[4*i, xi] = 1.0
            L_bound[4*i, ri] = -1.0
            lb_bound[4*i] = 0.0
            ub_bound[4*i] = np.inf
            # x_i - r_i <= 1
            L_bound[4*i+1, xi] = 1.0
            L_bound[4*i+1, ri] = -1.0
            lb_bound[4*i+1] = -np.inf
            ub_bound[4*i+1] = 1.0
            # y_i - r_i >= 0
            L_bound[4*i+2, yi] = 1.0
            L_bound[4*i+2, ri] = -1.0
            lb_bound[4*i+2] = 0.0
            ub_bound[4*i+2] = np.inf
            # y_i - r_i <= 1
            L_bound[4*i+3, yi] = 1.0
            L_bound[4*i+3, ri] = -1.0
            lb_bound[4*i+3] = -np.inf
            ub_bound[4*i+3] = 1.0
        
        # Non-overlap constraints: linearized at current point
        # ||p_i - p_j|| >= r_i + r_j
        # Linearize: (p_i - p_j) . (p_i - p_j) / ||p_i - p_j|| >= (r_i + r_j) * (p_i - p_j) / ||p_i - p_j||
        # Actually, the standard linearization of ||d|| >= r is d . d_hat >= r * ||d_hat|| where d_hat = d/||d||
        # But this is still nonlinear. Instead, use:
        # d . d_hat - r_i * d_hat_x - r_j * d_hat_x - ... 
        # Let d = p_i - p_j, d_hat = d / ||d||
        # Constraint: d . d_hat >= r_i + r_j
        # => d_hat_x * x_i - d_hat_x * x_j + d_hat_y * y_i - d_hat_y * y_j - r_i - r_j >= 0
        # This is linear in x!
        
        # We'll use SLSQP with nonlinear constraints for non-overlap, or use LinearConstraint with linearization.
        # Since the linearization changes each iteration, let's use SLSQP with a constraint function.
        
        return L_bound, lb_bound, ub_bound
    
    # Use SLSQP with nonlinear constraints for non-overlap
    def non_overlap_constraints(x):
        """Returns array of constraint values, all must be >= 0."""
        pos = x[:2*N].reshape(N, 2)
        r = x[N:]
        vals = []
        for i in range(N):
            for j in range(i+1, N):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Degenerate: push apart
                    vals.append(-1.0)
                else:
                    vals.append(dist - r[i] - r[j])
        return np.array(vals)
    
    def non_overlap_jacobian(x):
        """Jacobian of non_overlap_constraints."""
        pos = x[:2*N].reshape(N, 2)
        r = x[N:]
        J = np.zeros((N*(N-1)//2, 2*N + N))
        k = 0
        for i in range(N):
            for j in range(i+1, N):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    dist = 1e-12
                d_hat = d / dist
                # d = x_i - x_j where x_i = (pos[i][0], pos[i][1])
                # d_hat_x * x_i[0] - d_hat_x * x_j[0] + d_hat_y * x_i[1] - d_hat_y * x_j[1] - r_i - r_j
                J[k, i] = d_hat[0]
                J[k, N + i] = d_hat[1]
                J[k, j] = -d_hat[0]
                J[k, N + j] = -d_hat[1]
                J[k, 2*N + i] = -1.0
                J[k, 2*N + j] = -1.0
                k += 1
        return J
    
    # Bounds: positions in [0,1], radii in [0, 0.5]
    bounds = [(0, 1)] * (2 * N) + [(1e-6, 0.5)] * N
    
    # Iterative refinement
    max_iterations = 50
    for it in range(max_iterations):
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(
                objective,
                x0=best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': non_overlap_constraints, 'jac': non_overlap_jacobian}
                ],
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.success:
                new_val = -res.fun
                if new_val > best_val:
                    best_x = res.x.copy()
                    best_val = new_val
            else:
                # Try with a slightly relaxed constraint
                # Or just accept if it's feasible
                pos = res.x[:2*N].reshape(N, 2)
                r = res.x[N:]
                # Check feasibility
                feasible = True
                for i in range(N):
                    if pos[i][0] < r[i] - 1e-6 or pos[i][0] > 1 - r[i] + 1e-6 or \
                       pos[i][1] < r[i] - 1e-6 or pos[i][1] > 1 - r[i] + 1e-6:
                        feasible = False
                        break
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = pos[i] - pos[j]
                            if np.linalg.norm(d) < r[i] + r[j] - 1e-6:
                                feasible = False
                                break
                        if not feasible:
                            break
                    if feasible:
                        new_val = np.sum(r)
                        if new_val > best_val:
                            best_x = res.x.copy()
                            best_val = new_val
        except Exception:
            pass
    
    # Final polish: try to improve with more iterations
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective,
                x0=best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': non_overlap_constraints, 'jac': non_overlap_jacobian}
                ],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success:
                new_val = -res.fun
                if new_val > best_val:
                    best_x = res.x.copy()
                    best_val = new_val
        except Exception:
            pass
    
    # Ensure final result is valid
    pos = best_x[:2*N].reshape(N, 2)
    r = best_x[N:]
    
    # Clamp to be safe
    for i in range(N):
        r[i] = max(r[i], 1e-6)
        pos[i][0] = np.clip(pos[i][0], r[i], 1 - r[i])
        pos[i][1] = np.clip(pos[i][1], r[i], 1 - r[i])
    
    # Final check for overlaps; if any, reduce radii slightly
    for i in range(N):
        for j in range(i+1, N):
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            required = r[i] + r[j]
            if dist < required:
                # Scale down both radii
                scale = dist / required * 0.999
                r[i] *= scale
                r[j] *= scale
                pos[i][0] = np.clip(pos[i][0], r[i], 1 - r[i])
                pos[i][1] = np.clip(pos[i][1], r[i], 1 - r[i])
                pos[j][0] = np.clip(pos[j][0], r[j], 1 - r[j])
                pos[j][1] = np.clip(pos[j][1], r[j], 1 - r[j])
    
    result = np.column_stack([pos, r])
    return result

# EVOLVE-BLOCK-END
