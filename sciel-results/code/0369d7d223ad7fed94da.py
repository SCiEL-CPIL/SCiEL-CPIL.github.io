import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    num_pairs = len(pairs)
    
    def solve_qp_subproblem(x, y, r, max_iter=50):
        """
        Solves a local QP approximation of the packing problem.
        Variables: x, y, r.
        Objective: Maximize sum(r) (Minimize -sum(r)).
        Constraints:
        1. Boundary: x_i >= r_i, x_i + r_i <= 1, y_i >= r_i, y_i + r_i <= 1
        2. Pairwise (Linearized): 
           dist_ij >= r_i + r_j
           Approximation: 
           Let d_ij = ||p_i - p_j||.
           Constraint: d_ij - (r_i + r_j) >= 0
           Gradient of d_ij w.r.t p_i is (p_i - p_j)/d_ij.
           We enforce: (p_i - p_j) . (p_i - p_j)/d_ij_old - (r_i + r_j) + d_ij_old >= 0?
           Actually, a standard linearization of ||u|| >= v around u0 is:
           u0 . (u - u0) / ||u0|| + ||u0|| >= v  => u0 . u / ||u0|| >= v - ||u0|| + u0 . u0 / ||u0|| = v
           So: (p_i - p_j)_old . (p_i - p_j) / d_old >= r_i + r_j
           This is a linear constraint in x, y, r.
        """
        # Flatten variables for scipy
        z0 = np.concatenate([x, y, r])
        
        # Objective: -sum(r)
        def objective(z):
            return -np.sum(z[2*N:])
            
        def gradient(z):
            g = np.zeros(3*N)
            g[2*N:] = -1.0
            return g
            
        # Hessian: 0 (Linear objective)
        # But we need bounds and constraints.
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
        
        # Constraints
        # 1. Boundary constraints (Linear)
        # x_i - r_i >= 0  => -x_i + r_i <= 0
        # -x_i - r_i + 1 >= 0 => x_i + r_i <= 1
        # -y_i + r_i <= 0
        # y_i + r_i <= 1
        
        # 2. Pairwise constraints (Linearized)
        # For each pair (i, j):
        # (x_i - x_j)_old * (x_i - x_j) + (y_i - y_j)_old * (y_i - y_j) - d_old * (r_i + r_j) >= 0
        # Let A be the matrix for the linear part in x, y.
        # Let B be the matrix for r.
        # Constraint: A_pair @ z_xy + B_pair @ z_r >= 0
        
        cons = []
        
        # Add boundary constraints as LinearConstraint objects
        # We can combine all linear constraints into one LinearConstraint for efficiency
        # Ax + b >= 0
        
        # Number of constraints: 4*N (boundary) + num_pairs (pairwise)
        num_cons = 4 * N + num_pairs
        
        A_matrix = np.zeros((num_cons, 3 * N))
        b_vector = np.zeros(num_cons)
        
        # Boundary Constraints
        # 1. x_i - r_i >= 0
        for i in range(N):
            A_matrix[i, i] = 1.0           # x_i
            A_matrix[i, 2*N + i] = -1.0    # -r_i
            b_vector[i] = 0.0
            
        # 2. -x_i - r_i + 1 >= 0  => -x_i - r_i >= -1
        for i in range(N):
            idx = N + i
            A_matrix[idx, i] = -1.0        # -x_i
            A_matrix[idx, 2*N + i] = -1.0  # -r_i
            b_vector[idx] = -1.0
            
        # 3. y_i - r_i >= 0
        for i in range(N):
            idx = 2*N + i
            A_matrix[idx, N + i] = 1.0     # y_i
            A_matrix[idx, 2*N + i] = -1.0  # -r_i
            b_vector[idx] = 0.0
            
        # 4. -y_i - r_i + 1 >= 0 => -y_i - r_i >= -1
        for i in range(N):
            idx = 3*N + i
            A_matrix[idx, N + i] = -1.0    # -y_i
            A_matrix[idx, 2*N + i] = -1.0  # -r_i
            b_vector[idx] = -1.0
            
        # Pairwise Constraints (Linearized)
        # Constraint k: (x_i - x_j)_old * (x_i - x_j) + (y_i - y_j)_old * (y_i - y_j) - d_old * (r_i + r_j) >= 0
        # Let dx_old = x[i] - x[j], dy_old = y[i] - y[j]
        # d_old = sqrt(dx_old^2 + dy_old^2)
        # dx_old * x_i - dx_old * x_j + dy_old * y_i - dy_old * y_j - d_old * r_i - d_old * r_j >= 0
        
        for k in range(num_pairs):
            i = i_idx[k]
            j = j_idx[k]
            
            dx_old = x[i] - x[j]
            dy_old = y[i] - y[j]
            d_old = np.sqrt(dx_old*dx_old + dy_old*dy_old)
            
            if d_old < 1e-8:
                # Handle singularity: if circles are at same center, push apart arbitrarily?
                # Or just skip? Better to add a small regularization or skip.
                # If they are exactly on top, the linearization is ill-defined.
                # We'll add a constraint that they move apart in x direction.
                # x_i - x_j >= 0 (arbitrary direction)
                idx = 4*N + k
                A_matrix[idx, i] = 1.0
                A_matrix[idx, j] = -1.0
                b_vector[idx] = 0.0
                continue
            
            idx = 4*N + k
            
            # Coefficients for x
            A_matrix[idx, i] = dx_old
            A_matrix[idx, j] = -dx_old
            
            # Coefficients for y
            A_matrix[idx, N + i] = dy_old
            A_matrix[idx, N + j] = -dy_old
            
            # Coefficients for r
            A_matrix[idx, 2*N + i] = -d_old
            A_matrix[idx, 2*N + j] = -d_old
            
            b_vector[idx] = 0.0
            
        # Define LinearConstraint
        linear_constraint = {'type': 'ineq', 'fun': lambda z: A_matrix @ z + b_vector}
        
        # Trust Region Method requires Hessian or Hessian approximation.
        # Objective is linear, so Hessian is 0. But TR methods work better with some curvature or bounds.
        # We rely on the bounds and linear constraints.
        
        try:
            res = minimize(
                objective, 
                z0, 
                jac=gradient,
                method='trust-constr',
                bounds=bounds,
                constraints=[linear_constraint],
                options={
                    'maxiter': max_iter,
                    'gtol': 1e-6,
                    'verbose': 0
                }
            )
            return res.x
        except Exception:
            return z0

    def project_valid(x, y, r):
        """
        Ensures validity by shrinking radii if necessary.
        """
        # Boundary
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_bound)
        
        # Pairwise
        for _ in range(50):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            if np.max(violations) < 1e-9:
                break
            # Shrink all radii slightly
            r *= 0.95
            r = np.minimum(r, max_r_bound)
            
        return x, y, r

    best_global_state = None
    best_global_score = 0.0
    
    # Multi-Start Configurations
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 5.0)
    
    for k in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        start_k = time.time()
        
        # Generate Initial State z0 (shape 3N)
        if k % 2 == 0:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x[count] = (i + 0.5) / side
                        y[count] = (j + 0.5) / side
                        r[count] = 0.05
                        count += 1
        else:
            # Random
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.05)
            
        # Add noise
        x += np.random.normal(0, 0.005, N)
        y += np.random.normal(0, 0.005, N)
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        # Initial projection to validity
        x, y, r = project_valid(x, y, r)
        
        current_score = np.sum(r)
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_state = np.concatenate([x, y, r])
            
        # Iterative QP Refinement
        # We run the QP solver multiple times, each time using the result as the new center for linearization.
        # This effectively follows the solution path of the non-convex problem.
        for _ in range(20): # Max 20 QP steps per start
            if time.time() > deadline - 5.0:
                break
            
            z_new = solve_qp_subproblem(x, y, r, max_iter=50)
            
            x_new = z_new[:N]
            y_new = z_new[N:2*N]
            r_new = z_new[2*N:]
            
            # Check if we made progress
            new_score = np.sum(r_new)
            
            # Project to validity just in case TR method violated constraints slightly
            x_new, y_new, r_new = project_valid(x_new, y_new, r_new)
            final_score = np.sum(r_new)
            
            if final_score > current_score:
                x, y, r = x_new, y_new, r_new
                current_score = final_score
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_state = np.concatenate([x, y, r])
            else:
                # If no progress, break to save time for next start
                break

    # Fallback
    if best_global_state is None:
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        best_global_state = np.concatenate([x, y, r])
        best_global_score = np.sum(r)

    # Final safety projection
    x = best_global_state[:N]
    y = best_global_state[N:2*N]
    r = best_global_state[2*N:]
    
    x, y, r = project_valid(x, y, r)
    
    final_state = np.column_stack([x, y, r])
    
    return final_state

# EVOLVE-BLOCK-END
