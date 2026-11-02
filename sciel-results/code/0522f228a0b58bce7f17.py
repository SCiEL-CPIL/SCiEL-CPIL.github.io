"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Number of restarts to explore different local optima
    num_restarts = 10
    
    # Define constraints for SLSQP
    def boundary_constraints(i):
        # Returns list of constraint dicts for circle i
        cons = []
        # x_i - r_i >= 0  => r_i - x_i <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*i + 2]})
        # 1 - x_i - r_i >= 0 => x_i + r_i - 1 <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i + 2]})
        # y_i - r_i >= 0 => r_i - y_i <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i + 1] - x[2*i + 2]})
        # 1 - y_i - r_i >= 0 => y_i + r_i - 1 <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i + 1] - x[2*i + 2]})
        return cons

    def pairwise_constraints():
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                # SLSQP ineq constraint: fun(x) >= 0
                def fun(x, i=i, j=j):
                    xi, yi, ri = x[2*i], x[2*i+1], x[2*i+2]
                    xj, yj, rj = x[2*j], x[2*j+1], x[2*j+2]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                cons.append({'type': 'ineq', 'fun': fun})
        return cons

    all_constraints = []
    for i in range(N):
        all_constraints.extend(boundary_constraints(i))
    all_constraints.extend(pairwise_constraints())
    
    # Objective function: Maximize sum of radii => Minimize -sum of radii
    def objective(x):
        radii = x[2::3] # Indices 2, 5, 8...
        return -np.sum(radii)

    # Bounds for variables: x_i, y_i in [0,1], r_i in [0, 0.5]
    # Note: Max possible radius is 0.5 (if it fills half the square, but others exist)
    # Let's set upper bound for r_i to 0.5 to be safe.
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((0.0, 0.5)) # r_i

    for restart in range(num_restarts):
        np.random.seed(restart * 98765)
        
        # Initial guess: Random positions, small radii
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[2*i] = np.random.uniform(0.1, 0.9)
            x0[2*i+1] = np.random.uniform(0.1, 0.9)
            x0[2*i+2] = 0.01 # Small initial radius to ensure feasibility
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=all_constraints, options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                # res.fun is -sum(radii), so we want lower res.fun (more negative)
                # Wait, best_score is sum(radii). So we want higher sum(radii).
                # current_score = -res.fun
                current_score = -res.fun
                
                # Verify feasibility just in case SLSQP is loose
                # Check pairwise distances
                centers = res.x[0:2*N].reshape(N, 2)
                radii = res.x[2::3]
                
                valid = True
                for i in range(N):
                    # Boundary check
                    if radii[i] > centers[i][0] or radii[i] > 1.0 - centers[i][0] or \
                       radii[i] > centers[i][1] or radii[i] > 1.0 - centers[i][1]:
                        # Slight tolerance
                        if radii[i] > centers[i][0] + 1e-6 or radii[i] > 1.0 - centers[i][0] + 1e-6 or \
                           radii[i] > centers[i][1] + 1e-6 or radii[i] > 1.0 - centers[i][1] + 1e-6:
                            valid = False
                            break
                            
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.linalg.norm(centers[i] - centers[j])
                            if dist < radii[i] + radii[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and current_score > best_score:
                    best_score = current_score
                    # Format result: (x, y, r)
                    result = np.column_stack((centers, radii))
                    best_result = result.copy()
                    
        except Exception:
            continue
            
    if best_result is None:
        # Fallback: Simple grid or random
        # This shouldn't happen, but good for safety
        centers = np.random.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.01)
        best_result = np.column_stack((centers, radii))
        
    return best_result
# EVOLVE-BLOCK-END
