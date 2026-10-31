"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initial grid setup
    side = int(np.ceil(np.sqrt(N)))
    # 5x5 grid for 26 circles
    xs = []
    ys = []
    rs = []
    for i in range(side):
        for j in range(side):
            if len(xs) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            xs.append(x)
            ys.append(y)
            rs.append(r)
    
    x0 = np.array(xs + ys + rs)
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    # We will handle x>=r etc via constraints
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    
    # Constraints for SLSQP
    # 1. Boundary constraints
    # 2. Non-overlap constraints
    
    def boundary_constraints(x):
        cons = []
        for i in range(N):
            xi = x[2*i]
            yi = x[2*i+1]
            ri = x[2*N + i]
            # xi >= ri
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N + i]})
            # xi <= 1 - ri
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*N + i]})
            # yi >= ri
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N + i]})
            # yi <= 1 - ri
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*N + i]})
        return cons

    def non_overlap_constraints(x):
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                xi = x[2*i]
                yi = x[2*i+1]
                ri = x[2*N + i]
                xj = x[2*j]
                yj = x[2*j+1]
                rj = x[2*N + j]
                # dist >= ri + rj
                # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
                # This is non-smooth at dist=0, but unlikely to be an issue if init is good.
                # To be safe, we can use squared distance? No, constraint is on distance.
                # SLSQP handles this okay usually.
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - x[2*N+i] - x[2*N+j]
                })
        return cons

    constraints = boundary_constraints(x0) + non_overlap_constraints(x0)
    
    def objective(x):
        # Maximize sum of radii -> Minimize negative sum
        radii = x[2*N:]
        return -np.sum(radii)

    best_result = None
    best_score = -np.inf
    
    # Try SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
        if res.success:
            # Check validity
            x_res = res.x
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_res[2*i]-x_res[2*j])**2 + (x_res[2*i+1]-x_res[2*j+1])**2)
                    if dist < x_res[2*N+i] + x_res[2*N+j] - 1e-9:
                        valid = False
                        break
                if not valid: break
            for i in range(N):
                if x_res[2*i] < x_res[2*N+i] - 1e-9 or x_res[2*i] > 1 - x_res[2*N+i] + 1e-9:
                    valid = False
                    break
                if x_res[2*i+1] < x_res[2*N+i] - 1e-9 or x_res[2*i+1] > 1 - x_res[2*N+i] + 1e-9:
                    valid = False
                    break
            
            if valid:
                score = np.sum(x_res[2*N:])
                if score > best_score:
                    best_score = score
                    best_result = x_res
    except Exception as e:
        pass

    if best_result is not None:
        # Format output
        out = np.zeros((N, 3))
        for i in range(N):
            out[i, 0] = best_result[2*i]
            out[i, 1] = best_result[2*i+1]
            out[i, 2] = best_result[2*N+i]
        return out
    
    # Fallback: Return initial grid if optimization failed or invalid
    # Ensure initial grid is valid
    out = np.zeros((N, 3))
    for i in range(N):
        out[i, 0] = xs[i]
        out[i, 1] = ys[i]
        out[i, 2] = rs[i]
    return out

# EVOLVE-BLOCK-END
