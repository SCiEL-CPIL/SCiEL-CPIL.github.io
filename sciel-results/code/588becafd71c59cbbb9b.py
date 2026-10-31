"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10s safety margin
    
    # Initial guess: grid arrangement
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of positions
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append([(i + 0.5) / side, (j + 0.5) / side])
    
    # Ensure we have exactly N positions
    while len(positions) < N:
        positions.append([0.5, 0.5])
    positions = positions[:N]
    
    # Initial radii: small
    r0 = 0.02 * np.ones(N)
    pos0 = np.array(positions)
    
    # Combined state: [x0, y0, x1, y1, ..., xN-1, yN-1, r0, r1, ..., rN-1]
    x0 = np.concatenate([pos0.flatten(), r0])
    
    # Objective: maximize sum of radii => minimize -sum(r)
    def objective(x):
        if time.time() > deadline:
            return 1e10  # Penalize if over deadline
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        return -np.sum(radii)
    
    # Constraints:
    # 1. Circles inside unit square: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # 2. Disjoint interiors: dist(i,j) >= r_i + r_j for all i != j
    #    => dist(i,j) - r_i - r_j >= 0
    
    def constraints(x):
        if time.time() > deadline:
            return [1e10] * (4*N + N*(N-1)//2)
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        cons = []
        
        # Boundary constraints
        for i in range(N):
            xi, yi = pos[i]
            ri = radii[i]
            cons.append(xi - ri)  # >= 0
            cons.append(1.0 - xi - ri)  # >= 0
            cons.append(yi - ri)  # >= 0
            cons.append(1.0 - yi - ri)  # >= 0
        
        # Disjointness constraints
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                cons.append(dist - radii[i] - radii[j])  # >= 0
        
        return cons
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.0, 0.5))  # r
    
    # Use SLSQP with hard constraints
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Multiple restarts with different initial conditions
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb initial conditions
            np.random.seed(restart)
            pos_perturbed = pos0 + 0.1 * np.random.randn(N, 2)
            pos_perturbed = np.clip(pos_perturbed, 0.05, 0.95)
            r_perturbed = r0 * (1.0 + 0.5 * np.random.rand(N))
            r_perturbed = np.clip(r_perturbed, 0.01, 0.4)
            x_init = np.concatenate([pos_perturbed.flatten(), r_perturbed])
        
        try:
            result = minimize(
                objective,
                x_init,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_val:
                # Check feasibility
                pos = result.x[:2*N].reshape(N, 2)
                radii = result.x[2*N:]
                
                # Verify all constraints are satisfied
                feasible = True
                for i in range(N):
                    if pos[i][0] - radii[i] < -1e-6 or 1.0 - pos[i][0] - radii[i] < -1e-6:
                        feasible = False
                        break
                    if pos[i][1] - radii[i] < -1e-6 or 1.0 - pos[i][1] - radii[i] < -1e-6:
                        feasible = False
                        break
                
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                            if dist < radii[i] + radii[j] - 1e-6:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible and result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
        except Exception as e:
            pass
    
    # Extract best solution
    pos = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    
    # Final polish: ensure strict feasibility by slightly shrinking if needed
    # Check and adjust
    for i in range(N):
        # Ensure inside square
        if pos[i][0] - radii[i] < 0:
            radii[i] = max(0, pos[i][0])
        if 1.0 - pos[i][0] - radii[i] < 0:
            radii[i] = min(radii[i], 1.0 - pos[i][0])
        if pos[i][1] - radii[i] < 0:
            radii[i] = max(0, pos[i][1])
        if 1.0 - pos[i][1] - radii[i] < 0:
            radii[i] = min(radii[i], 1.0 - pos[i][1])
    
    # Re-check disjointness and shrink if needed
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
            if dist < radii[i] + radii[j]:
                # Shrink both slightly
                excess = radii[i] + radii[j] - dist
                shrink = excess / 2.0 + 1e-8
                radii[i] = max(0, radii[i] - shrink)
                radii[j] = max(0, radii[j] - shrink)
    
    # Build output
    result_array = np.zeros((N, 3))
    for i in range(N):
        result_array[i, 0] = pos[i][0]
        result_array[i, 1] = pos[i][1]
        result_array[i, 2] = radii[i]
    
    return result_array
# EVOLVE-BLOCK-END
