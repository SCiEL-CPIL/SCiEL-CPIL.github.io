"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 with disjoint interiors,
    maximizing the sum of their radii.
    
    Uses a force-directed / gradient-based optimization approach:
    - Start with a reasonable initial configuration (grid + perturbation)
    - Use scipy.optimize.minimize with L-BFGS-B to maximize sum of radii
      subject to non-overlap and boundary constraints
    - Multiple random restarts to avoid local minima
    """
    
    best_solution = None
    best_sum = -1.0
    
    # Number of restarts
    n_restarts = 8
    
    for restart in range(n_restarts):
        # Initialize positions and radii
        if restart == 0:
            # Start with a grid-like arrangement
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        positions.append([x, y])
        else:
            # Random initialization with some structure
            # Use a jittered grid to give a good starting point
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side + random.uniform(-0.1, 0.1) / side
                        y = (j + 0.5) / side + random.uniform(-0.1, 0.1) / side
                        # Clamp to [0,1]
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))
                        positions.append([x, y])
        
        positions = np.array(positions)
        
        # Initial radii: start small
        r0 = 0.02
        radii = np.full(N, r0)
        
        # Variables: [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = positions[i, 0]
            x0[3*i + 1] = positions[i, 1]
            x0[3*i + 2] = radii[i]
        
        # Objective: minimize negative sum of radii
        def objective(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            # Check validity
            for i in range(N):
                if rs[i] < 0:
                    return 1e10
                if xs[i] - rs[i] < -1e-9 or xs[i] + rs[i] > 1 + 1e-9:
                    return 1e10
                if ys[i] - rs[i] < -1e-9 or ys[i] + rs[i] > 1 + 1e-9:
                    return 1e10
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]
                    if dist_sq < min_dist * min_dist - 1e-9:
                        # Penalty for overlap
                        overlap = min_dist - np.sqrt(dist_sq)
                        return 1e10 + overlap * 100
            return -np.sum(rs)
        
        # Gradient for L-BFGS-B
        def gradient(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            
            grad = np.zeros(3 * N)
            
            # Gradient of -sum(r) w.r.t. r_i is -1
            for i in range(N):
                grad[3*i + 2] = -1.0
            
            # Add gradient from overlap penalties
            # For each pair, if overlapping, push apart
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]
                    if dist_sq > 1e-12 and dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        # Force proportional to overlap
                        overlap = min_dist - dist
                        # Unit vector from j to i
                        ux = dx / dist
                        uy = dy / dist
                        # Gradient contribution: pushing i away from j and j away from i
                        # The penalty is (min_dist - dist)^2 * lambda, but we use linear
                        # d/dx_i of (min_dist - dist) = -d(dist)/dx_i = -(dx/dist)
                        # So to minimize penalty, we increase dist
                        # Gradient of penalty w.r.t. x_i: -ux * lambda
                        # We want to move to reduce overlap, so add force in direction of separation
                        lambda_force = 100.0 * overlap
                        grad[3*i] += lambda_force * ux
                        grad[3*i + 1] += lambda_force * uy
                        grad[3*j] -= lambda_force * ux
                        grad[3*j + 1] -= lambda_force * uy
                        
                        # Also gradient w.r.t. radii: increasing radius increases overlap
                        grad[3*i + 2] += lambda_force
                        grad[3*j + 2] += lambda_force
            
            # Boundary penalties
            for i in range(N):
                # x boundaries
                if xs[i] - rs[i] < 0:
                    grad[3*i] += 50.0  # push right
                    grad[3*i + 2] -= 50.0  # shrinking helps
                if xs[i] + rs[i] > 1:
                    grad[3*i] -= 50.0  # push left
                    grad[3*i + 2] -= 50.0
                # y boundaries
                if ys[i] - rs[i] < 0:
                    grad[3*i + 1] += 50.0
                    grad[3*i + 2] -= 50.0
                if ys[i] + rs[i] > 1:
                    grad[3*i + 1] -= 50.0
                    grad[3*i + 2] -= 50.0
            
            return grad
        
        # Bounds: x, y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.001, 0.5))  # r
        
        # Run optimization
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            v_opt = result.x
            xs = v_opt[0::3]
            ys = v_opt[1::3]
            rs = v_opt[2::3]
            
            # Validate the solution
            valid = True
            for i in range(N):
                if rs[i] < 0:
                    valid = False
                    break
                if xs[i] - rs[i] < -1e-6 or xs[i] + rs[i] > 1 + 1e-6:
                    valid = False
                    break
                if ys[i] - rs[i] < -1e-6 or ys[i] + rs[i] > 1 + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx * dx + dy * dy
                        min_dist = rs[i] + rs[j]
                        if dist_sq < min_dist * min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Clamp to be safely inside
                for i in range(N):
                    rs[i] = min(rs[i], xs[i], 1 - xs[i], ys[i], 1 - ys[i])
                    # Re-validate after clamping
                # Final validation
                valid2 = True
                for i in range(N):
                    if rs[i] < 0:
                        valid2 = False
                        break
                    if xs[i] - rs[i] < -1e-7 or xs[i] + rs[i] > 1 + 1e-7:
                        valid2 = False
                        break
                    if ys[i] - rs[i] < -1e-7 or ys[i] + rs[i] > 1 + 1e-7:
                        valid2 = False
                        break
                if valid2:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist_sq = dx * dx + dy * dy
                            min_dist = rs[i] + rs[j]
                            if dist_sq < min_dist * min_dist - 1e-7:
                                valid2 = False
                                break
                        if not valid2:
                            break
                
                if valid2:
                    total = np.sum(rs)
                    if total > best_sum:
                        best_sum = total
                        best_solution = np.column_stack([xs, ys, rs])
                        
        except Exception as e:
            continue
    
    # If no valid solution found, fall back to the simple grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution

# EVOLVE-BLOCK-END
