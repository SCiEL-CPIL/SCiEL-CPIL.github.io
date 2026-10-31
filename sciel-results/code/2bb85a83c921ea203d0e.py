"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import random


def _circle_constraints(x, y, r, others_x, others_y, others_r):
    """
    Constraints for a single circle (x, y, r) given other fixed circles.
    Returns (objective, constraints_dict) where objective is -r (to maximize r).
    Constraints:
      - x - r >= 0
      - y - r >= 0
      - (1 - x) - r >= 0
      - (1 - y) - r >= 0
      - dist((x,y),(xi,yi)) - (r + ri) >= 0 for each other circle
    """
    def neg_r(v):
        return -v[2]
    
    # Inequality constraints: g(v) >= 0
    # We'll build a linear constraint for the box and non-linear for circle packing.
    # Actually, scipy minimize with SLSQP handles non-linear constraints.
    
    def box_constraints(v):
        # v = [x, y, r]
        # x - r >= 0  => v[0] - v[2] >= 0
        # y - r >= 0  => v[1] - v[2] >= 0
        # 1 - x - r >= 0 => 1 - v[0] - v[2] >= 0
        # 1 - y - r >= 0 => 1 - v[1] - v[2] >= 0
        return [
            v[0] - v[2],
            v[1] - v[2],
            1 - v[0] - v[2],
            1 - v[1] - v[2]
        ]
    
    def circle_constraints(v):
        # dist - (r + ri) >= 0
        cons = []
        for i in range(len(others_x)):
            dx = v[0] - others_x[i]
            dy = v[1] - others_y[i]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - (v[2] + others_r[i]))
        return cons
    
    def constraints_func(v):
        return box_constraints(v) + circle_constraints(v)
    
    return neg_r, constraints_func


def construct():
    random.seed(42)
    np.random.seed(42)
    
    # Start with a reasonable initial configuration
    # Use a hexagonal-like packing or just random initialization with local optimization
    
    # Strategy: 
    # 1. Initialize circles in a grid pattern with small radii
    # 2. Iteratively optimize each circle's position and radius given others are fixed
    # 3. Repeat multiple passes
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8  # slightly smaller to allow room
            cells.append([x, y, r])
    
    cells = np.array(cells[:N])
    
    # Local optimization: iteratively optimize each circle
    # For each circle, treat others as fixed and find optimal (x, y, r)
    
    max_iters = 50
    tol = 1e-6
    
    for iteration in range(max_iters):
        total_change = 0.0
        
        for idx in range(N):
            x0, y0, r0 = cells[idx]
            
            # Get other circles
            other_indices = [i for i in range(N) if i != idx]
            others_x = cells[other_indices, 0]
            others_y = cells[other_indices, 1]
            others_r = cells[other_indices, 2]
            
            # Objective: maximize r => minimize -r
            def objective(v):
                return -v[2]
            
            # Constraints: all >= 0
            def constraints_func(v):
                cons = []
                # Box constraints
                cons.append(v[0] - v[2])          # x - r >= 0
                cons.append(v[1] - v[2])          # y - r >= 0
                cons.append(1 - v[0] - v[2])      # 1 - x - r >= 0
                cons.append(1 - v[1] - v[2])      # 1 - y - r >= 0
                # Circle-circle constraints
                for i in range(len(others_x)):
                    dx = v[0] - others_x[i]
                    dy = v[1] - others_y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons.append(dist - (v[2] + others_r[i]))
                return cons
            
            # Initial guess
            x_init = [x0, y0, r0]
            
            # Bounds
            bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)]
            
            # Solve
            try:
                res = minimize(
                    objective,
                    x_init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraints_func}],
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.success and res.x[2] > 0:
                    new_x, new_y, new_r = res.x
                    # Ensure validity
                    if (new_x - new_r >= -1e-9 and new_y - new_r >= -1e-9 and
                        1 - new_x - new_r >= -1e-9 and 1 - new_y - new_r >= -1e-9):
                        # Check circle-circle
                        valid = True
                        for i in range(len(others_x)):
                            dx = new_x - others_x[i]
                            dy = new_y - others_y[i]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < new_r + others_r[i] - 1e-9:
                                valid = False
                                break
                        if valid:
                            cells[idx] = [new_x, new_y, new_r]
                            total_change += abs(new_r - r0) + abs(new_x - x0) + abs(new_y - y0)
            except Exception:
                pass
        
        if total_change < tol:
            break
    
    # Final validation and cleanup
    result = []
    for i in range(N):
        x, y, r = cells[i]
        # Clamp to valid range
        r = max(0.0, min(r, min(x, y, 1-x, 1-y)))
        if r > 0:
            result.append([x, y, r])
    
    # If we lost some circles due to clamping, add tiny ones
    while len(result) < N:
        # Find a spot
        for _ in range(100):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = 0.01
            valid = True
            for (ox, oy, orr) in result:
                if np.sqrt((x-ox)**2 + (y-oy)**2) < r + orr:
                    valid = False
                    break
            if valid and x - r >= 0 and y - r >= 0 and 1 - x - r >= 0 and 1 - y - r >= 0:
                result.append([x, y, r])
                break
        else:
            result.append([0.01, 0.01, 0.001])
    
    return np.array(result[:N])

# EVOLVE-BLOCK-END
