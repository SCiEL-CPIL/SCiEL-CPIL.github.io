"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Helper to solve for optimal radii given centers
    def solve_radii(centers):
        # Objective: maximize sum(r) -> minimize -sum(r)
        c = -np.ones(N)
        
        # Constraints:
        # 1. r_i + r_j <= d_ij
        # 2. r_i <= bound_i
        # 3. r_i >= 0
        
        A_ub = []
        b_ub = []
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
        
        # Boundary constraints
        bounds = []
        for i in range(N):
            b_i = min(centers[i][0], 1.0 - centers[i][0], centers[i][1], 1.0 - centers[i][1])
            row = np.zeros(N)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(b_i)
            bounds.append((0, None))
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback if LP fails (shouldn't happen for feasible geometry)
            return -1e6, np.zeros(N)

    # Parameters for Local Search
    num_restarts = 10
    steps_per_restart = 2000
    base_step_size = 0.05
    
    for restart in range(num_restarts):
        np.random.seed(restart * 1000 + 42)
        
        # Initialize with random positions
        centers = np.random.uniform(0.1, 0.9, size=(N, 2))
        
        # Get initial score
        current_score, current_radii = solve_radii(centers)
        
        if current_score > best_score:
            best_score = current_score
            best_result = np.column_stack((centers, current_radii)).copy()
            
        # Local Search
        for step in range(steps_per_restart):
            # Adaptive step size
            step_size = base_step_size * (1.0 - step / steps_per_restart) + 0.01
            
            # Pick a random circle to move
            i = np.random.randint(N)
            
            # Store original position
            old_pos = centers[i].copy()
            
            # Perturb position
            new_pos = old_pos + np.random.normal(0, step_size, size=2)
            new_pos = np.clip(new_pos, 0.0, 1.0)
            
            centers[i] = new_pos
            
            # Solve for new radii
            new_score, new_radii = solve_radii(centers)
            
            # If improved, keep; else revert
            if new_score > current_score:
                current_score = new_score
                current_radii = new_radii
                
                if current_score > best_score:
                    best_score = current_score
                    best_result = np.column_stack((centers, current_radii)).copy()
            else:
                centers[i] = old_pos
                
    return best_result
# EVOLVE-BLOCK-END
