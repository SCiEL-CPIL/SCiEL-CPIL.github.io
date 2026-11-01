"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Rectangle dimensions: W + H = 2. Let's start with square W=1, H=1.
    # We can also optimize W and H, but for N=21, square is often optimal or near-optimal.
    # Let's fix W=1, H=1 for simplicity and focus on circle optimization.
    W = 1.0
    H = 1.0
    
    best_val = 0.0
    best_x = None
    
    # Initial guess: Hexagonal packing in a 1x1 square
    # Approximate radius for 21 circles in 1x1: Area ~ 21*pi*r^2 <= 1 => r <= sqrt(1/(21*pi)) ~ 0.123
    # But packing density is lower, so r will be smaller.
    
    def create_initial_guess(seed):
        np.random.seed(seed)
        # Grid-like start with jitter
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                r = 0.1 # Initial small radius
                cells.append([x, y, r])
        # Shuffle to break symmetry
        np.random.shuffle(cells)
        return np.array(cells)

    def objective_and_penalty(params):
        # params: shape (N, 3) -> x, y, r
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalties
        pen = 0.0
        
        # Boundary constraints
        # x >= r, x <= W - r, y >= r, y <= H - r
        pen += np.sum(np.maximum(0, r - x)**2)
        pen += np.sum(np.maximum(0, r - (W - x))**2)
        pen += np.sum(np.maximum(0, r - y)**2)
        pen += np.sum(np.maximum(0, r - (H - y))**2)
        
        # Overlap constraints
        # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
        for i in range(N):
            for j in range(i + 1, N):
                dist2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist2 < min_dist**2:
                    diff = min_dist - np.sqrt(dist2)
                    pen += diff**2
        
        # We want to maximize sum_r - penalty
        # So minimize - (sum_r - penalty) = -sum_r + penalty
        return -sum_r + pen

    # Run multiple local optimizations
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        x0 = create_initial_guess(restart)
        
        # Use Nelder-Mead
        try:
            res = minimize(objective_and_penalty, x0.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                x_opt = res.x.reshape(-1, 3)
                # Check validity
                val = -res.fun
                # If penalty is non-zero, the solution is invalid or not fully converged
                # We need to ensure constraints are satisfied
                x = x_opt[:, 0]
                y = x_opt[:, 1]
                r = x_opt[:, 2]
                
                # Verify constraints strictly
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-9 or x[i] > W - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > H - r[i] + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                            min_dist = r[i] + r[j]
                            if dist2 < min_dist**2 - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    if val > best_val:
                        best_val = val
                        best_x = x_opt
        except Exception:
            continue

    if best_x is None:
        # Fallback to grid if optimization failed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_x

# EVOLVE-BLOCK-END
