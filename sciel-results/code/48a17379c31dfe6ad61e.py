"""Improved circle packing in perimeter-4 rectangle using SLSQP with multi-start and polishing."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Objective: maximize sum of radii
    # Constraints: circles disjoint, inside rectangle with perimeter <= 4
    # Variables: x_i, y_i, r_i for each circle i
    # Rectangle: width W, height H with 2(W+H) <= 4, so W+H <= 2
    # We can parameterize by W and H, but let's fix aspect ratio or optimize over it
    
    # Strategy: For a given rectangle (W, H) with W+H=2, pack circles
    # Actually, let's optimize W and H as well, or fix them and scale
    
    # Better approach: Use a fixed rectangle and optimize circle positions/radii
    # The rectangle has perimeter 4, so if width=W, height=H, then 2(W+H)=4 => W+H=2
    # Let's try different aspect ratios
    
    best_value = 0.0
    best_solution = None
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if remaining() < 5.0:
            break
        
        # W + H = 2, W/H = ar => W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Initial guess: grid placement
        side = int(np.ceil(np.sqrt(N)))
        cells_x = []
        cells_y = []
        cells_r = []
        
        # Place circles in a grid
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = 0.5 * min(W, H) / side * 0.9
            cells_x.append(x)
            cells_y.append(y)
            cells_r.append(r)
        
        x0 = np.array(cells_x + cells_y + cells_r)
        
        # Constraints:
        # 1. Circles inside rectangle: r_i <= x_i, r_i <= W - x_i, r_i <= y_i, r_i <= H - y_i
        # 2. Circles disjoint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        def objective(x):
            # Negative sum of radii (we minimize negative)
            radii = x[2*N:]
            return -np.sum(radii)
        
        def constraints(x):
            cons = []
            xs = x[:N]
            ys = x[N:2*N]
            rs = x[2*N:]
            
            # Inside rectangle constraints
            for i in range(N):
                # x_i >= r_i
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
                # W - x_i >= r_i
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - x[i] - x[2*N+i]})
                # y_i >= r_i
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
                # H - y_i >= r_i
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[N+i] - x[2*N+i]})
                # r_i > 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i]})
            
            # Disjointness constraints
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: 
                            (x[i] - x[j])**2 + (x[N+i] - x[N+j])**2 - (x[2*N+i] + x[2*N+j])**2
                    })
            
            return cons
        
        # Use SLSQP
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(1e-6, W-1e-6)] * N + [(1e-6, H-1e-6)] * N + [(1e-6, min(W,H)/2)] * N,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result.success:
            value = -result.fun
            if value > best_value:
                best_value = value
                best_solution = result.x
        
        # Try a few random restarts
        for restart in range(3):
            if remaining() < 3.0:
                break
            
            # Random initial positions
            np.random.seed(42 + restart + int(ar * 100))
            xs_rand = np.random.uniform(0.1*W, 0.9*W, N)
            ys_rand = np.random.uniform(0.1*H, 0.9*H, N)
            rs_rand = np.random.uniform(0.05, 0.15, N)
            
            x0_rand = np.concatenate([xs_rand, ys_rand, rs_rand])
            
            result_rand = minimize(
                objective,
                x0_rand,
                method='SLSQP',
                constraints=constraints,
                bounds=[(1e-6, W-1e-6)] * N + [(1e-6, H-1e-6)] * N + [(1e-6, min(W,H)/2)] * N,
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            
            if result_rand.success:
                value = -result_rand.fun
                if value > best_value:
                    best_value = value
                    best_solution = result_rand.x
    
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = 0.5 * min(W, H) / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells[:N])
    
    # Extract circles
    xs = best_solution[:N]
    ys = best_solution[N:2*N]
    rs = best_solution[2*N:]
    
    circles = np.column_stack([xs, ys, rs])
    
    return circles

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} circles")
    print(f"Sum of radii: {np.sum(result[:, 2])}")
