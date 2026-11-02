"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
TIME_LIMIT = 110.0  # Safety margin before 120s kill (evaluator kills at 150s, but let's be safe)

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(result, W, H, tol=1e-7):
        """Check if circles are disjoint and within bounds."""
        if len(result) != N:
            return False
        if W <= 0 or H <= 0:
            return False
        
        xs = result[:, 0]
        ys = result[:, 1]
        rs = result[:, 2]
        
        # Check radii
        if np.any(rs < 1e-6):
            return False
            
        # Check boundaries: circles must be inside [0, W] x [0, H]
        if np.any(xs - rs < -tol) or np.any(xs + rs > W + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > H + tol):
            return False
        
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def get_value(result):
        return np.sum(result[:, 2])

    def optimize_packing_coordinate_descent(circles, W, H, max_iters=50):
        """
        Refines the packing by iteratively maximizing each circle's radius.
        This is a greedy local search that often improves sum of radii significantly.
        """
        circles = circles.copy().astype(float)
        n = len(circles)
        
        for _ in range(max_iters):
            improved = False
            for i in range(n):
                x, y, r = circles[i]
                
                # Max radius limited by walls
                max_r = min(x, W - x, y, H - y)
                
                # Max radius limited by other circles
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj, rj = circles[j]
                    dx = x - xj
                    dy = y - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        # The new radius r_i can be at most dist - r_j
                        # We want to maximize r_i, so r_i <= dist - r_j
                        max_r = min(max_r, dist - rj)
                
                # Add a small safety margin to avoid numerical issues
                new_r = max(0.0, max_r) * 0.9999
                if new_r > r + 1e-9:
                    circles[i, 2] = new_r
                    improved = True
            if not improved:
                break
        return circles

    def run_optimization_cycle(params0, W_init, H_init, max_iter_main=200):
        """
        Runs the main optimization loop:
        1. Gradient descent to fix overlaps and improve layout.
        2. Coordinate descent to maximize radii.
        3. Final gradient descent polish.
        """
        # 1. Initial Gradient Descent
        # We optimize x, y, r, W. H is derived.
        # Objective: -sum(r) + penalties
        
        def unpack(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            W = params[3*N]
            H = 2.0 - W
            return x, y, r, W, H

        def objective(params):
            x, y, r, W, H = unpack(params)
            if W <= 0 or H <= 0:
                return 1e10
            penalty = 0.0
            # Pairwise overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            # Boundary penalty
            for i in range(N):
                if x[i] - r[i] < 0: penalty += (r[i] - x[i])**2
                if x[i] + r[i] > W: penalty += (x[i] + r[i] - W)**2
                if y[i] - r[i] < 0: penalty += (r[i] - y[i])**2
                if y[i] + r[i] > H: penalty += (y[i] + r[i] - H)**2
            
            return -np.sum(r) + 10.0 * penalty

        # Use L-BFGS-B
        res = minimize(objective, params0, method='L-BFGS-B',
                       options={'maxiter': max_iter_main, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun > 1e6: # Failed
            return None, -np.inf

        x, y, r, W, H = unpack(res.x)
        current_result = np.column_stack([x, y, r])
        
        # 2. Coordinate Descent Refinement
        # This step expands circles to their maximum possible size given current positions.
        # It might push circles out of bounds or cause overlaps if not careful, 
        # but the logic `max_r = min(...)` ensures it stays within bounds and non-overlapping 
        # relative to the *fixed* positions of other circles. 
        # However, since we only change radii, positions remain valid.
        # Wait, if I increase r_i, it might overlap with j?
        # No, `max_r` is calculated as `dist - r_j`. So `new_r <= dist - r_j` => `new_r + r_j <= dist`.
        # So non-overlap is preserved.
        # Boundary is preserved by `min(x, W-x, ...)`.
        
        current_result = optimize_packing_coordinate_descent(current_result, W, H, max_iters=20)
        
        # 3. Final Polish with Gradient Descent
        # Re-pack into params for minimize
        params_polish = np.concatenate([current_result[:,0], current_result[:,1], current_result[:,2], [W]])
        
        # Allow slight adjustment of W and positions to potentially squeeze more?
        # The coordinate descent maximized radii for fixed W.
        # Maybe a different W allows larger radii?
        # Let's run one more minimize pass to adjust W and positions slightly.
        
        res2 = minimize(objective, params_polish, method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res2.fun < res.fun: # Improved
            x2, y2, r2, W2, H2 = unpack(res2.x)
            current_result = np.column_stack([x2, y2, r2])
            W, H = W2, H2
            # One more CD pass?
            current_result = optimize_packing_coordinate_descent(current_result, W, H, max_iters=10)

        val = get_value(current_result)
        return current_result, val

    # Initial Guess Generator
    def initial_guess(W_init):
        H_init = 2.0 - W_init
        # Use a grid layout
        # Try to find a good grid aspect ratio
        # We want to place N circles.
        # Let's try a few grid configurations and pick the one with largest min radius
        best_params = None
        best_r = 0
        
        # Possible grid dimensions
        for n_cols in range(1, 10):
            n_rows = int(np.ceil(N / n_cols))
            if n_cols * n_rows < N:
                continue
            # Calculate cell size
            cell_w = W_init / n_cols
            cell_h = H_init / n_rows
            r_init = 0.5 * min(cell_w, cell_h) - 0.01
            if r_init > best_r:
                best_r = r_init
                x = np.zeros(N)
                y = np.zeros(N)
                r = np.zeros(N)
                for idx in range(N):
                    col = idx % n_cols
                    row = idx // n_cols
                    x[idx] = (col + 0.5) * cell_w
                    y[idx] = (row + 0.5) * cell_h
                    r[idx] = r_init
                best_params = np.concatenate([x, y, r, [W_init]])
        
        if best_params is None:
             # Fallback: square grid
             side = int(np.ceil(np.sqrt(N)))
             cell = min(W_init, H_init) / side
             x = np.zeros(N)
             y = np.zeros(N)
             r = np.zeros(N)
             for idx in range(N):
                 col = idx % side
                 row = idx // side
                 x[idx] = (col + 0.5) * cell
                 y[idx] = (row + 0.5) * cell
                 r[idx] = 0.5 * cell - 0.01
             best_params = np.concatenate([x, y, r, [W_init]])
             
        return best_params

    best_result = None
    best_val = -np.inf

    # Try multiple aspect ratios
    # Perimeter 4 => W + H = 2
    # Square is W=1, H=1
    # Rectangles: 0.8, 0.9, 1.0, 1.1, 1.2, 1.4
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
    
    for W_init in aspect_ratios:
        if elapsed() > TIME_LIMIT - 10:
            break
        
        params0 = initial_guess(W_init)
        res, val = run_optimization_cycle(params0, W_init, 2.0 - W_init, max_iter_main=300)
        
        if res is not None:
            W, H = 2.0 - (2.0 - W_init), 2.0 - W_init # Just for validity check, W is in res? No, W is optimized.
            # We need to extract W from the result if we want to check validity against the optimized W.
            # But `run_optimization_cycle` returns result and val. 
            # To check validity, we need W and H.
            # Let's re-extract W from the optimization. 
            # Actually, `run_optimization_cycle` doesn't return W. 
            # Let's modify it to return W, H as well or just assume the validity check inside is sufficient?
            # Better: `run_optimization_cycle` should return (result, val, W, H).
            
            # Let's refactor slightly to return W, H
            # For now, let's assume the result is valid if val is reasonable.
            # But we need to ensure perimeter <= 4.
            # The optimizer minimizes -sum(r) + penalty.
            # If we just take the result, we should verify it.
            
            # Let's do a quick validity check here using the W from the last iteration?
            # It's safer to re-run the unpacking logic or have the function return W, H.
            
            # Let's stick to the current structure but be careful.
            # The `run_optimization_cycle` uses `W` internally.
            # I will modify `run_optimization_cycle` to return W and H.
            
            pass # Placeholder, logic below handles it

    # Refactored run_optimization_cycle to return W, H
    def run_optimization_cycle_v2(params0, W_init, H_init, max_iter_main=300):
        def unpack(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            W = params[3*N]
            H = 2.0 - W
            return x, y, r, W, H

        def objective(params):
            x, y, r, W, H = unpack(params)
            if W <= 0 or H <= 0:
                return 1e10
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            for i in range(N):
                if x[i] - r[i] < 0: penalty += (r[i] - x[i])**2
                if x[i] + r[i] > W: penalty += (x[i] + r[i] - W)**2
                if y[i] - r[i] < 0: penalty += (r[i] - y[i])**2
                if y[i] + r[i] > H: penalty += (y[i] + r[i] - H)**2
            
            return -np.sum(r) + 10.0 * penalty

        res = minimize(objective, params0, method='L-BFGS-B',
                       options={'maxiter': max_iter_main, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun > 1e6:
            return None, -np.inf, 0, 0

        x, y, r, W, H = unpack(res.x)
        current_result = np.column_stack([x, y, r])
        
        # Coordinate Descent
        current_result = optimize_packing_coordinate_descent(current_result, W, H, max_iters=20)
        
        # Final Polish
        params_polish = np.concatenate([current_result[:,0], current_result[:,1], current_result[:,2], [W]])
        res2 = minimize(objective, params_polish, method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res2.fun < res.fun:
            x2, y2, r2, W2, H2 = unpack(res2.x)
            current_result = np.column_stack([x2, y2, r2])
            W, H = W2, H2
            current_result = optimize_packing_coordinate_descent(current_result, W, H, max_iters=10)
            val = get_value(current_result)
        else:
            val = get_value(current_result)
            
        return current_result, val, W, H

    # Main Loop
    for W_init in aspect_ratios:
        if elapsed() > TIME_LIMIT - 10:
            break
        
        params0 = initial_guess(W_init)
        res, val, W, H = run_optimization_cycle_v2(params0, W_init, 2.0 - W_init, max_iter_main=300)
        
        if res is not None:
            # Strict Validity Check
            if is_valid(res, W, H):
                if val > best_val:
                    best_val = val
                    best_result = res.copy()
            else:
                # If invalid, try to shrink radii slightly to make it valid?
                # Or just skip.
                pass

    # Fallback
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        # Use square 2x2=4, 3x3=9, 4x4=16, 5x5=25. 5x5 is enough for 21.
        # Let's use a 5x5 grid in a square of side 1 (Perimeter 4).
        W = 1.0
        H = 1.0
        cell = W / 5
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (j + 0.5) * cell
                y = (i + 0.5) * cell
                r = 0.5 * cell - 1e-4
                cells.append([x, y, r])
        best_result = np.array(cells[:N])

    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Result shape: {result.shape}")
    print(f"Sum of radii: {np.sum(result[:, 2])}")
