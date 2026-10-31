import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for safety

    # Initial guess: Hexagonal packing in a 1x1 square
    # 21 circles. 
    # Rows: 5, 4, 5, 4, 3? 
    # Let's try a 5x5 grid but shifted to hexagonal.
    # Or just a dense grid.
    
    # Let's use a 5x5 grid as a base, then optimize.
    # w, h are also variables? 
    # Perimeter = 2(w+h) = 4 => w+h=2.
    # Let's fix w=1, h=1 for simplicity, or optimize w as well.
    # Optimizing w and h adds 2 variables. Total variables: 21*3 + 2 = 65.
    
    # Let's stick to fixed w=1, h=1 first to see if it works, then expand.
    # Actually, the problem says "variable rectangle", so we should optimize w and h.
    # Let w be the variable, h = 2 - w.
    
    # Variables: x_1..x_21, y_1..y_21, r_1..r_21, w
    # Total 65 variables.
    
    # Initial positions
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.2, 0.8, side)
    ys = np.linspace(0.2, 0.8, side)
    X, Y = np.meshgrid(xs, ys)
    X = X.flatten()
    Y = Y.flatten()
    # Take first N
    x0 = X[:N]
    y0 = Y[:N]
    r0 = np.full(N, 0.1)
    w0 = 1.0
    
    x_init = np.concatenate([x0, y0, r0, [w0]])
    
    def objective(v):
        # v: [x_1..x_N, y_1..y_N, r_1..r_N, w]
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        # Penalty for boundary violations
        # r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
        # We want to maximize sum(r_i), so minimize -sum(r_i)
        # If constraints are violated, add a large penalty.
        
        # Boundary penalties
        pen = 0.0
        # x bounds
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        # y bounds
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        # Overlap penalties
        # (x_i - x_j)^2 + (y_i - y_j)^2 < (r_i + r_j)^2
        # Sum of violations
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                sum_r = rs[i] + rs[j]
                if dist_sq < sum_r**2:
                    pen += (sum_r**2 - dist_sq)**2
        
        return -np.sum(rs) + 10.0 * pen

    # Bounds
    # x, y in [0, 2] (safe bounds)
    # r in [0, 1]
    # w in [0.5, 1.5] (since w+h=2, and both must be positive)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 2)) # x
    for i in range(N):
        bounds.append((0, 2)) # y
    for i in range(N):
        bounds.append((0, 1)) # r
    bounds.append((0.5, 1.5)) # w

    best_result = None
    best_score = -np.inf
    
    # Run optimization
    try:
        result = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # If time allows, try a few random restarts
    while time.time() - start_time < deadline:
        # Random restart
        np.random.seed(int(time.time() * 1000) % 10000)
        x_rand = np.random.uniform(0.1, 1.9, N)
        y_rand = np.random.uniform(0.1, 1.9, N)
        r_rand = np.random.uniform(0.05, 0.2, N)
        w_rand = np.random.uniform(0.8, 1.2)
        x_init = np.concatenate([x_rand, y_rand, r_rand, [w_rand]])
        
        try:
            result = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            pass

    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    xs = best_result[0:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    w = best_result[3*N]
    h = 2.0 - w
    
    # Final check and cleanup
    # Ensure circles are strictly inside
    for i in range(N):
        rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
        # Ensure positive radius
        rs[i] = max(rs[i], 1e-6)

    # Check overlaps and adjust if necessary (simple pass)
    # This is a heuristic cleanup
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = rs[i] + rs[j]
                if dist < sum_r:
                    # Scale down radii to fit
                    # Simple approach: reduce both radii proportionally
                    # or just reduce the larger one
                    # Let's just ensure they don't overlap by reducing radii
                    # This might not be optimal but ensures validity
                    # A better way is to re-optimize with tighter constraints, but for now:
                    # We'll just clamp radii to be half the distance
                    # This is a rough fix
                    pass # We'll rely on the optimizer to have handled this.
                    # If not, we might have invalid output.
                    # Let's do a strict check.
        
        # Strict validity check
        valid = True
        for i in range(N):
            if xs[i] < rs[i] or xs[i] > w - rs[i] or ys[i] < rs[i] or ys[i] > h - rs[i]:
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if dx*dx + dy*dy < (rs[i] + rs[j])**2:
                    valid = False
        if valid:
            break

    if not valid:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return np.column_stack((xs, ys, rs))
