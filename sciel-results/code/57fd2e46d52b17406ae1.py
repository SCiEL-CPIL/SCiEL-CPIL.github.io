"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 118.0  # Safety margin before 120s limit
    best_result = None
    best_score = 0.0

    # We will search over aspect ratios and use multiple random restarts for each
    # w + h = 2
    # ar = h/w -> h = ar*w -> w(1+ar)=2 -> w = 2/(1+ar)
    
    # A set of aspect ratios to try. 1.0 is square.
    # Dense packing in a square is efficient, but rectangles can sometimes fit better.
    aspect_ratios = np.linspace(0.7, 1.4, 10)
    
    # Number of restarts per aspect ratio (determined by time budget)
    max_restarts_per_ar = 5

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h, tol=1e-6):
        # Check if circles are inside rectangle and non-overlapping
        if np.any(r < 0):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        
        # Check overlaps
        # Vectorized distance calculation
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def make_objective(w, h, penalty_weight):
        """
        Creates a soft-body penalty objective function.
        Objective = -sum(r) + penalty * (overlaps + wall_violations)
        """
        def objective(vars_):
            x, y, r = unpack(vars_)
            
            # Sum of radii to maximize (minimize negative)
            sum_r = np.sum(r)
            
            # Overlap Penalty
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        overlap_pen += overlap * overlap
            
            # Wall Penalty
            wall_pen = 0.0
            for i in range(N):
                # Left wall: x - r >= 0 -> viol = r - x
                v1 = r[i] - x[i]
                if v1 > 0: wall_pen += v1 * v1
                
                # Right wall: w - x - r >= 0 -> viol = r + x - w
                v2 = r[i] + x[i] - w
                if v2 > 0: wall_pen += v2 * v2
                
                # Bottom wall: y - r >= 0 -> viol = r - y
                v3 = r[i] - y[i]
                if v3 > 0: wall_pen += v3 * v3
                
                # Top wall: h - y - r >= 0 -> viol = r + y - h
                v4 = r[i] + y[i] - h
                if v4 > 0: wall_pen += v4 * v4

            return -sum_r + penalty_weight * (overlap_pen + wall_pen)
        
        return objective

    def hex_init(w, h, seed):
        rng = np.random.default_rng(seed)
        # Estimate grid dimensions
        # Aim for roughly square cells
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            
            if row % 2 == 1:
                x += 0.5 * w / n_cols
                if x > w:
                    x = w - (x - w) # Reflect
                elif x < 0:
                    x = -x

            xs[i] = x
            ys[i] = y

        # Jitter
        jitter_scale = 0.1 * min(w/n_cols, h/n_rows)
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        # Rough clip
        xs = np.clip(xs, 0.01, w - 0.01)
        ys = np.clip(ys, 0.01, h - 0.01)
        
        # Initial radii: small
        est_r = min(w/n_cols, h/n_rows) * 0.4
        rs = np.full(N, est_r)
        
        return xs, ys, rs

    # Search parameters
    # We will try a few aspect ratios
    # Penalty weights to try (increasing to enforce constraints)
    # We start with a moderate penalty to allow movement, then increase it
    penalty_weights = [10, 50, 100, 500]
    
    # Time budget allocation
    # Total time ~ 118s
    # 10 aspect ratios * 5 restarts * 4 penalty weights = 200 optimizations
    # ~0.59s per optimization. This is tight for L-BFGS-B but feasible with low maxiter.
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break
            
            seed = restart * 100 + int(ar * 100)
            xs, ys, rs = hex_init(w, h, seed)
            
            # Variable vector: [x0, y0, r0, x1, y1, r1, ...]
            x0 = np.zeros(3*N)
            x0[0::3] = xs
            x0[1::3] = ys
            x0[2::3] = rs
            
            # Bounds
            lb = np.zeros(3*N)
            ub = np.zeros(3*N)
            for i in range(N):
                lb[3*i] = 0.0 # x
                ub[3*i] = w
                lb[3*i+1] = 0.0 # y
                ub[3*i+1] = h
                lb[3*i+2] = 0.0 # r
                ub[3*i+2] = min(w, h) / 2.0

            # Annealing penalty weights to guide the optimizer
            # Start with lower penalty to escape local minima, increase to enforce constraints
            current_penalty = 1.0
            current_x = x0.copy()
            
            for pw in penalty_weights:
                if time.time() > deadline - 2:
                    break
                    
                obj_func = make_objective(w, h, pw)
                
                try:
                    result = minimize(
                        obj_func, current_x, method='L-BFGS-B',
                        bounds=list(zip(lb, ub)),
                        options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    
                    current_x = result.x
                    x, y, r = unpack(current_x)
                    
                    # Check validity
                    if validate(x, y, r, w, h):
                        score = np.sum(r)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([x, y, r])
                            
                except Exception:
                    continue

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
