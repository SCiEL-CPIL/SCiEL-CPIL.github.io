# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10():
    n = 10
    tol = 1e-7
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin

    def compute_score(centers, sides):
        """Returns (total_side_length, penalty). Lower penalty is better.
        total_side_length is the objective we want to maximize.
        penalty is a smooth measure of constraint violation."""
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Boundary violations
        pen = 0.0
        for i in range(n):
            # left/right
            if cx[i] - s[i]/2 < -tol:
                pen += (cx[i] - s[i]/2 + tol) ** 2
            if cx[i] + s[i]/2 > 1 + tol:
                pen += (cx[i] + s[i]/2 - 1 - tol) ** 2
            # bottom/top
            if cy[i] - s[i]/2 < -tol:
                pen += (cy[i] - s[i]/2 + tol) ** 2
            if cy[i] + s[i]/2 > 1 + tol:
                pen += (cy[i] + s[i]/2 - 1 - tol) ** 2
        
        # Overlap violations (smooth)
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # If both dx < t and dy < t, there's overlap
                # Use smooth max: violation_x = max(0, t - dx), violation_y = max(0, t - dy)
                # Overlap exists if both are positive
                vx = max(0.0, t - dx)
                vy = max(0.0, t - dy)
                if vx > 0 and vy > 0:
                    pen += (vx * vy) ** 2  # product squared for smoothness
        
        total_side = np.sum(s)
        return total_side, pen

    def is_strictly_valid(centers, sides):
        """Check with the exact evaluator rules."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def anneal(x0, t_max, t_min, n_steps, rng):
        """Run simulated annealing from initial state x0.
        x0 is a flat array of 30 values: s[0..9], cx[0..9], cy[0..9]"""
        x = x0.copy()
        T = t_max
        cooling = (t_min / t_max) ** (1.0 / n_steps)
        
        best_x = x.copy()
        best_score = compute_score(x[:n].reshape(-1,1).squeeze(), x[n:2*n])  # placeholder
        # Actually let's restructure: x = [s(10), cx(10), cy(10)]
        best_obj = -np.inf
        
        def unpack(x):
            s = x[:n]
            cx = x[n:2*n]
            cy = x[2*n:3*n]
            centers = np.column_stack([cx, cy])
            return centers, s
        
        def pack(centers, s):
            return np.concatenate([s, centers[:, 0], centers[:, 1]])
        
        centers, s = unpack(x)
        obj, pen = compute_score(centers, s)
        current_energy = pen - obj  # we minimize this (penalty - objective)
        
        best_x = x.copy()
        best_energy = current_energy
        best_obj_val = obj
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            T *= cooling
            
            # Generate neighbor
            x_new = x.copy()
            # Perturb randomly
            n_perturb = rng.integers(1, 4)
            for _ in range(n_perturb):
                idx = rng.integers(0, 3*n)
                if idx < n:  # side length
                    x_new[idx] += rng.normal(0, 0.02)
                    x_new[idx] = max(0.01, min(1.0, x_new[idx]))
                else:  # center
                    x_new[idx] += rng.normal(0, 0.02)
                    x_new[idx] = max(0.0, min(1.0, x_new[idx]))
            
            c_new, s_new = unpack(x_new)
            obj_new, pen_new = compute_score(c_new, s_new)
            new_energy = pen_new - obj_new
            
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                x = x_new
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_x = x_new.copy()
                    best_obj_val = obj_new
        
        c_best, s_best = unpack(best_x)
        obj_best, pen_best = compute_score(c_best, s_best)
        return obj_best, pen_best, c_best, s_best

    # Multiple restarts
    rng = np.random.default_rng(12345)
    best_overall_obj = -np.inf
    best_overall_centers = None
    best_overall_sides = None
    
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize: random valid-ish starting point
        # Try different initialization strategies
        if r % 3 == 0:
            # Grid-like start
            k = 4
            side = 0.2
            for i in range(n):
                col = i % k
                row = i // k
                cx_i = (col + 0.5) * side + rng.normal(0, 0.01)
                cy_i = (row + 0.5) * side + rng.normal(0, 0.01)
                s_i = side * (1 + rng.normal(0, 0.05))
                # Store
                if r == 0:
                    pass  # handled below
            # Build x0
            s0 = np.full(n, 0.2)
            cx0 = np.array([(i % 4 + 0.5) * 0.2 for i in range(n)]) + rng.normal(0, 0.01, n)
            cy0 = np.array([(i // 4 + 0.5) * 0.2 for i in range(n)]) + rng.normal(0, 0.01, n)
        elif r % 3 == 1:
            # Random small squares
            s0 = rng.uniform(0.05, 0.15, n)
            cx0 = s0/2 + rng.uniform(0, 1-s0)  # ensure in bounds
            cy0 = s0/2 + rng.uniform(0, 1-s0)
        else:
            # 3x3 + 1 style
            s0 = np.full(n, 0.3)
            positions = []
            for i in range(9):
                col = i % 3
                row = i // 3
                positions.append(((col + 0.5) / 3, (row + 0.5) / 3))
            positions.append((0.5, 0.5))
            cx0 = np.array([p[0] for p in positions]) + rng.normal(0, 0.01, n)
            cy0 = np.array([p[1] for p in positions]) + rng.normal(0, 0.01, n)
        
        x0 = np.concatenate([s0, cx0, cy0])
        
        # Annealing parameters
        t_max = 1.0
        t_min = 1e-6
        n_steps = 2000
        
        obj, pen, centers, sides = anneal(x0, t_max, t_min, n_steps, rng)
        
        if obj > best_overall_obj:
            best_overall_obj = obj
            best_overall_centers = centers
            best_overall_sides = sides
    
    # Final validation and cleanup
    if best_overall_centers is not None and is_strictly_valid(best_overall_centers, best_overall_sides):
        return best_overall_centers.astype(np.float64), best_overall_sides.astype(np.float64)
    
    # Fallback: simple 4x4 grid with 10 squares
    k = 4
    side = 1.0 / k
    cx = np.array([(i % k + 0.5) * side for i in range(n)])
    cy = np.array([(i // k + 0.5) * side for i in range(n)])
    centers = np.column_stack([cx, cy]).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
