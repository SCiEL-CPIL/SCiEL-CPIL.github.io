import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    def is_valid(centers, sides):
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
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def to_flat(centers, sides):
        flat = np.zeros(3 * n)
        flat[0::3] = centers[:, 0]
        flat[1::3] = centers[:, 1]
        flat[2::3] = sides
        return flat

    def from_flat(flat):
        centers = np.zeros((n, 2))
        centers[:, 0] = flat[0::3]
        centers[:, 1] = flat[1::3]
        sides = flat[2::3]
        return centers, sides

    def softplus(x, beta=1.0):
        # Smooth approximation of max(0, x)
        # log(1 + exp(beta*x)) / beta
        # For large beta, it approaches max(0,x)
        # To avoid overflow, use np.logaddexp(0, beta*x)
        return np.logaddexp(0, beta * x) / beta

    def objective(params, alpha=50.0, lam=100.0):
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # Maximize sum(s) -> Minimize -sum(s)
        obj = -np.sum(s)
        
        # Boundary penalties
        # x in [s/2, 1-s/2]
        pen_x_low = softplus(alpha * (s/2 - cx), alpha)
        pen_x_high = softplus(alpha * (cx - (1 - s/2)), alpha)
        pen_y_low = softplus(alpha * (s/2 - cy), alpha)
        pen_y_high = softplus(alpha * (cy - (1 - s/2)), alpha)
        
        obj += lam * (np.sum(pen_x_low) + np.sum(pen_x_high) + np.sum(pen_y_low) + np.sum(pen_y_high))
        
        # Overlap penalties
        # For each pair
        total_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sx = (s[i] + s[j]) / 2
                
                # If dx < sx and dy < sy, they overlap
                # Penalty should be 0 if dx >= sx OR dy >= sy
                # Use product of softpluses
                # p_x = softplus(alpha * (sx - dx))
                # p_y = softplus(alpha * (sy - dy))
                # penalty = p_x * p_y
                # This is 0 if sx-dx is very negative (separated)
                
                px = softplus(alpha * (sx - dx), alpha)
                py = softplus(alpha * (sx - dy), alpha) # Wait, sy is same as sx for squares? No, s_i + s_j is same.
                # Yes, t = (s_i+s_j)/2 is the half-sum.
                
                total_pen += px * py
                
        obj += lam * total_pen
        
        return obj

    start_time = time.time()
    deadline = start_time + 580
    
    best_val = 0.0
    best_centers = None
    best_sides = None
    
    # 1. Try Differential Evolution for global search
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 1.0)) # cx
        bounds.append((1e-4, 1.0)) # cy
        bounds.append((1e-4, 1.0)) # s
        
    try:
        # DE is stochastic, so we run it for a limited time
        # popsize=20, maxiter=200
        # This might take a while. Let's check time.
        # 30 vars. 20 pop. 200 iter. 4000 evals.
        # Each eval is ~45 pair checks. 45 * 2 softplus. Fast.
        
        res_de = differential_evolution(
            objective, bounds, 
            maxiter=100, popsize=15, 
            tol=1e-6, seed=42, 
            polish=True, # This runs SLSQP on the best result
            init='latinhypercube'
        )
        
        if time.time() < deadline - 10:
            c, s = from_flat(res_de.x)
            if is_valid(c, s):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_centers = c
                    best_sides = s
                    
    except Exception as e:
        pass

    # 2. Try specific structured initial guesses with SLSQP polish
    # Guess 1: 4-3-3 Rows
    centers1 = np.zeros((n, 2))
    sides1 = np.zeros(n)
    s1 = 0.25
    sides1[0:4] = s1
    centers1[0:4, 1] = s1 / 2
    centers1[0:4, 0] = np.array([0.125, 0.375, 0.625, 0.875])
    
    s2 = 1.0 / 3.0
    sides1[4:7] = s2
    centers1[4:7, 1] = 0.25 + s2 / 2
    centers1[4:7, 0] = np.array([1.0/6.0, 0.5, 5.0/6.0])
    
    s3 = 1.0 / 3.0
    sides1[7:10] = s3
    centers1[7:10, 1] = 0.25 + s2 + s3 / 2
    centers1[7:10, 0] = np.array([1.0/6.0, 0.5, 5.0/6.0])
    
    # Guess 2: 5-5 Rows (perturbed)
    centers2 = np.zeros((n, 2))
    sides2 = np.zeros(n)
    s_b = 0.2
    sides2[0:5] = s_b
    centers2[0:5, 1] = s_b / 2
    centers2[0:5, 0] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    s_t = 0.3
    sides2[5:10] = s_t
    centers2[5:10, 1] = s_b + s_t / 2
    centers2[5:10, 0] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Guess 3: Random valid-ish
    np.random.seed(42)
    centers3 = np.random.rand(n, 2)
    sides3 = np.random.rand(n) * 0.1 + 0.1
    
    guesses = [
        (centers1, sides1),
        (centers2, sides2),
        (centers3, sides3)
    ]
    
    for c_init, s_init in guesses:
        if time.time() > deadline - 20:
            break
            
        x0 = to_flat(c_init, s_init)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-9})
            
            c, s = from_flat(res.x)
            # Clean up
            s = np.maximum(s, 1e-6)
            c[:, 0] = np.clip(c[:, 0], s/2, 1 - s/2)
            c[:, 1] = np.clip(c[:, 1], s/2, 1 - s/2)
            
            if is_valid(c, s):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_centers = c
                    best_sides = s
        except Exception as e:
            pass

    # 3. If we still have time, do a final polish on the best result
    if best_centers is not None and time.time() < deadline - 5:
        x_best = to_flat(best_centers, best_sides)
        try:
            res_final = minimize(objective, x_best, method='SLSQP', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12})
            c, s = from_flat(res_final.x)
            s = np.maximum(s, 1e-6)
            c[:, 0] = np.clip(c[:, 0], s/2, 1 - s/2)
            c[:, 1] = np.clip(c[:, 1], s/2, 1 - s/2)
            
            if is_valid(c, s):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_centers = c
                    best_sides = s
        except Exception as e:
            pass

    # Fallback if no valid found (should not happen)
    if best_centers is None:
        best_centers = centers1
        best_sides = sides1

    return best_centers, best_sides
