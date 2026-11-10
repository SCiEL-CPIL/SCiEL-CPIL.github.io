import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _objective(centers, sides):
    """Return sum of sides if valid, else a large penalty."""
    n = len(sides)
    tol = 1e-7
    
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return -1e6
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return -1e6
        if s <= 0:
            return -1e6
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return -1e6
    
    return np.sum(sides)

def _smooth_penalty(centers, sides, beta=50.0):
    """
    Smooth surrogate for the constraint violations.
    Uses soft-min (log-sum-exp) to approximate the minimum separation.
    Returns a scalar penalty that is 0 when all constraints are satisfied (approximately).
    """
    n = len(sides)
    penalty = 0.0
    
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j])/2
            
            gap_x = dx - t
            gap_y = dy - t
            
            # Soft min of gap_x and gap_y
            # We want to penalize when min(gap_x, gap_y) < 0
            # min(a, b) approx = -1/beta * log(exp(-beta*a) + exp(-beta*b))
            # If min < 0, penalty > 0.
            
            # To avoid overflow in exp, we use the stable log-sum-exp formula:
            # log(exp(x) + exp(y)) = m + log(exp(x-m) + exp(y-m)) where m = max(x,y)
            # Here we have log(exp(-beta*gap_x) + exp(-beta*gap_y))
            # Let a = -beta*gap_x, b = -beta*gap_y
            # m = max(a, b) = -beta * min(gap_x, gap_y)
            
            m = min(gap_x, gap_y)
            
            # If m is very large positive, gaps are large, penalty is negligible
            if m > 100 / beta:
                continue
                
            # Calculate log(exp(-beta*gap_x) + exp(-beta*gap_y))
            # = -beta*m + log(exp(-beta*(gap_x - m)) + exp(-beta*(gap_y - m)))
            
            # Note: gap_x - m >= 0 and gap_y - m >= 0
            # So -beta*(gap_x - m) <= 0, no overflow
            
            val = -beta * m + np.log(np.exp(-beta * (gap_x - m)) + np.exp(-beta * (gap_y - m)))
            penalty += val / beta
    
    # Boundary penalties
    for i in range(n):
        s = sides[i]
        # Gaps to boundaries:
        # Left: cx - s/2
        # Right: 1 - cx - s/2
        # Bottom: cy - s/2
        # Top: 1 - cy - s/2
        
        gaps = [
            centers[i, 0] - s/2,
            1 - centers[i, 0] - s/2,
            centers[i, 1] - s/2,
            1 - centers[i, 1] - s/2
        ]
        
        # We want to penalize if any gap < 0.
        # Use soft-min of all gaps.
        m = min(gaps)
        
        if m > 100 / beta:
            continue
            
        # log(sum(exp(-beta * gap)))
        # = -beta*m + log(sum(exp(-beta*(gap - m))))
        
        gaps_arr = np.array(gaps)
        val = -beta * m + np.log(np.sum(np.exp(-beta * (gaps_arr - m))))
        penalty += val / beta
        
    return penalty

def _generate_initial_configs():
    """Generate several diverse initial configurations."""
    configs = []
    
    # Config 1: Standard 3.0 packing (2 large, 8 small)
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    sides[0] = 0.5; centers[0] = [0.25, 0.25]
    sides[1] = 0.5; centers[1] = [0.75, 0.25]
    for i in range(8):
        sides[i+2] = 0.25
        row = i // 4
        col = i % 4
        centers[i+2] = [0.125 + col*0.25, 0.625 + row*0.25]
    configs.append((centers, sides))
    
    # Config 2: 5x2 Grid (10 squares of 0.2)
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    for i in range(10):
        row = i // 5
        col = i % 5
        sides[i] = 0.2
        centers[i] = [0.1 + col*0.2, 0.1 + row*0.2]
    configs.append((centers, sides))
    
    # Config 3: Random Greedy
    rng = np.random.RandomState(42)
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    # Sort by size descending? No, just place them.
    # Let's try a simple random placement and then a quick repair
    for i in range(10):
        sides[i] = 0.2 + rng.random()*0.1
        centers[i] = [rng.uniform(sides[i]/2, 1-sides[i]/2), rng.uniform(sides[i]/2, 1-sides[i]/2)]
    configs.append((centers, sides))
    
    # Config 4: One large, rest small
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    sides[0] = 0.6
    centers[0] = [0.3, 0.3]
    for i in range(1, 10):
        sides[i] = 0.2
        # Place around the large one
        angle = 2 * np.pi * i / 9
        r = 0.4
        cx = 0.5 + r * np.cos(angle)
        cy = 0.5 + r * np.sin(angle)
        # Clamp
        cx = np.clip(cx, sides[i]/2, 1-sides[i]/2)
        cy = np.clip(cy, sides[i]/2, 1-sides[i]/2)
        centers[i] = [cx, cy]
    configs.append((centers, sides))
    
    return configs

def _local_search_smooth(centers, sides, time_limit=10.0, beta_start=10.0, beta_end=100.0):
    """
    Uses scipy.optimize.minimize with a smooth penalty function.
    We minimize -sum(sides) + penalty(centers, sides).
    """
    start_time = time.time()
    n = len(sides)
    
    # Flatten variables: [cx0, cy0, s0, cx1, cy1, s1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = sides[i]
    
    best_x = x0.copy()
    best_score = _objective(centers, sides)
    
    if best_score < 0:
        return centers, sides, best_score
    
    def unpack(x):
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(n):
            c[i, 0] = x[3*i]
            c[i, 1] = x[3*i+1]
            s[i] = x[3*i+2]
        return c, s
    
    def make_objective(beta):
        def obj(x):
            if time.time() - start_time > time_limit:
                return 1e6
            c, s = unpack(x)
            
            # Hard constraints for bounds to keep search space sane
            if np.any(s < 0.01) or np.any(s > 1.0):
                return 1e6
            
            pen = _smooth_penalty(c, s, beta=beta)
            val = -np.sum(s) + pen
            return val
        return obj
    
    # Run multiple steps with increasing beta (annealing the smoothness)
    betas = np.linspace(beta_start, beta_end, 5)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        obj = make_objective(beta)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})
            if res.success or res.fun < obj(x0):
                x0 = res.x
        except Exception:
            pass
        
        # Update best
        c, s = unpack(x0)
        # Clip sides to be positive
        s = np.clip(s, 0.01, 1.0)
        # Clip centers
        for i in range(n):
            c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - s[i]/2)
            c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)
        
        score = _objective(c, s)
        if score > best_score:
            best_score = score
            best_x = x0.copy()
            
    c, s = unpack(best_x)
    return c, s, best_score

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    
    Uses a hybrid approach:
    1. Generate diverse initial configurations.
    2. Use smooth optimization (L-BFGS-B with soft-min penalty) to refine.
    3. Track the best valid solution found.
    """
    deadline = time.time() + 600  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    configs = _generate_initial_configs()
    
    # Add some random perturbations of the best config found so far
    rng = np.random.RandomState(123)
    
    for idx, (centers, sides) in enumerate(configs):
        if time.time() > deadline - 30:
            break
            
        # Run smooth local search
        time_left = deadline - time.time() - 30
        if time_left < 5:
            break
            
        c_new, s_new, score_new = _local_search_smooth(
            centers, sides, 
            time_limit=min(20.0, time_left / 2), 
            beta_start=5.0, 
            beta_end=50.0
        )
        
        if score_new > best_score:
            best_score = score_new
            best_centers = c_new.copy()
            best_sides = s_new.copy()
            
        # If we found a good solution, try to perturb it and refine again
        if best_score > 3.0 and time.time() < deadline - 60:
            c_pert = best_centers + rng.normal(0, 0.02, (10, 2))
            s_pert = best_sides * (1 + rng.normal(0, 0.05, 10))
            s_pert = np.clip(s_pert, 0.05, 0.6)
            for i in range(10):
                c_pert[i, 0] = np.clip(c_pert[i, 0], s_pert[i]/2, 1 - s_pert[i]/2)
                c_pert[i, 1] = np.clip(c_pert[i, 1], s_pert[i]/2, 1 - s_pert[i]/2)
                
            c_new, s_new, score_new = _local_search_smooth(
                c_pert, s_pert, 
                time_limit=min(20.0, deadline - time.time() - 30), 
                beta_start=10.0, 
                beta_end=100.0
            )
            
            if score_new > best_score:
                best_score = score_new
                best_centers = c_new.copy()
                best_sides = s_new.copy()
                
    # Final polish with Nelder-Mead on the best solution
    if best_centers is not None and time.time() < deadline - 10:
        def nm_objective(x):
            c = np.zeros((10, 2))
            s = np.zeros(10)
            for i in range(10):
                c[i, 0] = x[3*i]
                c[i, 1] = x[3*i+1]
                s[i] = x[3*i+2]
            if np.any(s < 0.01) or np.any(s > 1.0):
                return 1e6
            # Use a hard penalty for NM to be safe
            if not _is_valid(c, s):
                return 1e6
            return -np.sum(s)
        
        x0 = np.zeros(30)
        for i in range(10):
            x0[3*i] = best_centers[i, 0]
            x0[3*i+1] = best_centers[i, 1]
            x0[3*i+2] = best_sides[i]
            
        try:
            res = minimize(nm_objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score: # i.e., -sum(s) < -best_score => sum(s) > best_score
                c = np.zeros((10, 2))
                s = np.zeros(10)
                for i in range(10):
                    c[i, 0] = res.x[3*i]
                    c[i, 1] = res.x[3*i+1]
                    s[i] = res.x[3*i+2]
                if _is_valid(c, s):
                    score = np.sum(s)
                    if score > best_score:
                        best_score = score
                        best_centers = c
                        best_sides = s
        except Exception:
            pass

    return best_centers, best_sides

if __name__ == "__main__":
    centers, sides = squares_in_square10()
    print("Score:", np.sum(sides))
    print("Valid:", _is_valid(centers, sides))
