import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 axis-aligned squares in the unit square.
    Uses a penalty-based optimization approach with multiple random restarts.
    """
    n = 10
    
    # Objective function to minimize: -sum(sides) + penalty(overlap)
    def objective(z):
        # z shape: (30,) -> [x1, y1, s1, x2, y2, s2, ...]
        x = z[0::3]
        y = z[1::3]
        s = z[2::3]
        
        # Maximize sum of sides -> Minimize negative sum
        obj = -np.sum(s)
        
        # Penalty for overlaps and out-of-bounds
        penalty = 0.0
        
        # Bounds penalty (soft, but L-BFGS-B handles hard bounds, so this is backup)
        # We rely on L-BFGS-B bounds for x, y, s.
        
        # Overlap penalty
        # For each pair i, j
        for i in range(n):
            for j in range(i + 1, n):
                # Required distance to not overlap
                req_x = (s[i] + s[j]) / 2.0
                req_y = (s[i] + s[j]) / 2.0
                
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                
                # If dx < req_x and dy < req_y, they overlap
                # We use a smooth penalty based on the amount of overlap
                # overlap_x = max(0, req_x - dx)
                # overlap_y = max(0, req_y - dy)
                # If both > 0, area is overlap_x * overlap_y
                # To make it differentiable, we can square the max terms
                ox = max(0.0, req_x - dx)
                oy = max(0.0, req_y - dy)
                
                # If they overlap in both dimensions, penalize
                # Using square of the product or sum of squares
                # Sum of squares is safer for gradient
                penalty += (ox * ox) + (oy * oy)
                
        # Weight for penalty
        # If penalty is non-zero, we want it to dominate
        # But if we use L-BFGS-B, we can just add a large multiplier
        # However, if the initial guess is bad, the gradient might be noisy
        # Let's use a dynamic weight or a very large static one
        # A common trick is to add penalty^2 or just penalty * 1000
        obj += 1000.0 * penalty
        
        return obj

    # Bounds:
    # x in [0, 1], y in [0, 1], s in [0, 1]
    # Actually, s can be up to 1, but x+s/2 <= 1 -> x <= 1 - s/2.
    # L-BFGS-B takes simple bounds.
    # If we set s in [0,1] and x in [0,1], we might violate x+s/2 <= 1.
    # We should include boundary violations in the penalty or adjust bounds.
    # Easier: Add boundary penalty to objective.
    
    def objective_with_bounds(z):
        x = z[0::3]
        y = z[1::3]
        s = z[2::3]
        
        obj = -np.sum(s)
        penalty = 0.0
        
        # Boundary penalties
        # x - s/2 >= 0 -> x >= s/2
        # x + s/2 <= 1 -> x <= 1 - s/2
        # Same for y
        
        # Violation if x < s/2
        vx = np.maximum(0.0, s / 2.0 - x)
        penalty += np.sum(vx * vx)
        
        # Violation if x > 1 - s/2
        vx = np.maximum(0.0, x - (1.0 - s / 2.0))
        penalty += np.sum(vx * vx)
        
        vy = np.maximum(0.0, s / 2.0 - y)
        penalty += np.sum(vy * vy)
        
        vy = np.maximum(0.0, y - (1.0 - s / 2.0))
        penalty += np.sum(vy * vy)
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                req_x = (s[i] + s[j]) / 2.0
                req_y = (s[i] + s[j]) / 2.0
                
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                
                ox = max(0.0, req_x - dx)
                oy = max(0.0, req_y - dy)
                
                penalty += (ox * ox) + (oy * oy)
                
        obj += 1000.0 * penalty
        return obj

    # Initial guesses
    # 1. Grid (current baseline)
    # 2. Random
    # 3. Structured: 4 corners, 1 center, etc.
    
    def get_initial_guess(type):
        if type == 0:
            # Grid
            k = 4
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            s = np.full(n, side)
            # Convert to bottom-left
            x = cx - s / 2.0
            y = cy - s / 2.0
            z = np.array([x[0], y[0], s[0], x[1], y[1], s[1], ...]) # Need to construct properly
            # Let's write a helper to flatten
            z_flat = np.zeros(n * 3)
            z_flat[0::3] = x
            z_flat[1::3] = y
            z_flat[2::3] = s
            return z_flat
            
        elif type == 1:
            # Random
            s = np.random.uniform(0.1, 0.4, n)
            # Place randomly
            x = np.random.uniform(0, 1, n)
            y = np.random.uniform(0, 1, n)
            # Clip to be inside?
            # Just let optimizer fix it
            z_flat = np.zeros(n * 3)
            z_flat[0::3] = x
            z_flat[1::3] = y
            z_flat[2::3] = s
            return z_flat
            
        elif type == 2:
            # "Plus" shape: 1 center, 4 around it, 5 more?
            # Let's try: 4 large in corners, 1 in center, 5 small in gaps?
            # Just use a perturbed grid
            k = 3
            side = 1.0 / k * 0.9 # Slightly smaller to allow movement
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            s = np.full(n, side)
            x = cx - s / 2.0
            y = cy - s / 2.0
            # Add small noise
            x += np.random.uniform(-0.01, 0.01, n)
            y += np.random.uniform(-0.01, 0.01, n)
            z_flat = np.zeros(n * 3)
            z_flat[0::3] = x
            z_flat[1::3] = y
            z_flat[2::3] = s
            return z_flat

    best_z = None
    best_score = -np.inf
    
    # Run optimization from several starts
    starts = [get_initial_guess(0), get_initial_guess(1), get_initial_guess(2), get_initial_guess(1), get_initial_guess(2)]
    
    bounds = [(0, 1) for _ in range(n * 3)] # x,y,s all in [0,1]
    
    for z0 in starts:
        try:
            res = minimize(objective_with_bounds, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            
            # Check validity of result
            z = res.x
            x = z[0::3]
            y = z[1::3]
            s = z[2::3]
            
            valid = True
            # Check bounds strictly
            for i in range(n):
                if x[i] < -1e-6 or y[i] < -1e-6:
                    valid = False
                if x[i] + s[i] > 1 + 1e-6 or y[i] + s[i] > 1 + 1e-6:
                    valid = False
                if s[i] < 1e-6:
                    valid = False
            
            if valid:
                # Check overlaps strictly
                for i in range(n):
                    for j in range(i + 1, n):
                        req_x = (s[i] + s[j]) / 2.0
                        req_y = (s[i] + s[j]) / 2.0
                        dx = abs(x[i] - x[j])
                        dy = abs(y[i] - y[j])
                        
                        # Overlap if dx < req_x AND dy < req_y
                        # Tolerance 1e-7
                        if dx < req_x - 1e-7 and dy < req_y - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(s)
                if score > best_score:
                    best_score = score
                    best_z = z.copy()
                    
        except Exception as e:
            pass

    if best_z is not None:
        x = best_z[0::3]
        y = best_z[1::3]
        s = best_z[2::3]
        
        # Convert back to centers
        cx = x + s / 2.0
        cy = y + s / 2.0
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = s.astype(np.float64)
        
        return centers, sides
    else:
        # Fallback to baseline
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

# EVOLVE-BLOCK-END
