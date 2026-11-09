# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 620  # leave 10s margin before 630s kill

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def objective(c, s):
        return np.sum(s)

    def cost_function(z):
        """
        z: flattened array of length 3*n [x0, y0, s0, x1, y1, s1, ...]
        Returns: -sum(s) + penalty
        """
        c = z[:2*n].reshape(n, 2)
        s = z[2*n:]
        
        # Ensure s is positive for calculation stability
        s_pos = np.maximum(s, 1e-6)
        
        # Objective: maximize sum(s) => minimize -sum(s)
        obj_val = -np.sum(s_pos)
        
        # Penalties
        penalty = 0.0
        
        # Boundary penalties
        # x - s/2 >= 0 => x - s/2 < 0 is violation
        # We want to push x into [s/2, 1-s/2]
        for i in range(n):
            cx, cy = c[i]
            si = s_pos[i]
            
            # Left boundary
            if cx - si/2 < 0:
                penalty += (0 - (cx - si/2))**2
            # Right boundary
            if cx + si/2 > 1:
                penalty += ((cx + si/2) - 1)**2
            # Bottom boundary
            if cy - si/2 < 0:
                penalty += (0 - (cy - si/2))**2
            # Top boundary
            if cy + si/2 > 1:
                penalty += ((cy + si/2) - 1)**2

        # Overlap penalties
        # For each pair, if they overlap, add penalty proportional to overlap depth
        for i in range(n):
            for j in range(i + 1, n):
                cx_i, cy_i = c[i]
                cx_j, cy_j = c[j]
                si = s_pos[i]
                sj = s_pos[j]
                
                # Separation required:
                # |cx_i - cx_j| >= (si + sj)/2  OR  |cy_i - cy_j| >= (si + sj)/2
                # If neither is true, they overlap.
                
                dx = abs(cx_i - cx_j)
                dy = abs(cy_i - cy_j)
                
                req_dx = (si + sj) / 2
                req_dy = (si + sj) / 2
                
                # Calculate how much they "violate" the separation
                # We can use a smooth min of the two separation constraints
                # violation_x = max(0, req_dx - dx)
                # violation_y = max(0, req_dy - dy)
                # If both are > 0, they overlap.
                # A common penalty is max(0, min(violation_x, violation_y))? No, that's not right.
                # If they overlap, both dx < req_dx AND dy < req_dy.
                # The "depth" of overlap in x is req_dx - dx. In y is req_dy - dy.
                # We want to penalize if they are too close in BOTH dimensions.
                
                # Let's use a penalty that is active only when they overlap.
                # Overlap happens if dx < req_dx AND dy < req_dy.
                # We can define penalty_x = max(0, req_dx - dx)
                # penalty_y = max(0, req_dy - dy)
                # If we just sum these, we penalize even if they are separated in one dim.
                # We want to penalize the "intersection".
                
                # A better way:
                # If dx >= req_dx or dy >= req_dy, penalty is 0.
                # Else, penalty is some function of the overlap.
                
                if dx < req_dx and dy < req_dy:
                    # They overlap
                    # Calculate the "soft" overlap area or just push them apart.
                    # Simple penalty: square of the minimum slack?
                    # Or just sum of squared violations?
                    # Let's use a penalty that encourages separation in the "tightest" dimension or both.
                    # penalty = (req_dx - dx)**2 + (req_dy - dy)**2
                    # This is smooth and drives them apart.
                    penalty += (req_dx - dx)**2 + (req_dy - dy)**2
                else:
                    # No overlap, no penalty.
                    # But to make it smooth for gradient-based methods, we might want a soft transition.
                    # However, L-BFGS-B can handle non-smoothness somewhat, but smooth is better.
                    # Let's stick with hard condition for now, or use a smooth max.
                    # For now, the hard check is fine for the penalty magnitude, 
                    # but the derivative is discontinuous.
                    # To improve smoothness, we can use:
                    # pen_x = max(0, req_dx - dx)
                    # pen_y = max(0, req_dy - dy)
                    # penalty = pen_x * pen_y ? No.
                    # Let's use a smooth approximation of the "AND" condition.
                    # Actually, the current quadratic penalty when overlapping is sufficient for local search.
                    pass

        # Weight for penalty
        lambda_pen = 100.0
        return obj_val + lambda_pen * penalty

    def random_init():
        """Generate a random initial configuration."""
        # Random sides
        s = np.random.uniform(0.1, 0.5, n)
        # Random centers
        c = np.zeros((n, 2))
        for i in range(n):
            c[i, 0] = np.random.uniform(s[i]/2, 1 - s[i]/2)
            c[i, 1] = np.random.uniform(s[i]/2, 1 - s[i]/2)
        return c, s

    def refine_local(c, s, max_rounds=50):
        """
        Local refinement: try to increase each square's size slightly 
        and adjust position to maintain feasibility.
        """
        c = c.copy()
        s = s.copy()
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                old_c = c[i].copy()
                old_s = s[i]
                
                # Max possible size based on boundaries
                max_s = min(2*old_c[0], 2*(1-old_c[0]), 2*old_c[1], 2*(1-old_c[1])) + 2*tol
                max_s = min(max_s, 1.0)
                
                if max_s <= old_s + 1e-12:
                    continue
                
                best_s = old_s
                best_c = old_c.copy()
                
                lo, hi = old_s, max_s
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if mid <= best_s + 1e-13:
                        break
                    
                    found = False
                    # Try current position
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid + s[j]) / 2 - tol
                        if abs(old_c[0] - c[j][0]) < t and abs(old_c[1] - c[j][1]) < t:
                            ok = False
                            break
                    if ok and old_c[0] - mid/2 >= -tol and old_c[0] + mid/2 <= 1+tol \
                            and old_c[1] - mid/2 >= -tol and old_c[1] + mid/2 <= 1+tol:
                        found = True
                        best_c = old_c.copy()
                    
                    if not found:
                        # Try random positions
                        for _ in range(50):
                            nx = np.random.uniform(mid/2, 1-mid/2)
                            ny = np.random.uniform(mid/2, 1-mid/2)
                            ok = True
                            for j in range(n):
                                if j == i:
                                    continue
                                t = (mid + s[j]) / 2 - tol
                                if abs(nx - c[j][0]) < t and abs(ny - c[j][1]) < t:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                best_c = np.array([nx, ny])
                                break
                    
                    if found:
                        best_s = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    c[i] = best_c
                    improved = True
            
            if not improved:
                break
        
        return c, s

    best_c = None
    best_s = None
    best_val = -1
    
    # Run multiple restarts
    np.random.seed(42)
    num_restarts = 50
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        c0, s0 = random_init()
        z0 = np.concatenate([c0.flatten(), s0])
        
        # Bounds for L-BFGS-B
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 1)) # s
        
        try:
            # Use L-BFGS-B for large scale
            res = minimize(cost_function, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success or res.fun < 1e-5: # If it converged or is very good
                z_opt = res.x
                c_opt = z_opt[:2*n].reshape(n, 2)
                s_opt = z_opt[2*n:]
                
                # Check feasibility
                if is_feasible(c_opt, s_opt):
                    val = objective(c_opt, s_opt)
                    if val > best_val:
                        best_val = val
                        best_c = c_opt.copy()
                        best_s = s_opt.copy()
                        
                        # Local refinement
                        c_ref, s_ref = refine_local(c_opt, s_opt, max_rounds=20)
                        if is_feasible(c_ref, s_ref):
                            val_ref = objective(c_ref, s_ref)
                            if val_ref > best_val:
                                best_val = val_ref
                                best_c = c_ref.copy()
                                best_s = s_ref.copy()
        except Exception:
            pass

    # If no solution found, use a simple grid fallback
    if best_c is None:
        centers = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
            [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        ], dtype=np.float64)
        sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        best_c = centers
        best_s = sides
    
    # Final polish
    c_final, s_final = refine_local(best_c, best_s, max_rounds=100)
    if is_feasible(c_final, s_final):
        val_final = objective(c_final, s_final)
        if val_final > best_val:
            best_val = val_final
            best_c = c_final.copy()
            best_s = s_final.copy()
    
    best_s = np.maximum(best_s, 1e-9)
    
    return best_c, best_s

# EVOLVE-BLOCK-END
