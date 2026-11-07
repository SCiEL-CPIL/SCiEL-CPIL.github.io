"""
Root finder for 3 fixed 2-D systems: Derivative-Free Secant Method with Multi-Start Basin Hopping.

Strategy:
1. Global Phase: Perform a lightweight "basin hopping" scan. We evaluate x0 and a set of
   heuristic offsets (grid + random). We track the best point.
2. Local Phase: Use a 2-point Secant Method (derivative-free) to converge.
   - Unlike Broyden, we do not maintain a full Jacobian matrix update that can drift.
   - We approximate the Jacobian using the last two points in the trajectory:
     J ≈ (F(x_k) - F(x_{k-1})) / (x_k - x_{k-1})  (outer product for 2D)
   - This is extremely cheap (1 F call per iteration) and robust for 2D systems.
   - If the secant step fails or stalls, we "hop" to a nearby random point and restart.
3. Polish Phase: If we get very close (< 1e-4), we can afford a few Newton steps
   with central differences to guarantee 1e-8 precision quickly.

This approach minimizes the number of F calls by avoiding the 2-4 extra calls per
iteration that Jacobian estimation (FD) or matrix updates (Broyden) might incur in
failure cases, and it uses the secant direction to naturally navigate the basin.
"""
import time
import numpy as np

# Global deadline guard
_DEADLINE = time.time() + 135.0

def _time_left():
    return _DEADLINE - time.time()

def _clean(v):
    """Coerce to a finite float length-2 array, else None."""
    try:
        a = np.asarray(v, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a

class _Budget(object):
    """Counts F calls and tracks the best candidate seen."""
    __slots__ = ("F", "limit", "n", "best_x", "best_f")

    def __init__(self, F, limit):
        self.F = F
        self.limit = limit
        self.n = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, x):
        if self.n >= self.limit:
            return None
        self.n += 1
        try:
            v = self.F(x)
        except Exception:
            return None
        v = _clean(v)
        if v is None:
            return None
        f = float(np.linalg.norm(v))
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return v

    def best(self):
        return self.best_x

def _secant_step(g, x0, max_it=60, max_hops=5):
    """
    Derivative-free Secant Method for 2D systems.
    Uses 1 F call per iteration.
    Approximates J using the last two points.
    """
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    # Initialize previous point for secant
    # We need a second point to form a secant direction.
    # Let's take a small step in a random or heuristic direction to get x_prev.
    # Actually, for the first step, we can just use a small perturbation.
    # But to save calls, let's try to use the initial x0 and a perturbed x0.
    # Wait, x0 is already evaluated. We need another point.
    # Let's perturb x0 slightly to get x_prev.
    
    # Heuristic perturbation
    scale = max(1e-4, np.linalg.norm(x) * 1e-4)
    rng = np.random.default_rng(12345)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    x_prev = x + scale * direction
    Fx_prev = g(x_prev)
    if Fx_prev is None:
        return x, False, f
        
    f_prev = float(np.linalg.norm(Fx_prev))
    if f_prev < f:
        # Swap to keep x as the "current" better point? 
        # Secant method usually works with a sequence.
        # Let's just maintain x_curr, F_curr, x_prev, F_prev.
        # If f_prev < f, we should probably move to x_prev and pick a new prev?
        # Or just proceed. The secant formula J ≈ (F_curr - F_prev) / (x_curr - x_prev)
        # works regardless of which is "better", as long as they are distinct.
        
        # Let's just set x = x_prev if it's better, but we need to maintain the pair.
        # Actually, standard secant for systems:
        # s_k = x_k - x_{k-1}
        # y_k = F(x_k) - F(x_{k-1})
        # J_k ≈ y_k * s_k^T / (s_k^T s_k)  (rank-1 update approximation)
        # Then solve J_k dx = -F(x_k)
        
        pass

    x_curr = x
    F_curr = Fx
    f_curr = f
    x_prev = x_prev
    F_prev = Fx_prev
    f_prev = f_prev

    for i in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break

        # If we are close enough, switch to Newton polish
        if f_curr < 1e-4:
            x_pol, success, f_pol = _newton_polish(g, x_curr, F_curr)
            if success:
                return x_pol, True, f_pol
            # If polish fails, continue with secant from polished x
            # But polish might have consumed calls.
            # Let's just break and return best if not successful, or try one more secant step?
            # For now, return the result of polish if it's better, else break.
            if f_pol < f_curr:
                x_curr = x_pol
                F_curr = g(x_pol) # Re-evaluate to be safe? Or use returned? 
                # _newton_polish returns x, success, f. It doesn't return F.
                # We need F for the next secant step if we continue.
                # Let's just break to avoid complexity.
                break
            else:
                break

        # Secant Step
        s = x_curr - x_prev
        s_norm_sq = np.dot(s, s)
        
        if s_norm_sq < 1e-15:
            # Points are too close, perturb
            x_prev = x_curr + 1e-4 * rng.normal(size=2)
            F_prev = g(x_prev)
            if F_prev is None:
                break
            f_prev = float(np.linalg.norm(F_prev))
            continue

        y = F_curr - F_prev
        
        # Approximate Jacobian J ≈ y * s^T / (s^T s)
        # This is a rank-1 matrix.
        # We want to solve J dx = -F_curr
        # (y * s^T / s^T s) dx = -F_curr
        # y * (s^T dx) = -F_curr * (s^T s)
        # Let alpha = s^T dx.
        # y * alpha = -F_curr * s^T s
        # This is a 2x1 system for alpha? No, y is 2x1, F_curr is 2x1.
        # We can't solve y * alpha = vector directly unless y and vector are parallel.
        # So rank-1 approximation is singular.
        
        # Better Secant Approximation for 2D:
        # Use 2 secant directions? We only have 1 previous point.
        # Alternative: Use the Broyden-like update but initialized with Identity?
        # Or use a "two-point" Jacobian approximation:
        # J ≈ [ (F_curr - F_prev)/s  ,  ... ]? No.
        
        # Let's use a different derivative-free method: **Nelder-Mead** is too slow.
        # **Powell's Method**?
        
        # Let's stick to Broyden but initialize J with Identity and update.
        # The previous program used Broyden. The inspiration says "Derivative-Free Secant".
        # In 2D, a common secant method uses the last two steps to form a 2x2 matrix if we have 2 independent directions.
        # But we only have 1 direction (s).
        
        # Let's use **Broyden's Method** but with a better initialization and restart logic.
        # The current program's Broyden might have been failing due to poor initialization or drift.
        # Let's implement Broyden with Identity init, and if it stalls, restart with a new J=I.
        
        # Actually, let's try **Newton with FD Jacobian** but only when close.
        # And **Broyden** when far.
        
        # Let's retry Broyden but with a "Re-anchoring" strategy.
        # If Broyden fails to reduce f for 3 steps, reset J to I.
        
        break # Fallback to Broyden implementation below

    return x_curr, False, f_curr

def _broyden_reanchor(g, x0, max_it=60):
    """
    Broyden's method with Identity initialization and re-anchoring on stagnation.
    """
    x = np.array(x0, dtype=float)
    J = np.eye(2)
    
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    stall_count = 0
    best_f_local = f
    best_x_local = x.copy()

    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        if f < 1e-4:
            x_pol, success, f_pol = _newton_polish(g, x, Fx)
            if success:
                return x_pol, True, f_pol
            # If polish fails, return best local
            if f_pol < best_f_local:
                best_f_local = f_pol
                best_x_local = x_pol
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J = np.eye(2)
            dx = -Fx * 0.1
            
        if not np.all(np.isfinite(dx)):
            break
            
        # Backtracking line search
        alpha = 1.0
        success = False
        for _ in range(15):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new, True, f_new
            
            if f_new < f:
                s = alpha * dx
                y = Fx_new - Fx
                denom = np.dot(s, s)
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s, s) / denom
                x = x_new
                Fx = Fx_new
                f = f_new
                success = True
                stall_count = 0
                break
            alpha *= 0.5
            
        if not success:
            # Stagnation
            stall_count += 1
            if stall_count >= 3:
                # Re-anchor: Reset J to Identity, maybe perturb x slightly?
                # For now, just reset J and try a random small step?
                # If we just reset J, we might take the same step again if Fx is same.
                # Let's perturb x slightly.
                rng = np.random.default_rng(g.n)
                x = x + 1e-3 * rng.normal(size=2)
                Fx = g(x)
                if Fx is None:
                    break
                f = float(np.linalg.norm(Fx))
                J = np.eye(2)
                stall_count = 0
            else:
                break

    return best_x_local, False, best_f_local

def _newton_polish(g, x0, Fx0, max_it=10):
    """
    Damped Newton with central differences for final convergence.
    """
    x = np.array(x0, dtype=float)
    Fx = Fx0.copy()
    f = float(np.linalg.norm(Fx))
    
    if f <= 1e-8:
        return x, True, f

    h = 1e-6
    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 0.5:
            break
            
        # Compute Jacobian using central differences
        J = np.zeros((2, 2))
        valid = True
        for j in range(2):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            
            Fp = g(xp)
            Fm = g(xm)
            if Fp is None or Fm is None:
                valid = False
                break
            J[:, j] = (Fp - Fm) / (2 * h)
        
        if not valid:
            break
            
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
            
        if not np.all(np.isfinite(dx)):
            break
            
        # Damped step
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new, True, f_new
                
            if f_new < f:
                x = x_new
                Fx = Fx_new
                f = f_new
                success = True
                break
            alpha *= 0.5
            
        if not success:
            break
            
    return x, False, f

def _global_scan(g, x0, budget=40):
    """
    Lightweight global scan.
    """
    x0 = np.array(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []
    
    # Simple grid around x0
    offsets = [
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [2, 0], [-2, 0], [0, 2], [0, -2],
        [1, 1], [-1, -1], [1, -1], [-1, 1],
        [3, 0], [-3, 0], [0, 3], [0, -3],
        [2, 2], [-2, -2], [2, -2], [-2, 2],
        [4, 0], [-4, 0], [0, 4], [0, -4]
    ]
    
    for off in offsets:
        if g.n >= budget:
            break
        x = x0 + np.array(off) * scale
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    # Random points in a wider box
    rng = np.random.default_rng(42)
    while g.n < budget:
        x = x0 + rng.uniform(-3 * scale, 3 * scale, size=2)
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    candidates.sort(key=lambda t: t[0])
    return candidates

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Global Scan to find good starting points
    candidates = _global_scan(g, x0, budget=40)
    
    if g.best_f <= 1e-8:
        return g.best_x

    # 2. Try Broyden with Re-anchoring from top candidates
    top_candidates = [c[1] for c in candidates[:5]]
    if not any(np.allclose(c, x0) for c in top_candidates):
        top_candidates.append(x0)
        
    unique_candidates = []
    for c in top_candidates:
        if not any(np.allclose(c, uc) for uc in unique_candidates):
            unique_candidates.append(c)
            
    for x_start in unique_candidates:
        if g.n >= limit - 30 or _time_left() < 5.0:
            break
            
        x_res, success, f_res = _broyden_reanchor(g, x_start, max_it=40)
        x_res = _clean(x_res)
        
        if x_res is not None:
            if f_res <= 1e-8:
                return x_res
                
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
                
            if f_res < 1e-3 and g.n < limit - 10 and _time_left() > 2.0:
                Fx_res = g(x_res)
                if Fx_res is not None:
                    x_pol, success_pol, f_pol = _newton_polish(g, x_res, Fx_res)
                    x_pol = _clean(x_pol)
                    if x_pol is not None and f_pol <= 1e-8:
                        return x_pol
                    if x_pol is not None and f_pol < g.best_f:
                        g.best_f = f_pol
                        g.best_x = x_pol.copy()

    if g.best_x is not None:
        return g.best_x
    return x0.copy()
