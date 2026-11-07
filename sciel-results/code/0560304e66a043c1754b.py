"""
Root finder for 3 fixed 2-D systems: Global-Local Broyden Strategy.

Strategy:
1. Global Phase: Perform a coarse scan (grid + random) to find a point with
   low ||F(x)||. This identifies the basin of attraction.
2. Local Phase: Use Broyden's method (quasi-Newton) with aggressive
   backtracking from the best global candidate. Broyden is chosen over
   Newton to avoid the cost of finite-difference Jacobians (which cost 2
   extra calls per iteration) and over Nelder-Mead because it converges
   quadratically near the root if started close enough.
3. Polish Phase: When Broyden gets close (||F|| < 1e-4), switch to a 
   damped Newton method with central differences for rapid, guaranteed
   convergence to 1e-8.
4. Fallback: If the primary local search fails, try a few other top global
   candidates.

This approach trades some initial global exploration calls for a much more
efficient local convergence, aiming to stay well under the 500 call limit
while robustly handling systems with varying basin structures.
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

def _broyden_local(g, x0, max_it=40):
    """
    Broyden's method with backtracking.
    Returns (x, success, f_val)
    """
    x = np.array(x0, dtype=float)
    J = np.eye(2)
    
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        # Switch to Newton polish if we are close enough
        if f < 1e-4:
            x_pol, success, f_pol = _newton_polish(g, x, Fx)
            if success:
                return x_pol, True, f_pol
            # If polish fails, continue with Broyden from the polished x
            x = x_pol
            Fx = g(x)
            if Fx is None:
                break
            f = float(np.linalg.norm(Fx))
            if f <= 1e-8:
                return x, True, f
            continue

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
            
            # Accept if we make progress (Armijo-like condition simplified)
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
                break
            alpha *= 0.5
            
        if not success:
            break
            
    return x, False, f

def _newton_polish(g, x0, Fx0, max_it=10):
    """
    Damped Newton with central differences for final convergence.
    Uses 3 calls per iteration (F(x), F(x+h), F(x-h)).
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

def _global_scan(g, x0, budget=50):
    """
    Fallback global search.
    Returns list of (f_val, x) sorted by f_val.
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

    # 2. Try Broyden from top candidates
    # Ensure x0 is in the list if it wasn't scanned
    top_candidates = [c[1] for c in candidates[:5]]
    if not any(np.allclose(c, x0) for c in top_candidates):
        top_candidates.append(x0)
        
    # Deduplicate
    unique_candidates = []
    for c in top_candidates:
        if not any(np.allclose(c, uc) for uc in unique_candidates):
            unique_candidates.append(c)
            
    for x_start in unique_candidates:
        if g.n >= limit - 30 or _time_left() < 5.0:
            break
            
        x_res, success, f_res = _broyden_local(g, x_start, max_it=40)
        x_res = _clean(x_res)
        
        if x_res is not None:
            # Check if we found a root
            if f_res <= 1e-8:
                return x_res
                
            # Update best
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
                
            # If we got very close but didn't hit 1e-8, try a quick Newton polish
            if f_res < 1e-3 and g.n < limit - 10 and _time_left() > 2.0:
                x_pol, success_pol, f_pol = _newton_polish(g, x_res, g(x_res) if g(x_res) is not None else np.zeros(2))
                x_pol = _clean(x_pol)
                if x_pol is not None and f_pol <= 1e-8:
                    return x_pol
                if x_pol is not None and f_pol < g.best_f:
                    g.best_f = f_pol
                    g.best_x = x_pol.copy()

    # Return best found
    if g.best_x is not None:
        return g.best_x
    return x0.copy()
