"""
Root finder for 3 fixed 2-D systems: Secant-Step Hybrid with Re-anchoring.

Strategy:
1. Initialize Jacobian via a two-point secant difference (2 calls) for a strong start.
2. Alternate between standard Broyden rank-1 updates and a full 3-call finite-difference
   Jacobian recomputation every k iterations to correct accumulated approximation error.
3. Use backtracking line search on the norm of F to ensure robustness.
4. Multi-start basin hopping to identify the basin of attraction efficiently.

This approach minimizes the number of F calls by leveraging the efficiency of Broyden's
method while preventing drift through periodic re-anchoring with exact Jacobians.
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

def _fd_jacobian(g, x, h=1e-6):
    """
    Compute finite-difference Jacobian (3 calls: base + 2 perturbations).
    Returns J (2x2) or None if budget exhausted.
    """
    if g.n >= g.limit - 3:
        return None
    
    Fx = g(x)
    if Fx is None:
        return None
    
    J = np.zeros((2, 2))
    
    for i in range(2):
        xp = x.copy()
        xp[i] += h
        Fp = g(xp)
        if Fp is None:
            return None
        J[:, i] = (Fp - Fx) / h
    
    return J

def _broyden_local(g, x0, max_it=30, reanchor_every=3):
    """
    Local search using Broyden's method with periodic re-anchoring.
    Initializes J via secant/FD, then alternates Broyden updates and FD recomputation.
    """
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    # Initialize Jacobian
    J = _fd_jacobian(g, x)
    if J is None:
        J = np.eye(2)

    for it in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        # Re-anchor every k iterations
        if it > 0 and it % reanchor_every == 0:
            J_new = _fd_jacobian(g, x)
            if J_new is not None:
                J = J_new
                Fx = g(x)
                if Fx is None:
                    break
                f = float(np.linalg.norm(Fx))
                if f <= 1e-8:
                    return x, True, f

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, use steepest descent
            dx = -Fx
            norm_dx = np.linalg.norm(dx)
            if norm_dx < 1e-12:
                break
            dx = dx / norm_dx * 0.1
            J = np.eye(2)

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
            
            # Accept if we make progress
            if f_new < f:
                # Broyden update
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

def _basin_hopping_scan(g, x0, budget=30):
    """
    Perform a series of short local searches from different starting points 
    to find the basin of attraction.
    """
    x0 = np.array(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0))
    
    # First, try x0
    x_res, success, f_res = _broyden_local(g, x0, max_it=10, reanchor_every=3)
    if f_res < g.best_f:
        g.best_f = f_res
        g.best_x = x_res.copy()
    if success:
        return x_res, True, f_res

    # Try a few other points
    directions = [
        np.array([1, 0]), np.array([-1, 0]),
        np.array([0, 1]), np.array([0, -1]),
        np.array([1, 1]), np.array([-1, -1]),
        np.array([1, -1]), np.array([-1, 1])
    ]
    
    for d in directions:
        if g.n >= budget + 20:
            break
        x_start = x0 + d * scale
        x_res, success, f_res = _broyden_local(g, x_start, max_it=10, reanchor_every=3)
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
        if success:
            return x_res, True, f_res
            
    return g.best_x, False, g.best_f

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Basin Hopping to find a good starting point
    x_start, success, f_start = _basin_hopping_scan(g, x0, budget=30)
    
    if success:
        return x_start
        
    # 2. If we found a good point (low f) but not a root, refine it
    if g.best_x is not None and g.best_f < 1e-3:
        # Run a longer local search from the best point found
        x_res, success, f_res = _broyden_local(g, g.best_x, max_it=50, reanchor_every=3)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 3. If still not found, try a more global random search
    if g.best_f > 1e-3 and g.n < limit - 50:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if g.n >= limit - 20 or _time_left() < 5.0:
                break
            x_rand = x0 + rng.uniform(-5, 5, size=2) * max(1.0, np.linalg.norm(x0))
            x_res, success, f_res = _broyden_local(g, x_rand, max_it=10, reanchor_every=3)
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
            if success:
                return x_res
                
    # Return best found
    if g.best_x is not None:
        return g.best_x
    return x0.copy()
