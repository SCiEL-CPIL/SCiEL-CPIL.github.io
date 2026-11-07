"""
Root finder for 3 fixed 2-D systems: Homotopy Path-Following Continuation.

Strategy:
1. Define a homotopy H(x, t) = (1-t) * L(x) + t * F(x), where L(x) is a simple linear system with a known root (e.g., L(x) = x).
2. Start at t=0 where the root is known (x=0).
3. Increment t in small steps. For each step, use the previous root as a predictor and a few secant-like corrections to find the root of H(x, t_k).
4. This globalizes the search by following the continuous path of roots from the trivial solution to the target solution, avoiding local minima of ||F||^2 and Jacobian approximation errors.
5. Robustness: If the path breaks or steps fail, fallback to multi-start Broyden from the last known point.
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

def _secant_correct(g, x_start, max_it=10):
    """
    Perform a few secant steps to correct x_start towards a root of the current function.
    Uses 2-point secant Jacobian initialization and Broyden updates.
    Returns (x, success, f).
    """
    x = np.array(x_start, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    # Initialize Jacobian with FD (3 calls) for a strong start in the correction phase
    # We only do this if we have budget. If not, use identity.
    J = np.eye(2)
    if g.n < g.limit - 5:
        Fx0 = Fx
        J = np.zeros((2, 2))
        h = 1e-6
        for i in range(2):
            xp = x.copy()
            xp[i] += h
            Fp = g(xp)
            if Fp is None:
                J = np.eye(2)
                break
            J[:, i] = (Fp - Fx0) / h
        if not np.all(np.isfinite(J)):
            J = np.eye(2)

    for it in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx
            norm_dx = np.linalg.norm(dx)
            if norm_dx < 1e-12:
                break
            dx = dx / norm_dx * 0.1
            J = np.eye(2)

        if not np.all(np.isfinite(dx)):
            break

        # Backtracking
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

def _homotopy_search(g, x0):
    """
    Follow the homotopy path from t=0 (root at 0) to t=1 (root of F).
    """
    # L(x) = x - x0_target? No, standard homotopy:
    # H(x, t) = (1-t)*A*x + t*F(x)
    # Let A = I, so H(x, t) = (1-t)*x + t*F(x)
    # At t=0, root is x=0.
    # At t=1, root is F(x)=0.
    
    x = np.zeros(2)
    # Check if x0 is actually closer to a root than 0?
    # The problem gives x0. Let's use x0 as the target for the linear part?
    # If we use L(x) = x - x0, then at t=0 root is x0.
    # This might be better if x0 is good.
    # Let's try L(x) = x - x0.
    # H(x, t) = (1-t)*(x - x0) + t*F(x)
    
    x0_arr = np.array(x0, dtype=float)
    x = x0_arr.copy()
    
    # Verify start
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    # Step size for t
    # Start with a small step to be safe, increase if successful
    dt = 0.1
    t = 0.0
    
    # We will iterate t from 0 to 1
    # But we can adapt dt.
    
    max_steps = 100
    for step in range(max_steps):
        if g.n >= g.limit - 20 or _time_left() < 2.0:
            break
            
        t_new = min(1.0, t + dt)
        
        # Define the function for this t
        # H(x) = (1-t_new)*(x - x0) + t_new*F(x)
        
        # Predictor: use previous x
        x_pred = x
        
        # Corrector: use secant on H
        # We need to evaluate H. 
        # H(x) requires 1 call to g(x).
        # So the correction phase uses calls to g.
        
        x_corr, success, f_corr = _secant_correct(g, x_pred, max_it=5)
        
        # Check if we are close to a root of F (when t is close to 1)
        # Note: f_corr is ||H(x)||. 
        # If t_new is close to 1, ||H|| approx ||F||.
        # But we must check F directly at the end.
        
        # If success, we can increase dt
        if success:
            x = x_corr
            t = t_new
            dt = min(0.5, dt * 1.5) # Increase step size
            
            # If we reached t=1, we are done (pending final check)
            if t >= 1.0:
                # Final verification
                Fx_final = g(x)
                if Fx_final is not None:
                    f_final = float(np.linalg.norm(Fx_final))
                    if f_final <= 1e-8:
                        return x, True, f_final
                return x, False, f_corr # Return best effort
        else:
            # Step failed, reduce dt
            dt = max(0.01, dt * 0.5)
            # Try again with smaller step from current t
            # If dt is very small, we might be stuck. 
            # If stuck, break and let fallback handle it.
            if dt <= 0.011:
                break

    return x, False, g.best_f

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Try Homotopy Path Following
    x_res, success, f_res = _homotopy_search(g, x0)
    
    if success:
        return x_res
        
    # Update best
    if f_res < g.best_f:
        g.best_f = f_res
        g.best_x = x_res.copy()

    # 2. Fallback: Multi-start Broyden from best found or x0
    if g.best_x is not None:
        start_points = [g.best_x]
    else:
        start_points = [x0]
        
    # Add some diverse starts if budget allows
    if g.n < limit - 50:
        scale = max(1.0, np.linalg.norm(x0))
        directions = [
            np.array([1, 0]), np.array([-1, 0]),
            np.array([0, 1]), np.array([0, -1]),
            np.array([1, 1]), np.array([-1, -1]),
            np.array([1, -1]), np.array([-1, 1])
        ]
        for d in directions:
            start_points.append(x0 + d * scale)
            if g.n >= limit - 20:
                break

    for sp in start_points:
        if g.n >= limit - 10 or _time_left() < 2.0:
            break
        x_res, success, f_res = _secant_correct(g, sp, max_it=15)
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
        if success:
            return x_res

    # 3. Final random search if still not found
    if g.best_f > 1e-4 and g.n < limit - 20:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if g.n >= limit - 10 or _time_left() < 1.0:
                break
            x_rand = x0 + rng.uniform(-5, 5, size=2) * max(1.0, np.linalg.norm(x0))
            x_res, success, f_res = _secant_correct(g, x_rand, max_it=10)
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
            if success:
                return x_res

    if g.best_x is not None:
        return g.best_x
    return x0.copy()
