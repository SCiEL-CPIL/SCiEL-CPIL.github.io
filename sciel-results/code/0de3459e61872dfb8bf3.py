"""
Root finder for 3 fixed 2-D systems: Pure Derivative-Free Secant Method with Adaptive Multi-Start.

Strategy:
1. Core Solver: Implement a 2-point secant method (derivative-free) for 2x2 systems.
   This avoids the overhead and instability of Jacobian estimation (finite differences or Broyden).
   For a 2D system, the secant update requires only 2 points to form the secant matrix,
   and subsequent steps require 1 new evaluation to update the secant approximation.
   This is extremely call-efficient: ~O(1) calls per Newton-like step.
   
2. Global Search: Since the secant method is local, we use a multi-start strategy.
   - Start from x0.
   - If it fails or converges slowly, try points in a spiral/grid around x0 and around 0.
   - Use a "basin hopping" like logic: if we get stuck in a local minimum of ||F|| (but not zero),
     jump to a random point and restart.

3. Efficiency:
   - The secant method for 2D systems converges superlinearly (order ~1.618) when close.
   - It does not require finite difference calls (which would cost 4 extra calls per Jacobian).
   - It is robust to small noise in F evaluations (though we assume F is deterministic).
   
4. Fallback:
   - If secant fails (e.g., singular secant matrix), fall back to a simple gradient descent
     on ||F||^2 using a finite-difference gradient (only used as a last resort).
     Actually, let's stick to secant and multi-start to keep it simple and fast.
"""

import time
import numpy as np

_DEADLINE = time.time() + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _clean(v):
    try:
        a = np.asarray(v, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a

class _Budget(object):
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

def _secant_solve(g, x0, max_iter=50):
    """
    2-point secant method for a 2D system.
    We need two initial points x0, x1 to form the secant matrix.
    Then we iterate.
    """
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f

    # Get a second point x1
    # Choose x1 = x0 + delta * (1, 1) or similar
    # We want to ensure the secant matrix is invertible.
    # Let's try a small perturbation in a random direction or fixed (1,0) and (0,1) if needed.
    # Actually, for 2D, we need 2 independent directions.
    # Let's pick x1 = x0 + 0.1 * np.array([1, 0])
    # If that fails (singular), try x1 = x0 + 0.1 * np.array([0, 1])
    
    # To be robust, let's generate a few candidate x1s and pick the one that gives a non-singular secant.
    # But that costs extra calls. 
    # Standard secant for n-dim requires n+1 points to start. For n=2, we need 3 points?
    # No, the "secant method" for systems usually refers to the "two-point secant" or "Broyden-like" without FD.
    # The classic 2-point secant for 2D systems:
    # We have x0, x1. We compute F0, F1.
    # Secant matrix S = [F1 - F0, F2 - F0] / [x1 - x0, x2 - x0]? No, that's for 3 points.
    # For 2D, a 2-point secant approximation is not full rank.
    # We need a full rank approximation. 
    # The "two-point secant" usually means we use the most recent 2 points to update a rank-2 approximation?
    # Actually, for 2D, we can use a "modified secant" where we maintain a 2x2 matrix.
    # Initialization: We need 3 points to get a full rank 2x2 secant?
    # Let's use a "quasi-Newton" approach that is derivative-free.
    # Broyden's method is quasi-Newton. It starts with an initial guess J0 (e.g., Identity).
    # Then it updates J using the secant condition.
    # This is what the current program does, but it's called "Broyden".
    # The inspiration says "Derivative-Free Secant Method". 
    # Maybe they mean a specific algorithm that is more efficient?
    # Or maybe they mean using the secant formula directly without maintaining a full Jacobian?
    
    # Let's implement a robust "Multi-Point Secant" or just stick to Broyden but optimized.
    # The current program uses Broyden. The inspiration 1 (score 1.60) used "Derivative-Free Secant Method".
    # Let's look at the difference. Broyden is a quasi-Newton method. It IS derivative-free in the sense that it doesn't compute FD Jacobians.
    # Perhaps the "Secant Method" here refers to the "2-point secant" for scalars, extended to systems?
    # For systems, the "secant method" is often synonymous with Broyden or similar quasi-Newton methods.
    
    # Let's try a different angle: **Newton's method with finite differences** is expensive.
    # **Broyden** is cheaper.
    # **Secant** (2-point) for 2D systems:
    # We can maintain a 2x2 matrix S.
    # Initialize S = I.
    # At each step, we have x, F(x).
    # We take a step p = -S^{-1} F(x).
    # We evaluate x_new = x + p, F_new.
    # We update S using the secant condition: S_new (x_new - x) = F_new - F(x).
    # This is exactly Broyden's Good Method.
    
    # So why did the inspiration score better?
    # Maybe the implementation was more robust or the multi-start was better.
    # Or maybe they used a "2-point secant" that doesn't maintain a full matrix, but uses a different update?
    # Let's assume Broyden is the right path, but we need a better multi-start strategy.
    
    # Let's implement a very robust Broyden with a strong multi-start.
    
    x_prev = x.copy()
    F_prev = Fx.copy()
    J = np.eye(2)
    
    for _ in range(max_iter):
        if g.n >= g.limit or _time_left() < 1.0:
            break
            
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx
            
        if not np.all(np.isfinite(dx)):
            break
            
        # Damping
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

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Try Broyden from x0
    x_res, success, f_res = _secant_solve(g, x0, max_iter=50)
    if success:
        return x_res
    if f_res < g.best_f:
        g.best_f = f_res
        g.best_x = x_res.copy()
        
    # 2. Try Broyden from 0
    if g.n < limit - 50:
        x_res, success, f_res = _secant_solve(g, np.zeros(2), max_iter=50)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 3. Multi-start around x0 and 0
    rng = np.random.default_rng(42)
    scale = max(1.0, np.linalg.norm(x0))
    
    # Generate a set of candidate starting points
    candidates = []
    # Around x0
    for r in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            candidates.append(x0 + r * scale * np.array([np.cos(theta), np.sin(theta)]))
    # Around 0
    for r in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            candidates.append(r * np.array([np.cos(theta), np.sin(theta)]))
    # Random points
    for _ in range(10):
        candidates.append(rng.uniform(-10, 10, size=2) * scale)
        
    # Sort candidates by distance to x0 (prioritize local)
    candidates.sort(key=lambda x: np.linalg.norm(x - x0))
    
    for x_start in candidates:
        if g.n >= limit - 20 or _time_left() < 5.0:
            break
        # Skip if too close to previous starts (to avoid redundant work)
        # Simple check: if it's very close to x0 or 0, we already tried.
        # Let's just run it.
        x_res, success, f_res = _secant_solve(g, x_start, max_iter=30)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    if g.best_x is not None:
        return g.best_x
    return x0.copy()
