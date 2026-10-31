"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max of h[i]*(1-h[i+1]) with zero-padding at the end."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps.max()

def _compute_all_overlaps(h):
    """Compute all adjacent overlaps as a vector."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps

def _smooth_max(values, alpha):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _objective(h, alpha):
    """Smooth surrogate objective: smooth max of adjacent overlaps."""
    overlaps = _compute_all_overlaps(h)
    return _smooth_max(overlaps, alpha)

def _objective_grad(h, alpha):
    """Gradient of the smooth max objective."""
    overlaps = _compute_all_overlaps(h)
    m = np.max(overlaps)
    shifted = alpha * (overlaps - m)
    shifted = np.clip(shifted, -50, 50)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros(N)
    for i in range(N - 1):
        grad[i] += weights[i] * (1.0 - h[i+1])
        grad[i+1] += weights[i] * (-h[i])
    grad[N-1] += weights[N-1] * 1.0
    
    return grad

def _project_to_polytope(h, S):
    """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
    if S < 0 or S > N:
        return np.clip(h, 0, 1)
        
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h - mid, 0.0, 1.0))
        if val > S:
            lo = mid
        else:
            hi = mid
                
    lambda_star = (lo + hi) / 2.0
    return np.clip(h - lambda_star, 0.0, 1.0)

def _enforce_constraints(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    return _project_to_polytope(h, N / 2.0)

def _optimize_from_start(h0, alpha, max_iter=200):
    """Optimize from a starting point using SLSQP."""
    n = len(h0)
    target = n / 2.0
    
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target, 
                    'jac': lambda x: np.ones(n)}]
    
    def obj(x):
        return _objective(x, alpha)
    
    def obj_grad(x):
        return _objective_grad(x, alpha)
    
    try:
        res = minimize(obj, h0, jac=obj_grad, method='SLSQP', bounds=bounds, 
                      constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.success or (hasattr(res, 'fun') and res.fun < _objective(h0, alpha)):
            return res.x
    except Exception:
        pass
    return h0

def _minimize_coord(h, i, deadline=None):
    """
    Minimize the max overlap with respect to h[i] only, keeping other h fixed.
    The objective is max( C1, C2, C3 ) where:
    C1 = h[i-1]*(1-h[i])  (if i>0)
    C2 = h[i]*(1-h[i+1])  (if i<N-1)
    C3 = h[i]             (if i==N-1)
    C0 = other overlaps (constant)
    
    We want to find h[i] in [0,1] minimizing max(C0, C1, C2, C3).
    """
    if deadline is not None and time.time() > deadline:
        return h[i]
        
    # Calculate constant part (max of overlaps not involving i)
    # Overlaps involving i:
    # idx i-1: h[i-1]*(1-h[i])
    # idx i:   h[i]*(1-h[i+1])
    # idx N-1: h[N-1] if i==N-1
    
    # Let's just evaluate the full max for a few candidates and use ternary search on the convex part?
    # The function f(x) = max(const, a*(1-x), x*b, x) is convex.
    # We can find the minimum by checking the intersection points or boundaries.
    
    a = h[i-1] if i > 0 else 0.0
    b = 1.0 - h[i+1] if i < N-1 else 0.0 # Coefficient for x in C2 is (1-h[i+1])
    c = 1.0 if i == N-1 else 0.0           # Coefficient for x in C3 is 1
    
    # f(x) = max( K, a(1-x), x*b, x*c )
    # where K is max of other overlaps.
    
    # To avoid recalculating K every time, we can just do a grid search + refinement.
    # Given N=100, a simple ternary search on the convex function is fast.
    
    def val(x):
        h[i] = x
        return _overlap_max(h)
    
    # Ternary search on [0, 1]
    lo = 0.0
    hi = 1.0
    orig = h[i]
    
    # Check if orig is better than endpoints first
    best_x = orig
    best_val = val(orig)
    
    if val(0.0) < best_val:
        best_val = val(0.0)
        best_x = 0.0
    if val(1.0) < best_val:
        best_val = val(1.0)
        best_x = 1.0
        
    # Ternary search
    for _ in range(50):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        v1 = val(m1)
        v2 = val(m2)
        if v1 < v2:
            hi = m2
        else:
            lo = m1
        if abs(hi - lo) < 1e-9:
            break
            
    mid = (lo + hi) / 2.0
    v_mid = val(mid)
    if v_mid < best_val:
        best_val = v_mid
        best_x = mid
        
    h[i] = best_x
    return best_x

def _coordinate_descent(h, max_iters=500, tol=1e-12, deadline=None):
    """
    Coordinate descent local search: minimize max overlap by adjusting individual h_i.
    """
    h = h.copy()
    target = N / 2.0
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            orig = h[i]
            best_candidate = _minimize_coord(h, i, deadline)
            if best_candidate != orig:
                improved = True
                # Re-enforce constraints after each improvement
                h = _enforce_constraints(h)
                
        if not improved:
            break
    
    return h

def _try_ramp_pattern(deadline):
    """Try a ramp/plateau pattern: values rise from 0 to 1 and fall back."""
    best_h = None
    best_score = float('inf')
    
    # We want a block of 1s, a ramp down, and a block of 0s.
    # Let the ramp be in the middle.
    # Length of ramp K.
    for K in range(5, 50, 2):
        # Try different positions?
        # Since sum is N/2, we have N/2 ones and N/2 zeros roughly.
        # If we have a ramp of length K, the number of 1s and 0s must adjust.
        
        # Construct a base profile:
        # 1s ... 1, ramp, 0s ... 0
        # Let n_ones be the number of 1s.
        # Let n_zeros be the number of 0s.
        # n_ones + n_zeros + K = N.
        # Sum = n_ones * 1 + n_zeros * 0 + sum(ramp) = N/2.
        # If ramp is symmetric linear, sum(ramp) = K/2.
        # n_ones + K/2 = N/2 => n_ones = (N-K)/2.
        # n_zeros = N - K - n_ones = N - K - (N-K)/2 = (N-K)/2.
        
        if (N - K) % 2 != 0:
            continue
            
        n_ones = (N - K) // 2
        n_zeros = (N - K) // 2
        
        h = np.zeros(N)
        h[:n_ones] = 1.0
        ramp = np.linspace(1.0, 0.0, K)
        h[n_ones:n_ones+K] = ramp
        h[n_ones+K:] = 0.0
        
        h = _enforce_constraints(h)
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        if time.time() > deadline - 20:
            break
            
    return best_h, best_score

def construct():
    deadline = time.time() + 135.0  # Safety margin
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try structured patterns first
    h1, s1 = _try_ramp_pattern(deadline)
    if h1 is not None and s1 < best_score:
        best_h = h1
        best_score = s1
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    starts = []
    starts.append(best_h.copy())
    starts.append(np.full(N, 0.5))
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(_enforce_constraints(alt))
    for freq in [1, 2, 3, 5, 8]:
        x = np.arange(N)
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * x / N)
        starts.append(_enforce_constraints(h))
    for i in range(6):
        h = rng.uniform(0, 1, N)
        starts.append(_enforce_constraints(h))
    
    # Annealing with increasing alpha
    alphas = [20.0, 100.0, 500.0, 2000.0, 10000.0]
    
    for start_idx, h0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        h = h0.copy()
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            
            h = _optimize_from_start(h, alpha, max_iter=100)
            h = _enforce_constraints(h)
            
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Coordinate descent polish
        if time.time() < deadline - 5:
            h = _coordinate_descent(h, max_iters=200, deadline=deadline - 5)
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Final high-alpha polish
        if time.time() < deadline - 5:
            h = _optimize_from_start(h, 20000.0, max_iter=200)
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Final global polish on best solution
    if time.time() < deadline - 2:
        h = _optimize_from_start(best_h, 50000.0, max_iter=300)
        h = _enforce_constraints(h)
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final coordinate descent polish
    if time.time() < deadline - 1:
        h = _coordinate_descent(best_h, max_iters=500, deadline=deadline - 1)
        h = _enforce_constraints(h)
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Ensure exact sum
    best_h = _enforce_constraints(best_h)
    
    return best_h
