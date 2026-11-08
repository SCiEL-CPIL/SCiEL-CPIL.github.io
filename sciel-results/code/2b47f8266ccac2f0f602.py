"""
Optimization of the second autocorrelation inequality constant C2.
Strategy: Low-rank DCT basis representation with progressive expansion.
Instead of optimizing 50 raw heights, we optimize coefficients of cosine basis
functions. This enforces global smoothness, reduces dimensionality, and prevents
the formation of high-frequency local spikes that typically degrade the L2-to-L1
ratio. We use a two-stage approach: first optimize a low-frequency basis, then
expand to higher frequencies and refine.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 148.0     # safety margin before 150s kill
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_value(h, beta=100.0):
    """
    Smooth surrogate for the value function using log-sum-exp for norm_inf.
    Returns negative value for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _cosine_basis(n_basis):
    """
    Generate cosine basis vectors for the first n_basis modes.
    Basis 0 is constant, basis 1 is linear (to allow asymmetry), 
    bases 2..n_basis-1 are standard DCT-II cosines.
    """
    x = np.arange(N)
    basis = []
    # Mode 0: Constant
    basis.append(np.ones(N))
    # Mode 1: Linear ramp (allows asymmetry, unlike pure DCT)
    basis.append(x / (N - 1))
    # Modes 2..n_basis-1: DCT-II cosines
    for k in range(1, n_basis - 1):
        # DCT-II: cos(pi * (2*n + 1) * k / (2*N))
        vec = np.cos(np.pi * (2 * x + 1) * k / (2 * N))
        basis.append(vec)
    return np.array(basis).T  # Shape (N, n_basis)

def _coeffs_to_heights(coeffs, n_basis):
    """
    Convert basis coefficients to heights.
    We use a softplus-like activation to ensure non-negativity and smoothness.
    Actually, simpler: just linear combination, then clip to 0.
    But clipping can create non-smoothness. 
    Better: use exp or softplus on the linear combination?
    Let's stick to linear combination + clip for now, as the basis itself
    is smooth. The clip only affects if the sum goes negative.
    To avoid hard clips, we can optimize with a penalty for negative values
    or just use bounds on coeffs? No, coeffs can be negative.
    
    Alternative: h = basis @ coeffs, then h = np.maximum(0, h).
    This is standard. The gradient is 0 where h<0, 1 where h>0.
    """
    basis = _cosine_basis(n_basis)
    h = basis @ coeffs
    return np.maximum(0.0, h)

def _objective_from_coeffs(coeffs, n_basis, beta):
    """
    Objective function in terms of basis coefficients.
    """
    h = _coeffs_to_heights(coeffs, n_basis)
    return _smooth_value(h, beta)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
            
    # Initial seed
    track(best_h)
    
    # Define stages of basis expansion
    # Start with low frequency (smooth), expand to higher frequency (detailed)
    stages = [
        (4, 20.0),   # 4 basis functions, low beta (smooth max)
        (8, 50.0),   # 8 basis functions, medium beta
        (15, 100.0), # 15 basis functions, high beta
        (25, 200.0), # 25 basis functions, very high beta
        (35, 500.0), # 35 basis functions, near-exact max
    ]
    
    # We will keep the current best coefficients for each stage to initialize the next
    current_coeffs = None
    
    for n_basis, beta in stages:
        if time.time() > deadline - 30.0:
            break
            
        # If we don't have initial coeffs for this size, initialize from best_h
        if current_coeffs is None or len(current_coeffs) < n_basis:
            # Project best_h onto the new basis to get initial coeffs
            basis = _cosine_basis(n_basis)
            # Least squares projection
            try:
                current_coeffs = np.linalg.lstsq(basis, best_h, rcond=None)[0]
            except:
                current_coeffs = np.zeros(n_basis)
            # Pad if necessary (shouldn't happen if we build correctly)
            if len(current_coeffs) < n_basis:
                current_coeffs = np.pad(current_coeffs, (0, n_basis - len(current_coeffs)))
        else:
            # Truncate or pad to match n_basis
            if len(current_coeffs) > n_basis:
                current_coeffs = current_coeffs[:n_basis]
            elif len(current_coeffs) < n_basis:
                current_coeffs = np.pad(current_coeffs, (0, n_basis - len(current_coeffs)))
        
        # Local optimization using L-BFGS-B
        # We optimize the coefficients directly.
        # Note: The mapping coeffs -> h -> value is not convex.
        # But the low-dimensional space is easier to navigate.
        
        x0 = current_coeffs.copy()
        
        # Multiple restarts for each stage to avoid local minima
        n_restarts = 5 if n_basis < 10 else 3
        for r in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
                
            if r == 0:
                x_start = x0
            else:
                # Perturb coefficients
                x_start = x0 + RNG.normal(0, 0.1, n_basis)
            
            try:
                res = opt.minimize(
                    lambda c: _objective_from_coeffs(c, n_basis, beta),
                    x_start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 0:
                    h_res = _coeffs_to_heights(res.x, n_basis)
                    track(h_res)
                    # Update current_coeffs to the best one found so far in this stage
                    if compute_value(h_res) > compute_value(_coeffs_to_heights(x0, n_basis)):
                         # Actually, we should track the best coeffs separately
                         pass
            except Exception:
                pass
        
        # After optimization, update current_coeffs to the best one found
        # We need to re-evaluate or keep track. Let's just re-project best_h for next stage
        # This ensures we always start from the global best found so far
        basis = _cosine_basis(n_basis)
        try:
            current_coeffs = np.linalg.lstsq(basis, best_h, rcond=None)[0]
        except:
            current_coeffs = np.zeros(n_basis)

    # Final polish in full 50D space using L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 500.0),
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 0:
                track(res.x)
        except Exception:
            pass
            
    # Nelder-Mead polish
    if time.time() < deadline - 2.0:
        try:
            nm_res = opt.minimize(
                lambda x: _smooth_value(x, 500.0),
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_res.fun < 0:
                track(nm_res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
