"""Signed autocorrelation C3-prime: minimize max|autoconv| / (sum)^2.
Uses continuous L-BFGS-B optimization on a smooth max surrogate,
then snaps to signs and does efficient local search.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400


def _autoconv(x):
    """Full linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)
    return c[:2 * n - 1]


def _score(x):
    """C3' = max|autoconv| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))


def _smooth_max_abs_autoconv(x, beta=50.0):
    """Smooth approximation of max|autoconv(x)| using log-sum-exp."""
    c = _autoconv(x)
    ac_abs = np.abs(c)
    m = np.max(ac_abs)
    # Log-sum-exp for numerical stability
    shifted = beta * (ac_abs - m)
    log_sum = m + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    return log_sum


def _objective_and_grad(x, beta=50.0, sum_penalty=10.0):
    """Objective: smooth max|autoconv| + penalty*(max(0,1-|sum|))^2.
    Returns (value, gradient)."""
    c = _autoconv(x)
    ac_abs = np.abs(c)
    m = np.max(ac_abs)
    shifted = beta * (ac_abs - m)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Value
    smooth_max = m + (1.0 / beta) * np.log(sum_exp)
    s = x.sum()
    pen = sum_penalty * max(0.0, 1.0 - abs(s)) ** 2
    val = smooth_max + pen
    
    # Gradient of smooth max
    # d/dx_j [m + (1/beta)*log(sum_k exp(beta*(|c_k| - m)))]
    # = (1/beta) * sum_k exp(beta*(|c_k|-m)) * beta * sign(c_k) * d|c_k|/dx_j / sum_exp
    # = sum_k w_k * sign(c_k) * d|c_k|/dx_j
    # where w_k = exp(beta*(|c_k|-m)) / sum_exp
    
    w = exp_shifted / sum_exp  # weights
    
    # Gradient of |c_k| with respect to x_j
    # c_k = sum_i x_i * x_{i+k-N+1} (for appropriate indices)
    # Actually c[k] = sum_{i} x[i] * x[i+k] for k in [-(N-1), N-1]
    # For full linear autoconv: c[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k] for k>=0
    #                            c[k] = sum_{i=0}^{N-1+k} x[i]*x[i-k] for k<0 (by symmetry)
    
    # d|c_k|/dx_j = sign(c_k) * (x[j+k] + x[j-k]) when indices valid
    # More carefully: c_k = sum_i x[i] x[i+k]
    # dc_k/dx_j = x[j+k] (if 0<=j+k<N) + x[j-k] (if 0<=j-k<N, from the i=j-k term: x[j-k]*x[j])
    # Wait, let me be more careful.
    # c_k = sum_{i=max(0,-k)}^{min(N-1,N-1-k)} x[i] * x[i+k]
    # dc_k/dx_j: j appears as x[i] when i=j (needs 0<=j<N and 0<=j+k<N)
    #            j appears as x[i+k] when i+k=j, i=j-k (needs 0<=j-k<N and 0<=j<N)
    
    # So dc_k/dx_j = x[j+k]*1_{0<=j+k<N} + x[j-k]*1_{0<=j-k<N}
    
    # This is expensive to compute for all k,j. Instead, use the fact that
    # the gradient of the autoconv w.r.t. x is a convolution-like operation.
    
    # Alternative: compute gradient numerically? No, too slow.
    # Better: use the structure. The autoconv c = x * x (linear).
    # d c_k / d x_j = x_{j+k} + x_{j-k} (with appropriate boundary handling)
    
    # Let me compute the gradient via the FFT approach or direct summation.
    # For N=400, direct computation of grad for all j is O(N^2) which is 160k, fine.
    
    grad = np.zeros(N)
    for k in range(2 * N - 1):
        # c_k corresponds to lag k-N+1 in standard notation
        # Actually my _autoconv returns c[0..2N-2] where c[k] = sum_i x[i]*x[i+k-(N-1)]
        # Let me just use the property: for the full linear autoconv of length 2N-1,
        # c[k] = sum_{i} x[i] * x[i + k - (N-1)]
        # So the lag is k - (N-1)
        lag = k - (N - 1)
        if ac_abs[k] < 1e-15:
            continue
        sign_ck = 1.0 if c[k] > 0 else -1.0
        wk = w[k]
        if wk < 1e-10:
            continue
        # dc_k/dx_j = x[j+lag] if 0<=j+lag<N + x[j-lag] if 0<=j-lag<N
        if lag >= 0:
            # j+lag in [0,N-1] => j in [0, N-1-lag]
            # j-lag in [0,N-1] => j in [lag, N-1]
            # For j in [0, N-1-lag]: dc/dx_j has x[j+lag]
            idx1 = np.arange(0, N - lag)
            grad[idx1] += wk * sign_ck * x[idx1 + lag]
            # For j in [lag, N-1]: dc/dx_j has x[j-lag]
            idx2 = np.arange(lag, N)
            grad[idx2] += wk * sign_ck * x[idx2 - lag]
        else:
            lag_abs = -lag
            # j+lag = j-lag_abs in [0,N-1] => j in [lag_abs, N-1]
            idx1 = np.arange(lag_abs, N)
            grad[idx1] += wk * sign_ck * x[idx1 - lag_abs]
            # j-lag = j+lag_abs in [0,N-1] => j in [0, N-1-lag_abs]
            idx2 = np.arange(0, N - lag_abs)
            grad[idx2] += wk * sign_ck * x[idx2 + lag_abs]
    
    # Gradient of penalty
    if s > 0:
        if abs(s) < 1.0:
            grad += 2.0 * sum_penalty * (1.0 - s) * np.ones(N)
    else:
        if abs(s) < 1.0:
            grad += 2.0 * sum_penalty * (1.0 + s) * (-np.ones(N))
    
    return val, grad


def _continuous_optimize(x0, beta, deadline, maxiter=200):
    """Run L-BFGS-B from x0."""
    if time.time() > deadline:
        return x0, _score(x0)
    
    def obj(x):
        v, _ = _objective_and_grad(x, beta)
        return v
    
    def grad(x):
        _, g = _objective_and_grad(x, beta)
        return g
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': 500,
                                'ftol': 1e-10, 'gtol': 1e-8,
                                'iprint': -1})
        x_opt = res.x
    except Exception:
        x_opt = x0
    
    return x_opt, _score(x_opt)


def _snap_to_signs(x):
    """Snap continuous values to +/-1, preserving sign."""
    return np.sign(x).astype(np.float64)


def _local_search(x, deadline, max_steps=50000):
    """Sign-flip local search minimizing the actual score."""
    x = x.copy().astype(np.float64)
    best = _score(x)
    n = len(x)
    rng = np.random.default_rng(999)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        j = int(rng.integers(n))
        x[j] = -x[j]
        sc = _score(x)
        if sc < best:
            best = sc
        else:
            x[j] = -x[j]
    
    return x, best


def _structured_seeds(rng):
    """Generate diverse initial seeds."""
    seeds = []
    seeds.append(np.ones(N, dtype=np.float64))
    seeds.append(np.tile([1.0, -1.0], N // 2).astype(np.float64))
    seeds.append(np.tile([1.0, 1.0, -1.0, -1.0], N // 4).astype(np.float64))
    for _ in range(8):
        seeds.append(rng.choice([-1.0, 1.0], size=N).astype(np.float64))
    for bs in [4, 8, 16, 32]:
        if N % bs == 0:
            nb = N // bs
            seeds.append(np.repeat(rng.choice([-1.0, 1.0], size=nb), bs).astype(np.float64))
    return seeds


def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    seeds = _structured_seeds(rng)
    betas = [20.0, 50.0, 100.0]
    
    for s in seeds:
        if time.time() > deadline - 15:
            break
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            # Continuous optimization
            x_cont, sc_cont = _continuous_optimize(s, beta, deadline - 10, maxiter=100)
            
            # Snap to signs
            x_s = _snap_to_signs(x_cont)
            sc_s = _score(x_s)
            
            if sc_s < best_score:
                best_score = sc_s
                best_x = x_s.copy()
            
            # Local search polish
            x_ls, sc_ls = _local_search(x_s, deadline - 5, max_steps=20000)
            if sc_ls < best_score:
                best_score = sc_ls
                best_x = x_ls.copy()
    
    # Final refinement
    if time.time() < deadline - 5:
        x_final, sc_final = _local_search(best_x, deadline - 2, max_steps=100000)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final
    
    return best_x
