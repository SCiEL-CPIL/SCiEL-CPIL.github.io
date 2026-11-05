"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _logsumexp_max(x, axis=None):
    """Compute log-sum-exp over x, handling large values."""
    if axis is None:
        m = np.max(x)
        if m == -np.inf:
            return -np.inf
        return m + np.log(np.sum(np.exp(x - m)))
    else:
        m = np.max(x, axis=axis, keepdims=True)
        m_safe = np.where(np.isneginf(m), 0.0, m)
        e = np.exp(x - m_safe)
        s = np.sum(e, axis=axis, keepdims=True)
        log_s = np.log(s)
        result = m_safe + log_s
        result = np.where(np.isneginf(m), -np.inf, result)
        return np.squeeze(result, axis=axis)


def _evaluate(h):
    """Compute the objective value."""
    h = np.asarray(h, dtype=np.float64)
    S = np.sum(h)
    if S == 0 or not np.isfinite(S):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    return abs(2.0 * N * peak / (S * S))


def _smooth_value(h, beta):
    """Smooth surrogate for the objective using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    S = np.sum(h)
    if S == 0 or not np.isfinite(S):
        return 1e30
    conv = np.convolve(h, h, mode='full')
    # We want to maximize the negative of the objective, or minimize it
    # Objective = 2*N*max(conv) / S^2
    # Smooth max: (1/beta) * log(sum(exp(beta * conv)))
    # But we need to handle the sign - actually we just want to minimize 2*N*max(conv)/S^2
    # Let's compute smooth_max = (1/beta)*logsumexp(beta*conv)
    beta_conv = beta * conv
    smooth_max = _logsumexp_max(beta_conv) / beta
    return 2.0 * N * smooth_max / (S * S)


def _smooth_value_grad(h, beta):
    """Compute smooth objective and gradient."""
    h = np.asarray(h, dtype=np.float64)
    S = np.sum(h)
    if S == 0 or not np.isfinite(S):
        return 1e30, np.zeros_like(h)
    conv = np.convolve(h, h, mode='full')
    beta_conv = beta * conv
    m = np.max(beta_conv)
    if m == -np.inf:
        smooth_max = -np.inf
        weights = np.zeros_like(conv)
    else:
        e = np.exp(beta_conv - m)
        s = np.sum(e)
        smooth_max = (m + np.log(s)) / beta
        weights = e / s  # softmax weights over conv entries
    
    # Gradient of smooth_max w.r.t. h
    # smooth_max = (1/beta) * logsumexp(beta * conv)
    # d(smooth_max)/d(conv_k) = weights_k
    # conv_k = sum_i h_i * h_{i+k-N+1} (for lag k)
    # This is a bit complex. Let's use the chain rule carefully.
    
    # Actually, let's just compute gradient numerically or use a simpler approach.
    # For now, let's compute the gradient of the smooth objective analytically.
    
    # The smooth objective is f(h) = 2*N*smooth_max(h) / S^2
    # df/dh_i = (2*N/S^2) * d(smooth_max)/dh_i - (4*N*smooth_max/S^3) * S_i
    # where S_i = dS/dh_i = 1
    
    # d(smooth_max)/dh_i = sum_k weights_k * d(conv_k)/dh_i
    # conv_k = sum_j h_j * h_{j+k-N+1} (with appropriate bounds)
    # This is equivalent to the cross-correlation.
    
    # Let's use a different approach: since conv = fft-based or direct,
    # and we need d(conv)/dh, which is essentially 2*h convolved with appropriate shifts.
    
    # For efficiency, let's use the fact that:
    # d(conv_k)/dh_i = sum over terms where h_i appears
    # conv_k = sum_j h_{j} h_{j+k-N+1}
    # h_i appears when j=i or j+k-N+1=i
    # So d(conv_k)/dh_i = h_{i+k-N+1} + h_{i-k+N-1} (with boundary handling)
    
    # This is getting complex. Let's use a simpler optimization approach.
    
    return 2.0 * N * smooth_max / (S * S), np.zeros_like(h)


def construct():
    """Construct step heights to minimize the autocorrelation upper bound."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Start with the seed
    h = np.ones(N)
    best_h = h.copy()
    best_value = _evaluate(h)
    
    # Strategy: Use coordinate descent with smooth surrogate
    # The key insight is that we can use a smooth approximation of max
    
    # Let's try a different approach: use the fact that we want to minimize
    # 2*N*max(conv)/S^2
    
    # One approach: normalize h so that S=1, then minimize max(conv)
    # Another: use iterative methods
    
    # Let's try a simple approach first: perturb and evaluate
    # But we need something more systematic.
    
    # Key insight: By Cauchy-Schwarz, sum(h^2)/S^2 >= 1/N, so value >= 2
    # But wait - the max of conv is NOT necessarily sum(h^2)!
    # conv at lag 0 is sum(h_i^2)
    # But other lags could have larger values if h has structure
    
    # Actually, for the full convolution, lag 0 is sum(h_i^2) which is the largest
    # by Cauchy-Schwarz: |sum h_i h_{i+k}| <= sqrt(sum h_i^2 * sum h_{i+k}^2) = sum h_i^2
    # So lag 0 is always the max (for non-negative h, or in general the max signed)
    
    # Wait - "signed entries, with no absolute value inside np.max"
    # So np.max(conv) takes the maximum signed value
    # If h has negative values, some conv entries could be negative
    # But lag 0 = sum(h_i^2) >= 0
    # And by Cauchy-Schwarz, |conv_k| <= sum(h_i^2) for all k
    # So conv_0 = sum(h_i^2) is always the maximum signed entry
    
    # Therefore value = 2*N*sum(h^2)/S^2 >= 2*N*(S^2/N)/S^2 = 2
    # Equality when all h_i are equal
    
    # So the minimum is 2, achieved by constant h. The seed already achieves this.
    
    # But wait - let me re-read the problem. The value is:
    # abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    # np.convolve(h, h) with mode='full' has length 2*N-1
    # Index 0 corresponds to lag -(N-1), index N-1 corresponds to lag 0
    
    # Let me verify: np.convolve([1,2],[1,2]) = [1, 4, 4]
    # Lag -1: 1*2 = 2? No...
    # Actually np.convolve uses: (f*g)[k] = sum_n f[n]g[k-n]
    # For [1,2] and [1,2]:
    # k=0: f[0]g[0] = 1
    # k=1: f[0]g[1]+f[1]g[0] = 2+2 = 4
    # k=2: f[1]g[1] = 4
    # So index N-1=1 corresponds to lag 0, which is sum(h_i^2) = 1+4 = 5? No, that's 4.
    
    # Hmm, let me re-check. np.convolve([1,2],[1,2]) should give [1,4,4]
    # But sum(h^2) = 1+4 = 5, not 4.
    
    # Actually, the autocorrelation at lag 0 should be sum(h_i^2).
    # Let me check numpy docs more carefully.
    
    # np.convolve(a, v) = sum_k a[k] * v[i-k]
    # For a=v=[1,2]:
    # i=0: a[0]*v[0] = 1
    # i=1: a[0]*v[1] + a[1]*v[0] = 2+2 = 4  
    # i=2: a[1]*v[1] = 4
    # So it's [1, 4, 4], and the middle value (i=1) is 4, but sum(h^2)=5.
    
    # This doesn't match. Let me think again.
    
    # Oh wait, I think I'm confusing convolution with correlation.
    # Autocorrelation would be np.correlate(h, h, 'full')
    # But the problem says np.convolve(h, h)
    
    # For h=[1,2], np.convolve(h,h) = [1,4,4]
    # np.max = 4
    # S = 3
    # value = abs(2*2*4/9) = 16/9 ≈ 1.78
    
    # But sum(h^2) = 5, so if it were autocorrelation, max would be 5
    # and value = 2*2*5/9 = 20/9 ≈ 2.22
    
    # So np.convolve is NOT autocorrelation - it's convolution!
    # This changes everything. The max of the convolution is NOT necessarily sum(h^2).
    
    # Let me reconsider. For h = ones(N), conv = [1, 2, 3, ..., N, N-1, ..., 1]
    # max = N
    # S = N
    # value = 2*N*N/N^2 = 2. OK that matches.
    
    # For general h, the convolution max can be different.
    # We need to minimize 2*N*max(conv(h,h))/S^2
    
    # This is a different optimization problem!
    # Let's try to find h that minimizes this.
    
    # Key idea: We want the convolution to be as flat as possible, or have a low peak.
    # One approach: use h with both positive and negative values to create cancellations.
    
    # Let's try a numerical optimization approach.
    
    # Use scipy.optimize if available, or implement a simple method.
    
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy:
        # Normalize: let S = sum(h) = 1 (we can scale later)
        # Minimize max(conv(h,h)) subject to sum(h) = 1
        
        # Use a smooth surrogate for max
        def objective(params):
            if time.time() > deadline - 10:
                return best_value * 1e10
            h_local = params.copy()
            S = np.sum(h_local)
            if abs(S) < 1e-12:
                return 1e10
            conv = np.convolve(h_local, h_local, mode='full')
            # Smooth max using log-sum-exp
            beta = 50.0
            m = np.max(conv)
            if m < -1e10:
                return 1e10
            smooth_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return 2.0 * N * smooth_max / (S * S)
        
        # Start from seed
        h0 = np.ones(N) / N  # normalize so sum = 1
        
        # Try a few different starting points and strategies
        results = []
        
        # Strategy 1: From uniform
        try:
            res = minimize(objective, h0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12})
            if time.time() < deadline - 10:
                results.append((objective(res.x), res.x))
        except Exception:
            pass
        
        # Strategy 2: From a structured initial guess
        # Try alternating signs
        h_alt = np.ones(N)
        h_alt[::2] = -0.5
        h_alt /= np.sum(h_alt)
        if abs(np.sum(h_alt)) > 1e-12:
            try:
                res = minimize(objective, h_alt, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12})
                if time.time() < deadline - 10:
                    results.append((objective(res.x), res.x))
            except Exception:
                pass
        
        # Strategy 3: Random restarts
        np.random.seed(42)
        for trial in range(10):
            if time.time() > deadline - 15:
                break
            h_rand = np.random.randn(N)
            S = np.sum(h_rand)
            if abs(S) < 1e-12:
                continue
            h_rand = h_rand / S
            
            try:
                res = minimize(objective, h_rand, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12})
                if time.time() < deadline - 10:
                    results.append((objective(res.x), res.x))
            except Exception:
                pass
        
        # Pick the best result
        if results:
            results.sort(key=lambda x: x[0])
            best_val, best_h_local = results[0]
            # Verify with exact evaluation
            exact_val = _evaluate(best_h_local)
            if exact_val < best_value:
                best_h = best_h_local.copy()
                best_value = exact_val
        
        # Also try coordinate descent from the best found
        if time.time() < deadline - 20:
            h_cd = best_h.copy()
            # Coordinate descent
            for it in range(2000):
                if time.time() > deadline - 15:
                    break
                improved = False
                for i in range(0, N, 2):  # step by 2 for speed
                    if time.time() > deadline - 15:
                        break
                    orig = h_cd[i]
                    # Try a few perturbations
                    for delta in [0.1, -0.1, 0.01, -0.01]:
                        h_cd[i] = orig + delta
                        val = _evaluate(h_cd)
                        if val < best_value - 1e-10:
                            best_h = h_cd.copy()
                            best_value = val
                            improved = True
                        else:
                            h_cd[i] = orig
                    if improved:
                        break
                if not improved:
                    break
    
    # Final cleanup: ensure finite values and nonzero sum
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    S = np.sum(best_h)
    if S == 0:
        best_h = np.ones(N)
    elif S < 0:
        best_h = -best_h
    
    return best_h
# EVOLVE-BLOCK-END
