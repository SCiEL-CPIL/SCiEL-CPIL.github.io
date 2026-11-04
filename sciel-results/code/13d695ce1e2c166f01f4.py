"""Second autocorrelation inequality: maximize L2^2 / (L1 * sup|autoconv| for 50 nonneg heights."""
import numpy as np
import time

N = 50


def _autoconv_peak(h):
    """Compute sup of autoconvolution h*h (full convolution)."""
    # Use FFT-based convolution for speed
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * H, size)[:2 * n - 1]
    return float(np.max(ac))


def _ratio(h):
    """Compute L2^2 / (L1 * sup|autoconv|). Returns 0 if invalid."""
    h = np.asarray(h, dtype=float)
    L1 = float(np.sum(h))
    if L1 <= 0:
        return 0.0
    sup = _autoconv_peak(h)
    if sup <= 0:
        return 0.0
    L2sq = float(np.dot(h, h))
    return L2sq / (L1 * sup)


def _smooth_max(ac, beta=20.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(ac)
    return m + np.log(np.sum(np.exp(beta * (ac - m)))) / beta


def _objective_and_grad(params, beta=20.0):
    """
    Objective: minimize -log(ratio) = -log(L2^2) + log(L1) + log(sup_ac)
    We use a smooth surrogate for the max.
    
    params: real vector of length N (unconstrained; we use h = softplus(params) for nonneg)
    """
    # h = softplus(params) ensures non-negativity
    # softplus(x) = log(1 + exp(x))
    # To avoid overflow: softplus(x) = max(x,0) + log1p(exp(-|x|))
    x = params
    h = np.where(x > 20, x, np.log1p(np.exp(x)))  # stable softplus
    h = np.maximum(h, 1e-15)  # ensure strictly positive for log
    
    L1 = float(np.sum(h))
    L2sq = float(np.dot(h, h))
    
    # Autoconvolution
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * H, size)[:2 * n - 1]
    ac = np.maximum(ac, 1e-15)
    
    # Smooth max
    m = np.max(ac)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (ac - m)))) / beta
    
    # -log(ratio) = -log(L2^2) + log(L1) + log(sup)
    val = -2.0 * np.log(L2sq) + np.log(L1) + np.log(log_sum_exp)
    
    # Gradient with respect to params
    # dh/dparams = sigmoid(params) = exp(x)/(1+exp(x)) = 1/(1+exp(-x))
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    # dL1/dh = 1
    # dL2sq/dh = 2h
    # dac/dh: ac[k] = sum_i h[i]*h[i+k], so dac/dh[j] = sum_k (h[j-k] + h[j+k]) (with bounds)
    # Actually for full conv: ac[m] = sum_i h[i] h[m-i] for appropriate indices
    # Let's compute grad_ac/dh numerically-ish or analytically
    # dac/dh[j] = sum over all pairs (i,k) where i+k = m of h[k] if i=j, plus h[i] if k=j
    # For autoconv: ac[m] = sum_i h[i] h[m-i]
    # d(ac[m])/d(h[j]) = h[m-j] + h[j-m] ... but m-j and j-m need to be in [0,N-1]
    # Actually: d(ac[m])/d(h[j]) = h[m-j] if 0<=m-j<N, plus h[j-m] if 0<=j-m<N
    # But since ac is symmetric, and we're summing over all m...
    # Let's just use: grad_ac = convolve(h, h_reversed) style
    # dac/dh[j] = sum_m d(ac[m])/d(h[j]) = sum_m [h[m-j] (if valid) + h[j-m] (if valid)]
    # = sum_m h[m-j] + sum_m h[j-m] (with bounds)
    # = sum_i h[i] + sum_i h[i] = 2*L1 ... no that's not right
    # Let me think more carefully.
    # ac[m] = sum_i h[i] h[m-i] where 0<=i<N and 0<=m-i<N
    # d(ac[m])/d(h[j]) = h[m-j] (if 0<=m-j<N) + h[j-m] (if 0<=j-m<N)
    # Note h[j-m] = h[m'-j] where m'=2m... no.
    # Actually d(ac[m])/d(h[j]): 
    #   term where i=j: h[m-j] (valid if 0<=m-j<N)
    #   term where m-i=j i.e. i=m-j: h[j] * d(h[m-i])/d(h[j]) = h[j] * delta(m-i,j) = h[j] if m-j=j i.e. m=2j
    # Wait, I'm confusing myself. Let me just do:
    # ac[m] = sum_i h[i]*h[m-i]
    # d/dh[j] = sum_i [delta(i,j)*h[m-i] + h[i]*delta(m-i,j)]
    # = h[m-j] (if 0<=m-j<N) + sum_i h[i]*delta(m-i,j)
    # = h[m-j] (if 0<=m-j<N) + h[m-j] (if 0<=m-j<N)  [since i=m-j]
    # Wait no: sum_i h[i]*delta(m-i,j) = h[j] when m-i=j, i.e., i=m-j. So it's h[m-j] if 0<=m-j<N.
    # Hmm, so d(ac[m])/d(h[j]) = 2*h[m-j] if 0<=m-j<N, else 0? That doesn't seem right either.
    # Let me just compute it properly:
    # ac[m] = sum_i h[i] h[m-i]
    # d(ac[m])/d(h[j]) = sum_i [delta_{i,j} h[m-i] + h[i] delta_{m-i,j}]
    # First term: h[m-j] if 0<=m-j<N
    # Second term: sum_i h[i] delta_{m-i,j} = h[j] if m-j = j, i.e., m=2j? No.
    # delta_{m-i,j} is 1 when m-i=j, i.e., i=m-j. So second term = h[m-j] if 0<=m-j<N.
    # So d(ac[m])/d(h[j]) = 2*h[m-j] if 0<=m-j<N, else 0.
    # But wait, that's only when the same index j appears in both factors. Let me re-examine.
    # ac[m] = sum_i h[i] * h[m-i]
    # The two factors are h[i] and h[m-i]. 
    # d/dh[j] of h[i] is delta_{i,j}
    # d/dh[j] of h[m-i] is delta_{m-i,j} = delta_{i,m-j}
    # So d(ac[m])/d(h[j]) = sum_i [delta_{i,j} h[m-i] + h[i] delta_{i,m-j}]
    # = h[m-j] (if 0<=m-j<N) + h[m-j] (if 0<=m-j<N)
    # = 2 h[m-j] if 0<=m-j<N, else 0.
    # Hmm, but this seems to miss cases where j appears in only one factor in a different way.
    # Actually I think I made an error. Let's say N=3, m=1.
    # ac[1] = h[0]*h[1] + h[1]*h[0] = 2 h[0] h[1]
    # d/dh[0] = 2 h[1], d/dh[1] = 2 h[0], d/dh[2] = 0.
    # Using formula: d(ac[1])/d(h[0]) = 2*h[1-0] = 2*h[1]. Correct.
    # d(ac[1])/d(h[1]) = 2*h[1-1] = 2*h[0]. Correct.
    # d(ac[1])/d(h[2]) = 0 (since 1-2=-1 not in range). Correct.
    # OK so the formula works.
    
    # So d(ac[m])/d(h[j]) = 2*h[m-j] if 0<=m-j<N else 0
    # Therefore d(sum_m ac[m])/d(h[j]) = 2 * sum_m h[m-j] (over valid m) = 2 * sum_i h[i] = 2*L1
    # But we need d(log_sum_exp)/d(h[j]), not d(sum)/d(h[j]).
    # d(log_sum_exp)/d(ac[m]) = exp(beta*(ac[m]-m_max)) / sum_k exp(beta*(ac[k]-m_max))
    # Let w[m] = exp(beta*(ac[m]-m_max)) / sum_k exp(beta*(ac[k]-m_max))
    # d(log_sum_exp)/d(h[j]) = sum_m w[m] * d(ac[m])/d(h[j]) = sum_m w[m] * 2*h[m-j] (if valid)
    
    # This is getting complex. Let me just compute the gradient numerically for simplicity
    # given the time constraints, or use a simpler approach.
    
    # For now, let's just return the value and use a simple gradient-free method
    # or compute gradient numerically.
    
    # Actually, let's just return the value and use Nelder-Mead or Powell
    return val, None  # gradient not provided, will use gradient-free


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def _time_left():
        return DEADLINE - (time.time() - t0)
    
    best = np.ones(N)
    best_val = _ratio(best)
    
    def _try(h):
        nonlocal best, best_val
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best = h.copy()
    
    # Try various structured profiles
    # 1. Uniform
    _try(np.ones(N))
    
    # 2. Gaussian-like profiles
    for center in range(0, N, 5):
        for sigma in [2, 3, 5, 8, 12, 20]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            _try(h)
            # Try with different amplitudes (though ratio is scale-invariant)
    
    # 3. Multi-peak profiles
    for n_peaks in [2, 3, 4, 5]:
        for spread in [5, 10, 15]:
            centers = np.linspace(5, N - 5, n_peaks)
            h = np.zeros(N)
            for c in centers:
                h += np.exp(-0.5 * ((np.arange(N) - c) / spread) ** 2)
            _try(h)
    
    # 4. Sparse profiles (few non-zero entries)
    from itertools import combinations
    for sparsity in [5, 8, 12, 15]:
        # Try random sparse vectors
        for trial in range(20):
            if _time_left() < 5:
                break
            idx = np.random.default_rng(trial * 7 + sparsity).choice(N, sparsity, replace=False)
            h = np.zeros(N)
            h[idx] = np.random.default_rng(trial * 13 + sparsity).uniform(0.5, 2.0, sparsity)
            _try(h)
    
    # 5. Use scipy.optimize with different methods
    from scipy.optimize import minimize
    
    # Use a parametrization that ensures non-negativity: h = exp(params)
    # But this can have issues with scale. Let's use h = softplus(params)
    
    def _neg_log_ratio(params):
        x = params
        h = np.where(x > 20, x, np.log1p(np.exp(x)))
        h = np.maximum(h, 1e-15)
        L1 = float(np.sum(h))
        if L1 <= 0:
            return 1e10
        sup = _autoconv_peak(h)
        if sup <= 0:
            return 1e10
        L2sq = float(np.dot(h, h))
        if L2sq <= 0:
            return 1e10
        return -(2.0 * np.log(L2sq) - np.log(L1) - np.log(sup))
    
    # Try optimization from different starting points
    starts = [
        np.zeros(N),  # h = softplus(0) = log(2) ~ 0.69
        np.ones(N) * 0.5,
        np.ones(N) * -0.5,
        np.random.default_rng(42).normal(0, 1, N),
    ]
    
    for start in starts:
        if _time_left() < 30:
            break
        try:
            res = minimize(
                _neg_log_ratio, start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.x is not None:
                x = res.x
                h = np.where(x > 20, x, np.log1p(np.exp(x)))
                h = np.maximum(h, 0.0)
                _try(h)
        except Exception:
            pass
    
    # 6. Coordinate descent on the original variables
    if _time_left() > 20:
        h = best.copy()
        for iteration in range(500):
            if _time_left() < 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a range of values for h[i]
                for trial_val in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    h[i] = trial_val
                    v = _ratio(h)
                    if v > _ratio(np.concatenate([h[:i], [old], h[i+1:]])):
                        improved = True
                        break
                else:
                    h[i] = old
            if not improved:
                break
        _try(h)
    
    # 7. Final polish with L-BFGS-B on a smoothed objective
    if _time_left() > 15:
        def _smoothed_obj(params):
            x = params
            h = np.where(x > 20, x, np.log1p(np.exp(x)))
            h = np.maximum(h, 1e-15)
            L1 = float(np.sum(h))
            if L1 <= 0:
                return 1e10
            n = len(h)
            size = 1
            while size < 2 * n - 1:
                size *= 2
            H = np.fft.rfft(h, size)
            ac = np.fft.irfft(H * H, size)[:2 * n - 1]
            ac = np.maximum(ac, 1e-15)
            beta = 30.0
            m = np.max(ac)
            log_sum_exp = m + np.log(np.sum(np.exp(beta * (ac - m)))) / beta
            L2sq = float(np.dot(h, h))
            if L2sq <= 0:
                return 1e10
            return -(2.0 * np.log(L2sq) - np.log(L1) - np.log(log_sum_exp))
        
        # Start from best
        # Invert softplus: params = log(exp(h) - 1)
        h_best = best.copy()
        h_best = np.maximum(h_best, 1e-15)
        params_init = np.log(np.expm1(h_best))
        
        try:
            res = minimize(
                _smoothed_obj, params_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxls': 50}
            )
            if res.x is not None:
                x = res.x
                h = np.where(x > 20, x, np.log1p(np.exp(x)))
                h = np.maximum(h, 0.0)
                _try(h)
        except Exception:
            pass
    
    return [max(0.0, float(x)) for x in best]
