"""Second autocorrelation inequality - sparse support + amplitude optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 135.0  # safety margin before 150s kill

def _autoconv_peak(h):
    """Compute max of autoconvolution (circular not needed, linear)."""
    n = len(h)
    # autoconv[k] = sum_i h[i]*h[i+k] for k=0..n-1
    # Use FFT for speed
    fft_h = np.fft.rfft(h, n=2*n)
    autoconv = np.fft.irfft(fft_h * np.conj(fft_h), n=2*n)[:n]
    return autoconv.max()

def _score(h):
    """Compute the ratio (L2^2)/(L1 * ||autoconv||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.dot(h, h)
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _sidon_like_support(n_total, k):
    """Generate a Sidon-like set of k positions from [0, n_total).
    A Sidon set has all pairwise differences distinct (or as few collisions as possible)."""
    # Greedy: pick positions maximizing minimum difference
    support = []
    for _ in range(k):
        best_pos = 0
        best_min_diff = -1
        for pos in range(n_total):
            if pos in support:
                continue
            # Compute min difference to existing support
            if support:
                min_diff = min(abs(pos - s) for s in support)
            else:
                min_diff = n_total
            # Prefer positions that are far from existing ones
            if min_diff > best_min_diff:
                best_min_diff = min_diff
                best_pos = pos
        support.append(best_pos)
    return sorted(support)

def _optimize_amplitudes(support, time_budget):
    """Given a fixed support, optimize amplitudes to maximize the ratio.
    Use iterative method: for fixed support, the problem is
    maximize (sum a_i^2) / (max_k sum_i a_i a_{i+k})
    subject to sum a_i = 1, a_i >= 0.
    
    We can use a power-iteration-like approach or scipy minimize.
    """
    k = len(support)
    # Map support indices to 0..k-1
    # Build the autoconvolution structure: for each lag, which pairs contribute
    # autoconv[lag] = sum over i where both i and i+lag are in support of a_i * a_{i+lag}
    
    # Precompute: for each lag L (0..N-1), list of (i, j) pairs where j-i=L and both in support
    sup_set = set(support)
    pair_by_lag = {}
    for i_idx, i_pos in enumerate(support):
        for j_idx, j_pos in enumerate(support):
            lag = j_pos - i_pos
            if lag >= 0:
                if lag not in pair_by_lag:
                    pair_by_lag[lag] = []
                pair_by_lag[lag].append((i_idx, j_idx))
    
    # For the peak, we need max over lags of the sum
    # This is a non-smooth optimization. Use scipy with smoothing or penalty.
    
    # Simpler approach: parameterize a_i > 0, sum=1, use log parameters
    # Maximize log(sum a_i^2) - log(max_lag sum_{(i,j) in pairs[lag]} a_i a_j)
    
    def neg_log_ratio(log_a):
        a = np.exp(log_a)
        a = a / a.sum()  # normalize
        l2_sq = np.dot(a, a)
        if l2_sq < 1e-30:
            return 0.0
        # Compute autoconv peak
        peak = 0.0
        for lag, pairs in pair_by_lag.items():
            s = 0.0
            for i_idx, j_idx in pairs:
                s += a[i_idx] * a[j_idx]
            if s > peak:
                peak = s
        if peak < 1e-30:
            return 0.0
        return -np.log(l2_sq) + np.log(peak)
    
    # Start from uniform
    log_a0 = np.zeros(k)
    
    best_val = -np.inf
    best_a = None
    
    t0 = time.time()
    # Try a few random restarts
    n_restarts = min(5, max(1, int((time_budget - (time.time()-t0)) / 2.0)))
    
    for r in range(n_restarts):
        if r == 0:
            x0 = np.zeros(k)
        else:
            x0 = np.random.randn(k) * 0.5
        x0 = x0 - x0.mean()  # center
        
        res = minimize(neg_log_ratio, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        a = np.exp(res.x)
        a = a / a.sum()
        val = _score(np.zeros(N))  # placeholder
        # Compute actual score
        h = np.zeros(N)
        for i_idx, pos in enumerate(support):
            h[pos] = a[i_idx]
        val = _score(h)
        
        if val > best_val:
            best_val = val
            best_a = a.copy()
        
        if time.time() - t0 > time_budget * 0.7:
            break
    
    if best_a is None:
        best_a = np.ones(k) / k
    
    return best_a, best_val

def construct():
    t_start = time.time()
    time_budget = _DEADLINE - t_start - 5.0  # 5s safety margin
    if time_budget < 1.0:
        return [1.0] * N
    
    best_h = np.ones(N, dtype=float)
    best_score = _score(best_h)
    
    # Try different sparsity levels
    # For N=50, try k from 3 to 15 (sparse supports)
    # Also try the full support with optimized amplitudes
    
    k_values = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    
    for k in k_values:
        if time.time() - t_start > _DEADLINE - 10.0:
            break
        
        # Generate a few different support patterns
        n_supports = 3
        for s_idx in range(n_supports):
            if time.time() - t_start > _DEADLINE - 10.0:
                break
            
            if s_idx == 0:
                # Sidon-like
                support = _sidon_like_support(N, k)
            elif s_idx == 1:
                # Arithmetic progression (minimizes distinct lags but max collision)
                # Actually for minimizing peak, we want spread out
                step = N // k
                support = [(i * step + step // 2) % N for i in range(k)]
                support = sorted(set(support))
                if len(support) < k:
                    # Fill in
                    remaining = [p for p in range(N) if p not in support]
                    for p in remaining:
                        if len(support) >= k:
                            break
                        support.append(p)
                    support = sorted(support[:k])
            else:
                # Random
                support = sorted(np.random.choice(N, k, replace=False).tolist())
            
            if len(support) < 2:
                continue
            
            # Optimize amplitudes
            sub_budget = max(1.0, time_budget * 0.3)
            a_opt, val = _optimize_amplitudes(support, sub_budget)
            
            h = np.zeros(N)
            for i_idx, pos in enumerate(support):
                h[pos] = a_opt[i_idx]
            
            sc = _score(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()
    
    # Also try: full support with non-uniform amplitudes
    # Use a Gaussian-like profile
    if time.time() - t_start < _DEADLINE - 15.0:
        centers = [N//2, N//4, 3*N//4]
        widths = [5, 8, 12]
        for c in centers:
            for w in widths:
                h = np.exp(-((np.arange(N) - c)**2) / (2*w*w))
                h = h / h.sum()
                sc = _score(h)
                if sc > best_score:
                    best_score = sc
                    best_h = h.copy()
        
        # Triangular
        h = np.maximum(0, 1 - np.abs(np.arange(N) - N//2) / (N//2))
        h = h / h.sum()
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Final: try a few random full-support patterns
    if time.time() - t_start < _DEADLINE - 10.0:
        for _ in range(10):
            h = np.random.rand(N)
            h = h / h.sum()
            sc = _score(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()
            if time.time() - t_start > _DEADLINE - 10.0:
                break
    
    # Ensure nonnegative and return as list
    best_h = np.maximum(0, best_h)
    if best_h.sum() < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
