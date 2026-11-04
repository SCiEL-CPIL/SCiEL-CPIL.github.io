"""
Optimized PSL minimization for signed step function with N=400 intervals.
Uses vectorized single-flip gradient descent with incremental ACF updates.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf


def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _update_acf_flip(h, acf, i):
    """
    Update ACF after flipping h[i]. Returns new acf (full array).
    Uses the fact that flipping h[i] changes acf[k] by -4*h[i]*h[i±k] terms.
    """
    n = len(h)
    new_acf = acf.copy()
    
    # For each lag k (1..n-1), the change in acf[k] when flipping h[i]:
    # acf[k] = sum_j h[j]*h[j+k]
    # Flipping h[i] affects terms where j=i (h[i]*h[i+k]) and j=i-k (h[i-k]*h[i])
    # Also for the symmetric part... actually ACF via FFT is circular, but we use linear.
    # Let's just recompute the affected lags.
    
    # Actually, for linear autocorrelation:
    # acf[k] = sum_{j=0}^{n-1-k} h[j]*h[j+k] for k >= 0
    # When we flip h[i], the terms that change are those where j=i or j+k=i
    # i.e., j=i (if i+k < n) contributes h[i]*h[i+k] -> -h[i]*h[i+k], change = -2*h[i]*h[i+k]
    # and j+k=i, i.e., j=i-k (if i-k >= 0) contributes h[i-k]*h[i] -> h[i-k]*(-h[i]), change = -2*h[i-k]*h[i]
    
    # So for each lag k:
    # delta[k] = -2*h[i]*h[i+k] (if i+k < n) - 2*h[i-k]*h[i] (if i-k >= 0)
    #          = -2*h[i]*(h[i+k] + h[i-k]) with boundary conditions
    
    # But acf from FFT is circular. Let me just recompute ACF after flip for safety,
    # but that's O(n log n). For speed, let me use the linear ACF formula.
    
    # Actually, let me just recompute the full ACF since n=400 is small.
    # But that defeats the purpose. Let me do the incremental update properly.
    
    # The ACF computed by FFT is circular autocorrelation. For our purposes,
    # we want the linear autocorrelation (zero-padded). Let me verify which one _score uses.
    # _score uses FFT with zero-padding, which gives linear autocorrelation.
    
    # For linear ACF: acf[k] = sum_{j=0}^{n-1-k} h[j]*h[j+k]
    # Flipping h[i]:
    #   If 0 <= i < n-k: term h[i]*h[i+k] changes by -2*h[i]*h[i+k]
    #   If k <= i < n: term h[i-k]*h[i] changes by -2*h[i-k]*h[i]
    #   Note: if both conditions hold (k <= i < n-k), both terms change.
    
    # Let's build the delta vector for all k in 1..n-1
    delta = np.zeros(n)  # delta[k] for k=0..n-1, we only use k>=1
    
    # Terms where j=i: valid if i + k < n, i.e., k < n-i
    # Contribution: -2*h[i]*h[i+k]
    for k in range(1, n - i):
        delta[k] += -2.0 * h[i] * h[i + k]
    
    # Terms where j+k=i: valid if i-k >= 0, i.e., k <= i
    # Contribution: -2*h[i-k]*h[i]
    for k in range(1, min(i, n-1) + 1):
        delta[k] += -2.0 * h[i - k] * h[i]
    
    new_acf = acf + delta
    return new_acf


def _vectorized_gradient_step(h, acf, n):
    """
    For each position i, compute the resulting PSL if we flip h[i].
    Returns array of new PSL values for each i.
    """
    # For each i, we need delta_i[k] for k=1..n-1
    # delta_i[k] = -2*h[i]*(h[i+k] if i+k<n else 0) - 2*h[i-k]*(if i-k>=0 else 0)*h[i]
    #            = -2*h[i]*(h[i+k]*1{i+k<n} + h[i-k]*1{i-k>=0})
    
    # Let's build a matrix D of shape (n, n-1) where D[i,k-1] = delta for flipping i at lag k
    # D[i, k-1] = -2*h[i]*(h[i+k] if i+k<n else 0) - 2*h[i-k]*(if i-k>=0 else 0)*h[i]
    
    # Vectorized: for each k, compute the contribution for all i
    # Let's do it differently: for each i, the set of affected lags and their deltas.
    
    # Actually, let me just loop over i and compute the new PSL for each.
    # n=400, so 400 iterations, each O(n) = O(n^2) = 160k ops, which is fast.
    
    new_psls = np.zeros(n)
    acf_lags = acf[1:n]  # shape (n-1,)
    
    for i in range(n):
        delta = np.zeros(n - 1)
        hi = h[i]
        
        # j=i terms: k such that i+k < n, i.e., k < n-i
        max_k1 = min(n - i, n)
        if max_k1 > 1:
            delta[:max_k1 - 1] += -2.0 * hi * h[i + 1:i + max_k1]
        
        # j+k=i terms: k such that i-k >= 0, i.e., k <= i
        max_k2 = min(i, n - 1)
        if max_k2 > 0:
            delta[:max_k2] += -2.0 * h[i - max_k2:i][::-1] * hi
        
        new_acf_lags = acf_lags + delta
        new_psls[i] = np.max(np.abs(new_acf_lags))
    
    return new_psls


def _fast_gradient_optimize(h, max_iters=2000, time_budget=30.0, start_time=None):
    """
    Fast gradient-based PSL optimization.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Find best flip
        new_psls = _vectorized_gradient_step(h, acf, n)
        best_i = int(np.argmin(new_psls))
        best_new_psl = new_psls[best_i]
        
        if best_new_psl >= current_psl - 1e-12:
            break
        
        # Apply flip
        h[best_i] = -h[best_i]
        acf = _update_acf_flip(h, acf, best_i)
        current_psl = _psl_value(acf, n)
        
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl


def _random_init(n, seed=0):
    """Random ±1 initialization."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=n)
    return h


def _m_sequence_init(n, seed=0):
    """
    Generate a maximal-length sequence using LFSR.
    """
    m = 9
    while (1 << m) - 1 < n:
        m += 1
    
    # Primitive polynomial taps for m=9: x^9 + x^4 + 1
    primitive_polys = {
        9: [0, 4],
        10: [0, 3],
        11: [0, 2],
        12: [0, 4],
        13: [0, 4],
        14: [0, 5],
        15: [0, 1],
    }
    
    if m not in primitive_polys:
        primitive_polys[m] = [0, m // 2]
    
    taps = primitive_polys[m]
    
    state = np.zeros(m, dtype=np.int32)
    state[0] = 1
    
    seq = []
    for _ in range(n):
        seq.append(state[0])
        feedback = state[0]
        for tap in taps:
            feedback ^= state[tap]
        state = np.roll(state, -1)
        state[0] = feedback
    
    h = np.array([1.0 if b == 1 else -1.0 for b in seq])
    return h


def _multi_flip_search(h, time_budget=20.0, start_time=None):
    """
    Multi-flip local search: try flipping 2 positions at a time.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl
    
    rng = np.random.default_rng(42)
    
    max_attempts = 5000
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() - start_time > time_budget:
            break
        
        attempts += 1
        
        # Pick two random positions
        i, j = rng.choice(n, size=2, replace=False)
        i, j = int(i), int(j)
        
        if i == j:
            continue
        
        # Compute new PSL after flipping both i and j
        # First flip i
        h[i] = -h[i]
        acf_i = _update_acf_flip(h, acf, i)
        
        # Then flip j
        h[j] = -h[j]
        acf_ij = _update_acf_flip(h, acf_i, j)
        
        new_psl = _psl_value(acf_ij, n)
        
        if new_psl < best_psl:
            best_psl = new_psl
            best_h = h.copy()
            acf = acf_ij
            current_psl = new_psl
        else:
            # Revert
            h[j] = -h[j]
            h[i] = -h[i]
    
    return best_h, best_psl


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Try multiple diverse initializations.
    2. For each, run fast gradient-based PSL optimization.
    3. Use multi-flip search to escape local minima.
    4. Return the best sequence found.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Gradient optimization from multiple diverse starts
    n_starts = 10
    for s in range(n_starts):
        if time.time() > deadline - 40:
            break
        
        # Different initializations
        candidates = []
        candidates.append(_random_init(N, seed=s * 7919 + 13))
        candidates.append(_m_sequence_init(N, seed=s))
        
        # Also try a structured initialization
        rng = np.random.default_rng(s * 104729 + 7)
        h_struct = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h_struct)
        
        for h_init in candidates:
            if time.time() > deadline - 40:
                break
            
            remaining = deadline - 40 - (time.time() - start_time)
            if remaining < 5:
                break
            
            h_opt, psl_opt = _fast_gradient_optimize(
                h_init,
                max_iters=3000,
                time_budget=min(15.0, remaining),
                start_time=start_time
            )
            
            if psl_opt < best_score:
                best_score = psl_opt
                best_h = h_opt.copy()
    
    # Phase 2: Multi-flip search from best found
    if best_h is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - (time.time() - start_time)
        if remaining > 5:
            h_mf, psl_mf = _multi_flip_search(
                best_h,
                time_budget=remaining,
                start_time=start_time
            )
            if psl_mf < best_score:
                best_score = psl_mf
                best_h = h_mf
    
    # Phase 3: Final gradient polish
    if best_h is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - (time.time() - start_time)
        if remaining > 2:
            h_final, psl_final = _fast_gradient_optimize(
                best_h,
                max_iters=1000,
                time_budget=remaining,
                start_time=start_time
            )
            if psl_final < best_score:
                best_score = psl_final
                best_h = h_final
    
    # Ensure valid result
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h

# EVOLVE-BLOCK-END
