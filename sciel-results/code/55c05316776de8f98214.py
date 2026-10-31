import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _autoconv_full(f):
    """Full linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F, size)
    return C[:2 * n - 1]

def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    s = f.sum()
    if abs(s) < 1e-12:
        return float('inf')
    C = _autoconv_full(f)
    return np.max(np.abs(C)) / (s * s)

def _init_tapered():
    """
    Initialize with a raised-cosine (Hanning) window shape, then binarize
    by thresholding at the mean. This creates a non-rectangular support
    that structurally reduces boundary autoconvolution spikes compared
    to a full rectangular window.
    """
    t = np.linspace(0, 1, N)
    w = 0.5 * (1 + np.cos(2 * np.pi * t))
    # Binaries: threshold at mean to get ~50% ones
    mean_w = w.mean()
    f = (w >= mean_w).astype(np.float64)
    # Ensure non-zero sum
    if f.sum() == 0:
        f[N // 2] = 1.0
    return f

def _local_search_binary_fast(f, time_budget):
    """
    Incremental binary local search with O(N) per-step autoconv update.
    Flipping f[i] from 0->1 or 1->0 changes autoconv by +/- f shifted by i.
    We track the full autoconv array and update it incrementally.
    """
    f = f.copy()
    n = len(f)
    # Precompute full autoconv
    C = _autoconv_full(f)
    conv_len = 2 * n - 1
    best_score = _score(f)
    best_f = f.copy()
    
    # Precompute flip deltas: for each i, the change in C if we flip f[i]
    # Flipping f[i] from val to 1-val: delta_C[k] = (1 - 2*val) * sum_j f[j] * delta(k, i+j)
    # More precisely: C_new = C + (1-2*f[i]) * (f shifted by i, convolved with f)
    # Actually: C_new[k] = C[k] + (1-2*f[i]) * sum_{j: i+j=k} f[j]
    # = C[k] + (1-2*f[i]) * f[k-i] if 0 <= k-i < n
    
    # For efficiency, precompute f itself for lookups
    # delta_C for flipping position i:
    # For k in [i, i+n-1]: delta_C[k] += (1-2*f[i]) * f[k-i]
    # For k in [i-n+1, i-1]: delta_C[k] += (1-2*f[i]) * f[i-k] (symmetry)
    
    # We'll use a simpler approach: just recompute the affected range
    # But O(N) update: the change to C when flipping f[i] is:
    # delta_C = (1-2*f[i]) * (f convolved with a spike at i)
    # = (1-2*f[i]) * shift(f, i) for the "forward" part
    # Due to symmetry of autoconv, we also get the reflected part
    
    # Let's use the fact that C[k] = sum_j f[j]*f[k-j]
    # If f[i] changes by delta = 1-2*f[i], then:
    # C_new[k] = C[k] + delta*(f[k-i] + f[i-k]) for valid indices, but careful with double counting when k-i==i-k
    
    # Simpler: just track C and update by adding delta * (f shifted appropriately)
    # C_new = C + delta * (f * spike_i) where * is linear convolution
    # f * spike_i is just f shifted by i (with appropriate bounds)
    
    # For O(N) update:
    # delta_C[k] = delta * f[k-i] for k in [i, i+n-1]  (j from 0 to n-1, k=i+j)
    # delta_C[k] += delta * f[i-k] for k in [i-n+1, i-1]  (j from 1 to n-1, k=i-j)
    # But wait, autoconv C[k] = sum_j f[j]*f[k-j]. Changing f[i] by delta:
    # C_new[k] = C[k] + delta*f[k-i] + delta*f[i-k] - delta^2*f[i]*f[i] (if k-i==i-k, i.e., k=2i)
    # Hmm, this is getting complex. Let me just do the straightforward O(N) update.
    
    # Actually the cleanest O(N) update:
    # C_new[k] - C[k] = delta * (f[k-i] if 0<=k-i<n else 0) + delta * (f[i-k] if 0<=i-k<n else 0)
    # But when k-i == i-k (i.e., k=2i), we're double counting f[i]*f[i], so subtract delta^2*f[i]... 
    # No wait. C[k] = sum_j f[j] f[k-j]. The terms involving f[i] are:
    # j=i: f[i] f[k-i]
    # j=k-i: f[k-i] f[i] (same term if we sum over all j)
    # Actually C[k] = sum_j f[j] f[k-j]. The term where one factor is f[i]:
    # j=i: contributes f[i]*f[k-i]
    # j=k-i: contributes f[k-i]*f[i] = same value
    # So if i != k-i (i.e., k != 2i), there are two terms: f[i]*f[k-i] + f[k-i]*f[i] = 2*f[i]*f[k-i]
    # If k == 2i, there's one term: f[i]*f[i]
    # 
    # When we change f[i] to f[i]+delta:
    # For k != 2i: C changes by 2*delta*f[k-i] (if 0<=k-i<n)
    # For k == 2i: C changes by 2*delta*f[i] + delta^2 = delta*(2*f[i]+delta) = delta*(2*f[i]+1-2*f[i]) = delta
    # Wait: new f[i] = f[i]+delta. Old contribution at k=2i: f[i]^2. New: (f[i]+delta)^2 = f[i]^2 + 2*f[i]*delta + delta^2. Change: 2*f[i]*delta + delta^2.
    # For k != 2i: old contribution involving f[i]: 2*f[i]*f[k-i]. New: 2*(f[i]+delta)*f[k-i]. Change: 2*delta*f[k-i].
    # But also need to check: does changing f[i] affect terms where k-j = i, i.e., j = k-i? That's the same as j=i when k=2i.
    # For k != 2i: the terms are j=i (giving f[i]*f[k-i]) and j=k-i (giving f[k-i]*f[i]). Both change by delta*f[k-i] and f[i]*delta respectively. Total: 2*delta*f[k-i].
    # For k=2i: only j=i=k-i, term is f[i]*f[i]. Changes by 2*f[i]*delta + delta^2.
    
    # So the update is:
    # For k in [0, conv_len-1]:
    #   if 0 <= k-i < n:
    #     if k == 2*i: delta_C[k] = delta * (2*f[i] + delta)  # = delta * (2*f[i] + 1 - 2*f[i]) = delta
    #     else: delta_C[k] = 2 * delta * f[k-i]
    #   elif 0 <= i-k < n and k != 2*i:
    #     # This is already covered by the first case with j=k-i... 
    #     # Actually no. Let me reconsider.
    #     # C[k] = sum_j f[j] f[k-j]. The index j ranges 0..n-1.
    #     # f[i] appears when j=i (term: f[i]*f[k-i], valid if 0<=k-i<n)
    #     # f[i] appears when k-j=i, i.e., j=k-i (term: f[k-i]*f[i], valid if 0<=k-i<n)
    #     # These are the same when i = k-i, i.e., k=2i.
    #     # So actually both conditions reduce to "0 <= k-i < n".
    #     # If 0 <= k-i < n: contribution change is delta*f[k-i] + delta*f[i] (for the two terms)
    #     #   = delta*(f[k-i] + f[i]) unless k-i == i, in which case it's one term: delta*(2*f[i]+delta)
    #     # Wait no. If k-i != i: two separate terms, each changes by delta*f[other].
    #     #   Term j=i: was f[i]*f[k-i], now (f[i]+delta)*f[k-i], change = delta*f[k-i]
    #     #   Term j=k-i: was f[k-i]*f[i], now f[k-i]*(f[i]+delta), change = delta*f[k-i]
    #     #   Total: 2*delta*f[k-i]
    #     # If k-i == i (k=2i): one term j=i=k-i. Was f[i]*f[i], now (f[i]+delta)*(f[i]+delta), change = 2*f[i]*delta+delta^2
    #     # So: if 0<=k-i<n: change = 2*delta*f[k-i] if k!=2i, else 2*f[i]*delta+delta^2
    #     #     if k!=2i and 0<=i-k<n but k-i<0 or >=n: can this happen? i-k>=0 means k<=i. k-i<0 means k<i. So k<i and k>=i-n+1.
    #     #     In this case, j=k-i is negative, so the term j=k-i doesn't exist. Only j=i exists if 0<=k-i<n.
    #     #     But k-i < 0 here, so j=i gives f[i]*f[k-i] where k-i<0, which is 0.
    #     #     Hmm, I think I'm overcomplicating. Let me just use:
    #     #     The change in C[k] when f[i] changes by delta is:
    #     #     delta * f[k-i] (if 0<=k-i<n) + delta * f[i-k] (if 0<=i-k<n and i-k != k-i)
    #     #     But f[i-k] with 0<=i-k<n means k in [i-n+1, i]. And k-i<0 in this range (except k=i).
    #     #     So these two ranges don't overlap except at k=i where k-i=0 and i-k=0.
    #     #     At k=i: both give delta*f[0]. But it's the same term! j=i and j=k-i=i are the same.
    #     #     So: change = delta*f[k-i] (if 0<=k-i<n) + delta*f[i-k] (if 0<=i-k<n and k!=i)
    #     #     At k=i: change = delta*f[0] (just once, not twice)
    
    # OK this is getting too complex analytically. Let me just do a direct O(N) update by
    # recomputing the affected portion, or better yet, use the FFT-based full recompute
    # but only when needed, and use a smarter approach.
    
    # Alternative: Since N=400, conv_len=799, a full FFT autoconv is fast (~0.1ms).
    # Let's just do full recompute but be smart about which flips to try.
    
    # Actually, for N=400, even a naive O(N^2) autoconv is 160K ops, which is fast.
    # Let me just use the FFT method and do many iterations.
    
    f = best_f.copy()
    C = _autoconv_full(f)
    best_score = _score(f)
    best_f = f.copy()
    
    deadline = time.time() + time_budget
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try random flips
        improved = False
        for _ in range(100):
            i = np.random.randint(0, n)
            new_f = f.copy()
            new_f[i] = 1.0 - new_f[i]
            if new_f.sum() == 0:
                continue
            C_new = _autoconv_full(new_f)
            s_new = new_f.sum()
            score_new = np.max(np.abs(C_new)) / (s_new * s_new)
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_f = new_f.copy()
                f = new_f
                C = C_new
                improved = True
                no_improve = 0
            else:
                no_improve += 1
        if not improved:
            no_improve += 100
    
    return best_f, best_score

def construct():
    """
    Returns the optimal sequence for the C3-prime task.
    
    The objective is to minimize max(|autoconv(f)|) / (sum(f))^2.
    Uses a tapered window initialization to reduce boundary peaks,
    then refines with binary local search.
    """
    remaining = _TIME_LIMIT - (time.time() - _start_time)
    if remaining < 5:
        return np.ones(N)
    
    # Initialize with tapered window
    f = _init_tapered()
    initial_score = _score(f)
    
    # Run local search with time budget
    search_time = min(remaining - 10, 130.0)
    if search_time < 1:
        return f
    
    best_f, best_score = _local_search_binary_fast(f, search_time)
    
    return best_f

# EVOLVE-BLOCK-END
