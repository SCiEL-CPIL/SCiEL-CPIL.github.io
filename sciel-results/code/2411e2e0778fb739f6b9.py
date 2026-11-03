import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _legendre_init():
    """Initialize with Legendre symbol sequence for low additive energy."""
    # Find a prime p such that p-1 >= N, or just use p=401 (prime)
    # Legendre symbol (a/p) = 1 if a is quadratic residue mod p, -1 if not, 0 if a=0
    p = 401
    h = np.zeros(N, dtype=np.float64)
    for i in range(N):
        a = i + 1  # avoid 0
        # Compute a^(p-1)/2 mod p
        # If result is 1, a is residue (+1); if p-1, non-residue (-1)
        if pow(a, (p - 1) // 2, p) == 1:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

# ---------------------------------------------------------------------------
# Autocorrelation Helpers
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))

def _flip_ac_fast(ac, h, i):
    """
    Vectorised O(N) update of ac after flipping h[i].
    h is the array BEFORE the flip.
    Returns the new ac array.
    """
    new_ac = ac.copy()
    hi = h[i]
    # Flipping h[i] changes the product h[j]*h[j+k] 
    # Delta for lag k:
    # If we flip h[i], the terms involving h[i] are h[i]*h[i+k] and h[i-k]*h[i].
    # Old contribution: h[i]*h[i+k]
    # New contribution: -h[i]*h[i+k]
    # Change: -2 * h[i] * h[i+k]
    
    # For k >= 0:
    # Term 1: h[i] is the first element. Valid for k where i+k < N.
    # k ranges from 0 to N-1-i.
    # The value h[i+k] is h[i : N].
    # So new_ac[0 : N-i] += -2 * h[i] * h[i : N]
    
    k_hi = N - 1 - i
    if k_hi >= 0:
        new_ac[0:k_hi + 1] += -2.0 * hi * h[i:i + k_hi + 1]
        
    # Term 2: h[i] is the second element. Valid for k where i-k >= 0.
    # k ranges from 0 to i.
    # The value h[i-k] is h[0 : i+1] reversed? No.
    # h[i-k] for k=0 is h[i]. For k=i is h[0].
    # So the sequence of h[i-k] for k=0..i is h[i], h[i-1], ..., h[0].
    # This corresponds to h[0:i+1][::-1].
    # So new_ac[0 : i+1] += -2 * h[i] * h[0:i+1][::-1]
    
    k_lo = i
    if k_lo >= 0:
        # Note: k=0 is included in both?
        # Term 1 at k=0: h[i]*h[i]. Change -2*h[i]*h[i] = -2.
        # Term 2 at k=0: h[i]*h[i]. Change -2*h[i]*h[i] = -2.
        # Wait, the autocorrelation is sum_{j} h[j]h[j+k].
        # If we flip h[i]:
        # The term j=i contributes h[i]h[i+k].
        # The term j=i-k contributes h[i-k]h[i].
        # These are distinct terms in the sum for k>0.
        # For k=0, the term is h[i]h[i]. It appears only once in the sum for k=0?
        # ac[0] = sum_{j=0}^{N-1} h[j]^2 = N.
        # If we flip h[i], ac[0] remains N.
        # Let's check the delta for k=0.
        # Term 1 (j=i): -2*h[i]*h[i] = -2.
        # Term 2 (j=i): -2*h[i]*h[i] = -2.
        # Total change -4? But ac[0] should not change.
        # Ah, for k=0, the "Term 1" and "Term 2" refer to the same product h[i]h[i].
        # The formula ac[k] = sum_{j=0}^{N-1-k} h[j]h[j+k].
        # For k=0, sum_{j=0}^{N-1} h[j]^2.
        # Flipping h[i] changes h[i]^2 to (-h[i])^2 = h[i]^2. No change.
        # My delta formula adds -2*h[i]*h[i+k] for the term where h[i] is the first factor.
        # And -2*h[i-k]*h[i] for the term where h[i] is the second factor.
        # For k=0, the first factor index is i, second factor index is i.
        # They are the same term.
        # So for k=0, we should only add the delta once, or ensure we don't double count.
        # Actually, for k>0, the indices j=i and j=i-k are distinct.
        # For k=0, j=i is the only term involving h[i].
        # So:
        # If k > 0: add both deltas.
        # If k == 0: add only one delta (or zero, since h[i]^2 is constant).
        
        # Let's re-verify.
        # ac_new[k] = ac_old[k] - 2*h[i]*h[i+k] - 2*h[i-k]*h[i]
        # For k=0: ac_new[0] = ac_old[0] - 2*h[i]*h[i] - 2*h[i]*h[i] = N - 4.
        # But ac_new[0] must be N.
        # So the formula is wrong for k=0.
        # Correctly:
        # The term h[i]h[i] is present in ac[0].
        # Flipping h[i] does not change h[i]^2.
        # So delta[0] should be 0.
        
        # For k > 0:
        # The term h[i]h[i+k] is in ac[k]. Flipping h[i] changes it to -h[i]h[i+k]. Delta: -2h[i]h[i+k].
        # The term h[i-k]h[i] is in ac[k]. Flipping h[i] changes it to h[i-k](-h[i]). Delta: -2h[i-k]h[i].
        # These are two different terms in the sum for ac[k] (unless i+k == i-k which implies k=0).
        
        # So:
        # new_ac[0] = ac[0] (no change)
        # new_ac[k] for k>0:
        
        # Let's implement carefully.
        pass
    return new_ac

# Re-implementing _flip_ac_fast correctly
def _flip_ac_fast(ac, h, i):
    new_ac = ac.copy()
    hi = h[i]
    
    # Update lags k > 0
    # 1. h[i] is the first element: pairs with h[i+k] for k=1..N-1-i
    #    Delta: -2 * h[i] * h[i+k]
    #    Indices in new_ac: 1 to N-1-i
    #    Values from h: h[i+1 : N]
    max_k1 = N - 1 - i
    if max_k1 >= 1:
        new_ac[1:max_k1 + 1] += -2.0 * hi * h[i + 1:i + max_k1 + 1]
        
    # 2. h[i] is the second element: pairs with h[i-k] for k=1..i
    #    Delta: -2 * h[i-k] * h[i]
    #    Indices in new_ac: 1 to i
    #    Values from h: h[0:i] reversed?
    #    For k=1, partner is h[i-1]. For k=i, partner is h[0].
    #    So the sequence of partners for k=1..i is h[i-1], h[i-2], ..., h[0].
    #    This is h[0:i][::-1].
    if i >= 1:
        new_ac[1:i + 1] += -2.0 * hi * h[0:i][::-1]
        
    # new_ac[0] remains unchanged (sum of squares)
    return new_ac

def _flip_pair_ac_fast(ac, h, i):
    """Update ac after flipping h[i] and h[i+1]."""
    # Flip i
    new_ac = _flip_ac_fast(ac, h, i)
    # Flip i+1 on the updated sequence
    h_tmp = h.copy()
    h_tmp[i] *= -1.0
    new_ac = _flip_ac_fast(new_ac, h_tmp, i + 1)
    return new_ac

# ---------------------------------------------------------------------------
# Search Algorithms
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    """
    Greedy single-flip and pair-flip polish.
    """
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    
    # We will do passes until no improvement or time runs out
    while time.time() - start < time_budget:
        improved = False
        
        # Single flip sweep
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < cur:
                h[i] *= -1.0
                ac = ac2
                cur = c
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
                    
        # Pair flip sweep
        for i in range(N - 1):
            ac2 = _flip_pair_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < cur:
                h[i] *= -1.0
                h[i+1] *= -1.0
                ac = ac2
                cur = c
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
                    
        if not improved:
            break
            
    return best_h, best

def _anneal(seed, time_budget, init_h=None):
    """
    Simulated Annealing with block flips.
    """
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    
    T = 4.0
    T_min = 1e-7
    start = time.time()
    
    while time.time() - start < time_budget:
        # Determine block length
        progress = (time.time() - start) / time_budget
        max_len = max(1, int(32.0 * (1.0 - 0.8 * progress)))
        L = int(rng.integers(1, max_len + 1))
        
        i = int(rng.integers(0, N - L + 1))
        
        # Flip block
        h[i:i + L] *= -1.0
        ac_new = _ac_full(h)
        new = _peak_sidelobe(h)
        
        # Acceptance criterion
        delta = new - cur
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            cur = new
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0 # Revert
            
        T = max(T_min, T * 0.999)
        
    return best_h, best

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()
        
        def _remaining():
            return hard_deadline - (time.time() - start)
            
        # Initialize with Legendre symbol sequence (better starting point)
        global_best_h = _legendre_init()
        global_best = _peak_sidelobe(global_best_h)
        
        # Phase 1: Quick Greedy Polish on the Legendre-initialized sequence
        budget = min(5.0, _remaining() - 10.0)
        if budget > 1.0:
            h, c = _greedy_polish(global_best_h, budget, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
                
        # Phase 2: Simulated Annealing with Block Flips, starting from polished Legendre
        budget = min(30.0, _remaining() - 20.0)
        if budget > 5.0:
            h, c = _anneal(12345, budget, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish the result of annealing
            sub = min(5.0, _remaining() - 20.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
                    
        # Phase 3: More Annealing Restarts if time permits
        restart_id = 100
        while _remaining() > 15.0:
            sub = min(10.0, _remaining() - 15.0)
            if sub < 2.0:
                break
            h, c = _anneal(restart_id, sub, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 15.0)
            if sub2 > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 100
            
        # Phase 4: Final High-Precision Polish
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
                
        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
