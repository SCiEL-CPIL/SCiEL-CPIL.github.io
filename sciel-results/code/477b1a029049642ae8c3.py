"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autocorr(s):
    """Compute autocorrelation for lags 0..N-1."""
    n = len(s)
    ac = np.zeros(n)
    for k in range(n):
        ac[k] = np.dot(s[:n-k], s[k:])
    return ac


def _peak_sidelobe(ac):
    """Peak of |ac[k]| for k=1..N-1."""
    if len(ac) <= 1:
        return 0.0
    return np.max(np.abs(ac[1:]))


def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    s = np.ones(N, dtype=np.float64)
    ac = _compute_autocorr(s)
    best_val = _peak_sidelobe(ac)
    best_s = s.copy()
    
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Compute delta for flipping s[i]
            old = s[i]
            # Delta for each lag k
            # s[i] appears in R(k) as:
            #   term1: s[i]*s[i+k] if i+k < N
            #   term2: s[i-k]*s[i] if i-k >= 0
            # Flipping s[i] from old to -old changes these by -2*old*s[neighbor]
            # So delta_ac[k] = -2*old*(s[i+k] if i+k<N else 0) - 2*old*(s[i-k] if i-k>=0 else 0)
            
            # Only lags k where i is in range [k, N-k-1] are affected
            # k <= i and k <= N-1-i, i.e., k <= min(i, N-1-i)
            # Also k >= 0
            max_k = min(i, N - 1 - i)
            if max_k < 0:
                continue
            
            # Compute delta for affected lags
            # For lag k in 0..max_k:
            #   delta = -2*old*s[i+k] - 2*old*s[i-k]
            # But wait, for k=0, s[i]*s[i] is the only term, so delta = -2*old*s[i] = -2*old^2 = -2
            # Actually for k=0: R(0) = sum s[j]^2 = N, flipping doesn't change R(0) since s[j]^2=1 always.
            # So R(0) is always N. We only care about k>=1.
            
            # For k >= 1:
            # delta_ac[k] = -2*old*s[i+k] - 2*old*s[i-k]
            
            new_ac = ac.copy()
            for k in range(1, max_k + 1):
                delta = -2.0 * old * s[i + k] - 2.0 * old * s[i - k]
                new_ac[k] = ac[k] + delta
            
            # Also need to check if flipping affects any lag > max_k
            # For k > max_k, s[i] doesn't appear in R(k), so no change.
            
            val = _peak_sidelobe(new_ac)
            if val < best_val:
                # Accept flip
                s[i] = -old
                ac = new_ac
                best_val = val
                best_s = s.copy()
                improved = True
        if time.time() > deadline:
            break
    
    return best_s
# EVOLVE-BLOCK-END
