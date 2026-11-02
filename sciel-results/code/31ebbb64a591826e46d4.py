"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 140  # 10s safety margin
_w = 0.5 / N  # interval width


def _autoconv_l1_bound(h):
    """Compute the L1 norm of the discrete autocorrelation, scaled by w."""
    R = np.correlate(h, h, mode='full')
    return _w * np.sum(np.abs(R))


def _fft_correlation(h):
    """Compute full autocorrelation via FFT for speed."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    F = np.fft.rfft(h, n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n_fft)[:2 * n - 1]
    # R[k] for k=0..2n-2 corresponds to lag k-(n-1)
    # We need the full correlation: lags from -(n-1) to (n-1)
    # np.correlate(h, h, 'full') gives lags from -(n-1) to (n-1)
    # FFT gives circular correlation; for zero-padded case it matches
    # Reorder: FFT result at index k corresponds to lag k for k < n, 
    # and lag k-n for k >= n (negative lags)
    # Actually for autocorrelation of real signal, R_fft[k] = sum h[i]*h[(i+k)%n]
    # With zero-padding to 2n-1, this equals the linear autocorrelation
    # R_fft[0] = R[0] (lag 0), R_fft[1] = R[-(n-1)] or R[1]?
    # Let's just use np.correlate for correctness since N=400 is small
    # But for speed in the inner loop, let's precompute properly
    return R


def _evaluate_flip(h, idx, current_bound):
    """Estimate change in bound if we flip h[idx] from +1 to -1 or vice versa.
    
    Flipping h[idx] changes R[k] for all k. Specifically, R[k] = sum_i h[i]*h[i+k].
    When we flip h[idx] (multiply by -1), the change in R[k] is:
    delta_R[k] = -2 * h[idx] * (contribution of h[idx] to R[k])
    
    For lag k: h[idx] contributes h[idx]*h[idx-k] to R[k] (if valid index)
    and h[idx]*h[idx+k] to R[-k]... this gets complicated.
    
    Let's just recompute via FFT for simplicity and speed.
    """
    pass


def _hill_climb(h, time_limit):
    """Greedy coordinate descent: repeatedly flip the sign that most reduces the bound."""
    n = len(h)
    h = h.copy()
    current = _autoconv_l1_bound(h)
    
    # Precompute FFT size
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    def _bound_from_fft(h):
        F = np.fft.rfft(h, n_fft)
        power = F * np.conj(F)
        R = np.fft.irfft(power, n_fft)[:2 * n - 1]
        # R from FFT: R[k] for k=0..2n-2
        # For autocorrelation with zero-padding, R_fft[k] = sum_{i} h[i]*h[(i+k) mod n_fft]
        # Since we padded to >= 2n-1, this equals linear autocorrelation
        # But the ordering: R_fft[0]=lag0, R_fft[1]=lag1, ..., R_fft[n-1]=lag(n-1)
        # R_fft[n]=lag-(n-1), R_fft[n+1]=lag-(n-2), ...
        # np.correlate(h,h,'full') gives: index 0 = lag-(n-1), ..., index n-1 = lag 0, ..., index 2n-2 = lag(n-1)
        # So we need to reorder
        # R_correlate[k] = R_fft[(k + n - 1) % (2*n - 1)]? Let me check
        # Actually, let's just use np.correlate since N=400, it's fast enough
        return _w * np.sum(np.abs(R))
    
    # For N=400, np.correlate is fast. Let's benchmark and decide.
    # Use np.correlate for simplicity and correctness.
    
    no_improve = 0
    max_no_improve = 50
    indices = np.arange(n)
    
    while time.time() < time_limit:
        # Shuffle order to avoid bias
        np.random.shuffle(indices)
        
        improved = False
        for i in indices:
            if time.time() > time_limit:
                break
            
            # Try flipping h[i]
            old_val = h[i]
            h[i] = -old_val
            new_bound = _autoconv_l1_bound(h)
            
            if new_bound < current - 1e-12:
                current = new_bound
                improved = True
            else:
                h[i] = old_val  # revert
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    
    return h, current


def construct():
    global _deadline
    
    # Start with balanced random signs
    rng = np.random.default_rng(42)
    h_init = rng.choice([-1.0, 1.0], size=N)
    # Ensure exact balance: 200 +1, 200 -1
    n_pos = np.sum(h_init > 0)
    if n_pos != N // 2:
        if n_pos > N // 2:
            idx_pos = np.where(h_init > 0)[0]
            to_flip = idx_pos[n_pos - N // 2:]
            h_init[to_flip] = -1.0
        else:
            idx_neg = np.where(h_init < 0)[0]
            to_flip = idx_neg[N // 2 - n_pos:]
            h_init[to_flip] = 1.0
    
    best_h = h_init.copy()
    best_score = _autoconv_l1_bound(best_h)
    
    # Try a few different random initializations with hill climbing
    seeds = [42, 123, 7, 99, 2024]
    for seed in seeds:
        if time.time() > _deadline:
            break
        
        rng_s = np.random.default_rng(seed)
        h_s = rng_s.choice([-1.0, 1.0], size=N)
        n_pos = np.sum(h_s > 0)
        if n_pos != N // 2:
            if n_pos > N // 2:
                idx_pos = np.where(h_s > 0)[0]
                h_s[idx_pos[n_pos - N // 2:]] = -1.0
            else:
                idx_neg = np.where(h_s < 0)[0]
                h_s[idx_neg[N // 2 - n_pos:]] = 1.0
        
        h_opt, score_opt = _hill_climb(h_s, _deadline - 5)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    return best_h
# EVOLVE-BLOCK-END
