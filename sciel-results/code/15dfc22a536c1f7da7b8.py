"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return np.max(np.abs(acf[1:]))


def _flip_cost(h, i, acf):
    """
    Estimate change in max sidelobe when flipping h[i].
    We use the fact that flipping h[i] changes the autocorrelation at lag k
    by -4*h[i]*h[(i+k) % n] for lag k (circular) but our acf is linear.
    Actually, for linear autocorrelation, flipping h[i] changes acf[k] for
    k where both i and i+k are in [0, n-1].
    The change in acf[k] is: delta_acf[k] = -4 * h[i] * h[j] where j = i+k or i-k.
    
    More precisely, acf[k] = sum_{t=0}^{n-1-k} h[t] * h[t+k].
    If we flip h[i] (i.e., h[i] -> -h[i]), then for each t such that t=i or t+k=i:
      - if t=i: term h[i]*h[i+k] becomes -h[i]*h[i+k], change = -2*h[i]*h[i+k]
      - if t+k=i i.e. t=i-k: term h[i-k]*h[i] becomes h[i-k]*(-h[i]), change = -2*h[i-k]*h[i]
    So total change in acf[k] is:
      - if i < n-k: -2*h[i]*h[i+k]  (from t=i)
      - if i >= k: -2*h[i-k]*h[i]  (from t=i-k)
    Both can apply if k <= i < n-k.
    
    For simplicity and speed, we compute the exact change for all k.
    """
    n = len(h)
    # Compute the change in each acf[k] for k=1..n-1
    # delta_acf[k] = -2*h[i]*h[i+k] (if i+k < n) - 2*h[i-k]*h[i] (if i-k >= 0)
    # Note: h[i] is the current value before flip.
    
    # We need to find the new max of |acf[k] + delta_acf[k]| for k=1..n-1.
    # To do this efficiently, we can precompute acf and then for each k compute delta.
    
    # But doing this for all k is O(n). We want to be faster.
    # Alternative: just compute the new acf directly? That's O(n^2) which is too slow.
    
    # Let's just compute the delta for all k in O(n).
    delta = np.zeros(n)
    hi = h[i]
    for k in range(1, n):
        d = 0.0
        if i + k < n:
            d -= 2.0 * hi * h[i + k]
        if i - k >= 0:
            d -= 2.0 * h[i - k] * hi
        delta[k] = d
    
    new_acf = acf + delta
    return np.max(np.abs(new_acf[1:]))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    np.random.seed(12345)
    
    # Start with a random +/-1 sequence
    best_h = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
    best_val = _max_sidelobe(best_h)
    best_acf = None
    
    # Compute best_acf once
    n = N
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(best_h, n=n_fft)
    power = F * np.conj(F)
    best_acf = np.fft.irfft(power, n=n_fft).real[:n]
    
    # Greedy local search with efficient cost estimation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single flips, pick the best one
        best_delta_val = best_val
        best_i = -1
        for i in range(N):
            if time.time() > deadline:
                break
            new_val = _flip_cost(best_h, i, best_acf)
            if new_val < best_delta_val:
                best_delta_val = new_val
                best_i = i
        
        if best_i >= 0 and best_delta_val < best_val:
            best_h[best_i] = -best_h[best_i]
            best_val = best_delta_val
            # Recompute acf for the new best_h
            F = np.fft.rfft(best_h, n=n_fft)
            power = F * np.conj(F)
            best_acf = np.fft.irfft(power, n=n_fft).real[:n]
            improved = True
    
    # If time remains, try simulated annealing
    if time.time() < deadline:
        h = best_h.copy()
        current_val = best_val
        T = 0.5
        cooling = 0.9995
        
        while time.time() < deadline and T > 0.005:
            i = np.random.randint(N)
            h[i] = -h[i]
            new_val = _flip_cost(h, i, best_acf)  # Note: this uses best_acf, not h's acf
            # Actually, _flip_cost assumes the acf corresponds to h before flip.
            # We need to be careful. Let's just use the full computation for SA.
            
            # For SA, let's just compute the actual max sidelobe
            n_fft_sa = 1 << (2 * n - 1).bit_length()
            F = np.fft.rfft(h, n=n_fft_sa)
            power = F * np.conj(F)
            acf = np.fft.irfft(power, n=n_fft_sa).real[:n]
            actual_new_val = np.max(np.abs(acf[1:]))
            
            delta = actual_new_val - current_val
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_val = actual_new_val
                if actual_new_val < best_val:
                    best_h = h.copy()
                    best_val = actual_new_val
            else:
                h[i] = -h[i]  # revert
            
            T *= cooling
    
    return best_h
# EVOLVE-BLOCK-END
