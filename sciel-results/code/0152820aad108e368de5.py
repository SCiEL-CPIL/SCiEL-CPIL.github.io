import numpy as np
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Strategy: Use binary sequences (h in {0,1}) with N/2 ones and N/2 zeros.
    For binary h, the overlap C(k) = sum_i h[i]*(1-h[(i+k)%N]) = (N/2) - autocorr(k).
    Minimizing max C(k) is equivalent to minimizing max autocorr(k).
    
    We use a gradient-based local search that:
    1. Starts from a random binary sequence
    2. At each step, finds the shift k* with maximum autocorrelation
    3. Moves h in the direction that reduces autocorr at k* while maintaining sum
    4. Projects back to [0,1] and fixes the sum constraint
    
    The key structural change from the current program: instead of starting from
    the constant 0.5 solution and doing small perturbations, we use a proper
    projection-based gradient descent that can explore the full [0,1]^N space
    more effectively, potentially finding solutions with lower peak overlap.
    """
    N = 100
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s limit
    
    # Helper: compute autocorrelation via FFT
    def compute_autocorr(h):
        """Circular autocorrelation: ac[k] = sum_i h[i]*h[(i+k)%N]"""
        h_pad = np.concatenate([h, np.zeros(N)])
        F = np.fft.fft(h_pad, n=2*N)
        ac = np.real(np.fft.ifft(F * np.conj(F)))[:N]
        return ac
    
    # Helper: compute max overlap (excluding k=0)
    def max_overlap(h):
        ac = compute_autocorr(h)
        # C(k) = sum(h) - ac[k] for k=1..N-1
        C = h.sum() - ac[1:]
        return np.max(C)
    
    # Helper: project h to satisfy sum = N/2 and 0 <= h <= 1
    def project(h):
        """Project h to the feasible set: sum(h)=N/2, 0<=h<=1"""
        h = np.clip(h, 0, 1)
        # Fixed-point iteration to fix the sum
        for _ in range(50):
            diff = N/2.0 - h.sum()
            if abs(diff) < 1e-12:
                break
            # Distribute the difference evenly, but respect bounds
            # Increase where h < 1, decrease where h > 0
            h = np.clip(h + diff/N, 0, 1)
        # Final adjustment: if still off, adjust a few elements
        diff = N/2.0 - h.sum()
        if abs(diff) > 1e-10:
            # Find indices with room to adjust
            if diff > 0:
                # Need to increase sum: find indices with h < 1
                candidates = np.where(h < 1.0 - 1e-9)[0]
                if len(candidates) > 0:
                    # Add to the first candidate
                    h[candidates[0]] += diff
            else:
                # Need to decrease sum: find indices with h > 0
                candidates = np.where(h > 1e-9)[0]
                if len(candidates) > 0:
                    h[candidates[0]] += diff
        return h
    
    # Initialize with a random binary sequence
    rng = np.random.default_rng(42)
    h = np.zeros(N)
    indices = rng.permutation(N)
    h[indices[:N//2]] = 1.0
    h[indices[N//2:]] = 0.0
    
    best_h = h.copy()
    best_val = max_overlap(h)
    
    # Gradient-based local search
    step_size = 0.1
    for iteration in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        ac = compute_autocorr(h)
        max_ac = np.max(ac)
        k_star = np.argmax(ac)
        
        # Gradient of ac[k_star] wrt h[i] is h[(i+k_star)%N]
        grad = np.roll(h, k_star)
        
        # We want to decrease ac[k_star], so move h in direction -grad
        # But we need to maintain sum(h) = N/2, so subtract the mean
        # mean(grad) = mean(h) = 0.5 (since sum(h) = N/2)
        delta = -step_size * (grad - 0.5)
        
        h_new = h + delta
        h_new = project(h_new)
        
        val = max_overlap(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # If no improvement, try a smaller step or random perturbation
            # Occasionally do a random binary flip to escape local minima
            if iteration % 100 == 0:
                # Try a random swap of two elements
                i, j = rng.integers(0, N, size=2)
                if i != j:
                    h_trial = h.copy()
                    # Swap values at i and j
                    h_trial[i], h_trial[j] = h_trial[j], h_trial[i]
                    h_trial = project(h_trial)
                    val_trial = max_overlap(h_trial)
                    if val_trial < best_val:
                        best_val = val_trial
                        best_h = h_trial.copy()
                        h = h_trial.copy()
    
    # Final validation and cleanup
    h = best_h.copy()
    h = np.clip(h, 0, 1)
    
    # Ensure sum is exactly N/2
    diff = N/2.0 - h.sum()
    if abs(diff) > 1e-10:
        # Adjust a single element
        if diff > 0:
            idx = np.argmin(h)
            h[idx] += diff
        else:
            idx = np.argmax(h)
            h[idx] += diff
        h = np.clip(h, 0, 1)
    
    # Final check
    assert np.isclose(h.sum(), N/2, atol=1e-8), f"Sum {h.sum()} != {N/2}"
    assert np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10)
    
    return h
