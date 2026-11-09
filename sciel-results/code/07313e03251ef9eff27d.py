import numpy as np
import time

N = 400

def _compute_autoconv(f):
    """Compute full autoconvolution via FFT."""
    size = 1
    n = len(f)
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto

def _objective_excl_zero(f, auto):
    """Compute max(|autoconv| for k!=0) / (sum(f))^2"""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    # Exclude k=0 (index 0)
    # Autoconv array has indices 0 to 2N-2.
    # k=0 is at index 0.
    # We want max abs of indices 1 to 2N-2.
    if len(auto) > 1:
        max_abs = np.max(np.abs(auto[1:]))
    else:
        max_abs = 0.0
    return max_abs / (s * s)

def _update_autoconv_inplace(auto, f, idx, old_val):
    """
    Update autoconvolution in place when f[idx] is flipped.
    old_val is the value before flip.
    new_val is -old_val.
    f[idx] should be updated BEFORE calling this? 
    Let's say f[idx] is already flipped.
    old_val is the previous value.
    
    C(k) = sum_{i} f_i f_{i+k}
    Delta C(k) = -2 * old_val * (f_{idx-k} + f_{idx+k})
    Note: bounds must be checked.
    """
    new_val = -old_val
    # The change in sum of products involving f[idx]
    # Terms: f[idx] * f[idx+k] and f[idx-k] * f[idx]
    # Old contribution: old_val * f[idx+k] + f[idx-k] * old_val
    # New contribution: new_val * f[idx+k] + f[idx-k] * new_val
    # Delta = (new_val - old_val) * (f[idx+k] + f[idx-k]) = -2 * old_val * (f[idx+k] + f[idx-k])
    
    # We need to update auto[0] to auto[2N-2]
    # auto[k] corresponds to lag k? 
    # np.fft.irfft gives circular correlation? No, linear.
    # Index 0 is lag 0. Index 1 is lag 1. ... Index N-1 is lag N-1.
    # Index N is lag -(N-1)? 
    # Standard linear autoconvolution via FFT:
    # C(k) for k >= 0 is at index k.
    # C(k) for k < 0 is at index 2N-2 + k? 
    # Actually, for real symmetric autoconvolution, C(k) = C(-k).
    # The array returned by irfft of |F|^2 is the linear autoconvolution.
    # Indices 0 to N-1 are lags 0 to N-1.
    # Indices N to 2N-2 are lags -(N-1) to -1.
    
    for k in range(1, N):
        # Lag k
        # Indices in f: idx and idx+k
        # If idx+k < N, term exists in C(k) and C(-k)
        
        # Update C(k) (index k)
        # Contribution from f[idx] * f[idx+k]
        if idx + k < N:
            auto[k] += -2 * old_val * f[idx + k]
        
        # Update C(-k) (index 2N-2-k)
        # Contribution from f[idx-k] * f[idx]
        if idx - k >= 0:
            auto[2 * N - 2 - k] += -2 * old_val * f[idx - k]
            
        # Note: C(0) is also affected.
        # C(0) = sum f_i^2. Flipping sign doesn't change f_i^2.
        # So C(0) is invariant. We don't need to update index 0.
        # But wait, the formula above for k=0:
        # auto[0] += -2 * old_val * f[idx] (from idx+0) + ... 
        # Since f[idx] becomes -old_val, f[idx]^2 is same.
        # So auto[0] should not change.
        # My loop starts at k=1, so auto[0] is untouched. Correct.

def construct():
    deadline = time.time() + 110
    
    best_f = None
    best_score = float('inf')
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random signs
        f = rng.choice([-1, 1], size=N).astype(float)
        auto = _compute_autoconv(f)
        current_score = _objective_excl_zero(f, auto)
        
        # Local Search / Simulated Annealing
        # We want to minimize current_score
        temp = 0.1
        for step in range(5000):
            if time.time() > deadline - 5:
                break
                
            # Pick random index to flip
            idx = rng.integers(N)
            old_val = f[idx]
            
            # Flip
            f[idx] = -old_val
            
            # Update autoconv
            _update_autoconv_inplace(auto, f, idx, old_val)
            
            # Calculate new score
            new_score = _objective_excl_zero(f, auto)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                # Accept with probability exp(-(new_score - current_score) / temp)
                delta = new_score - current_score
                if delta > 0 and temp > 0:
                    if rng.random() < np.exp(-delta / temp):
                        current_score = new_score
                    else:
                        # Revert
                        f[idx] = old_val
                        _update_autoconv_inplace(auto, f, idx, -old_val) # Note: old_val is now the "new" val to be reverted?
                        # Wait, if we revert, we flip back.
                        # Current f[idx] is -old_val.
                        # We want to go back to old_val.
                        # So "old_val" for the update function should be the CURRENT value (-old_val).
                        # Let's be careful.
                        # State before flip: f[idx]=old_val.
                        # State after flip: f[idx]=-old_val.
                        # If we revert, we flip -old_val to old_val.
                        # So in _update_autoconv_inplace, the "old_val" argument should be -old_val.
                        pass # The update function logic:
                        # It assumes f[idx] is ALREADY flipped to the new value.
                        # And it uses the "old_val" (pre-flip) to calculate delta.
                        # So if we revert:
                        # Current f[idx] is -old_val.
                        # We want to flip it to old_val.
                        # So the "old_val" parameter for the function should be -old_val.
                        # And the function will add -2 * (-old_val) * ...
                        # Which is +2 * old_val * ...
                        # This correctly reverts the change.
                        _update_autoconv_inplace(auto, f, idx, -old_val)
                else:
                    # Revert
                    f[idx] = old_val
                    _update_autoconv_inplace(auto, f, idx, -old_val)
            
            # Anneal temperature
            temp *= 0.99

    if best_f is None:
        # Fallback to all ones
        best_f = np.ones(N)
        
    return best_f
