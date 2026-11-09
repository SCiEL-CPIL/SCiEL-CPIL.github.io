"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv(x):
    """Compute full autoconvolution using FFT."""
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * np.conj(f), size=size)
    return ac

def _score(x):
    """Compute the C3' score: max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _barker_init(n):
    """Generate a Barker-like initial pattern."""
    # Barker code of length 13, then repeat/extend
    barker = np.array([1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1])
    x = np.tile(barker, n // len(barker) + 1)[:n]
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 115s for safety
    
    # Initial pattern: Barker-like
    x = _barker_init(N).astype(float)
    best_x = x.copy()
    best_score = _score(x)
    
    # Also try all +1 and all -1 as baselines
    x_ones = np.ones(N)
    s_ones = _score(x_ones)
    if s_ones < best_score:
        best_score = s_ones
        best_x = x_ones.copy()
    
    x_neg = -np.ones(N)
    s_neg = _score(x_neg)
    if s_neg < best_score:
        best_score = s_neg
        best_x = x_neg.copy()
    
    # Peak shaving: iteratively flip the sign of the element most correlated with the max peak
    x = best_x.copy()
    current_score = best_score
    
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        s = np.sum(x)
        if s == 0:
            # Try to fix zero sum
            x[np.argmin(np.abs(x))] *= -1
            continue
        
        current_score = max_ac / (s * s)
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Find the lag with maximum |autoconv|
        lag_idx = np.argmax(np.abs(ac))
        
        # The autoconv at lag k is sum_i x[i] * x[i+k]
        # To reduce this peak, we want to flip a sign that is most "aligned" with the peak
        # Compute correlation of x with its shifted version at the peak lag
        if lag_idx == 0:
            # Peak at lag 0 is always N (for ±1), can't reduce
            # Try to reduce other peaks instead
            ac_no0 = ac.copy()
            ac_no0[0] = 0
            lag_idx = np.argmax(np.abs(ac_no0))
            if np.abs(ac_no0[lag_idx]) < 1e-10:
                break
        
        # Shifted version
        if lag_idx > 0:
            # x[i] correlates with x[i+lag_idx] for i in range(N-lag_idx)
            # The contribution to ac[lag_idx] is sum_{i=0}^{N-lag-1} x[i]*x[i+lag]
            # Flipping x[j] changes ac[lag] by -2*x[j]*x[j+lag] (if j+lag < N)
            # and also affects other lags
            # Simple heuristic: flip the element that has the largest |x[i]*x[i+lag]| contribution
            # which is always 1, so pick the one that reduces the peak most
            # Try flipping each element and see which reduces the max peak most (expensive)
            # Faster: flip the element at position that is "most correlated" 
            # Use the gradient approach: the derivative of ac[lag] w.r.t. x[j] is x[j+lag] (for j+lag<N)
            # So flipping x[j] changes ac[lag] by -2*x[j]*x[j+lag]
            
            # Compute the impact of flipping each x[j] on the peak lag
            # For lag L, flipping x[j] changes ac[L] by -2*x[j]*x[j+L] if j+L < N
            # We want to minimize the new max|ac|, so pick j that most reduces ac[L]
            
            L = lag_idx
            # Impact on ac[L] from flipping x[j]:
            # If j + L < N: delta_ac_L = -2 * x[j] * x[j+L]
            # We want delta_ac_L to have opposite sign to ac[L]
            # So if ac[L] > 0, we want delta_ac_L < 0, i.e., x[j]*x[j+L] > 0
            # If ac[L] < 0, we want delta_ac_L > 0, i.e., x[j]*x[j+L] < 0
            
            # Find j that maximizes the reduction
            # For ac[L] > 0: maximize x[j]*x[j+L] (want positive)
            # For ac[L] < 0: maximize -x[j]*x[j+L] (want negative product)
            
            if L < N:
                prod = x[:N-L] * x[L:]
                if ac[L] > 0:
                    j = np.argmax(prod)
                else:
                    j = np.argmin(prod)
                
                # Flip x[j]
                x[j] *= -1
            else:
                break
        else:
            # All peaks are at lag 0 or too small, try random flips
            j = np.random.randint(N)
            x[j] *= -1
    
    return best_x
# EVOLVE-BLOCK-END

