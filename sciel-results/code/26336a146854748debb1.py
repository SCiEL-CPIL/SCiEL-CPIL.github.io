import time
import numpy as np

N = 600
TOTAL_TIME = 150.0
SAFETY_MARGIN = 5.0  # seconds to leave before the hard kill

def _autoconv_peak(f):
    """Compute the normalized maximum of the autoconvolution of f.
    
    f is a step function with N equal intervals on [-1/4, 1/4].
    The autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    We compute this on a fine grid and return the max value divided by (sum f * dx)^2
    to get the normalized peak. Actually, let me reconsider the normalization.
    
    The task says "normalized maximum of its autoconvolution". Let me think about what
    normalization makes sense. For the uniform function f=1 on [-1/4,1/4], the 
    autoconvolution is a triangle function with peak at t=0 equal to the width = 1/2.
    The L1 norm squared is (1/2)^2 = 1/4. So normalized peak = (1/2)/(1/4) = 2.
    
    Wait, let me reconsider. The score for the seed (uniform) is 1.0, and the search
    score is seed/value. So if seed returns value V_seed, then score = V_seed / V.
    Since seed has score 1.0, V_seed / V_seed = 1.0. We want to minimize the raw value.
    
    Let me compute: for f = 1 on [-1/4, 1/4] (N=600 intervals, each of width dx=1/(2*N)):
    - L1 norm = sum(f_i * dx) = N * 1 * dx = 1/2
    - Autoconvolution peak at t=0: integral of f(s)f(-s) ds over s where both are in support
      = integral from -1/4 to 1/4 of 1*1 ds = 1/2
    - If normalized by L1^2 = 1/4, then value = (1/2)/(1/4) = 2
    
    But actually, maybe the normalization is different. Let me think about what gives
    a meaningful upper bound on C1 (the first autocorrelation constant).
    
    Actually, for the purpose of this task, I just need to compute a consistent metric
    that the evaluator will use. The key insight is: minimize the peak of the 
    autoconvolution relative to some normalization. Let me use:
    
    value = max_t (f*f)(t) / (integral f)^2
    
    This is scale-invariant, which makes sense for an upper bound on a constant.
    """
    dx = 1.0 / (2.0 * N)
    # Compute autoconvolution using FFT for efficiency
    # f is defined on [-1/4, 1/4] with N intervals
    # The autoconvolution (f*f)(t) for t in [-1/2, 1/2]
    # We can compute it via FFT
    
    # Create the signal: f on [-1/4, 1/4], zero elsewhere
    # For convolution, we need to handle the domain properly
    # Let's use a grid that covers [-1/2, 1/2] for the result
    
    # Actually, let's use numpy's convolve or FFT approach
    # The autoconvolution on the discrete grid:
    # (f*f)[k] = sum_j f[j] * f[k-j] * dx  (discrete approximation)
    
    # Using FFT:
    size = 2 * N  # enough to cover the full support of the autoconvolution
    f_padded = np.zeros(size)
    f_padded[:N] = f  # f is defined on the first N intervals
    
    # Linear convolution via FFT
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    # The peak of the autoconvolution
    peak = np.max(conv) * dx  # multiply by dx for the integral approximation
    
    # Normalization: (integral f)^2
    l1 = np.sum(f) * dx
    normalized_peak = peak / (l1 * l1)
    
    return normalized_peak

def construct():
    start_time = time.time()
    
    # Start with a non-uniform initialization to escape the uniform critical point
    # Use an asymmetric shape that reduces overlap of f(t) and f(-t)
    # Strategy: make f higher on one side and lower on the other, or use a 
    # shape that has less self-overlap
    
    # Try a few different initializations and pick the best
    best_f = None
    best_val = float('inf')
    
    # Deadline check
    deadline = start_time + TOTAL_TIME - SAFETY_MARGIN
    
    # Several candidate initializations
    initializations = []
    
    # 1. Asymmetric ramp: high on left, low on right
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    init1 = 1.0 + 0.5 * (t + 0.25)  # ranges from 1.0 to 1.5
    initializations.append(init1)
    
    # 2. Cosine shape
    init2 = 0.5 + 0.5 * np.cos(np.pi * t)  # peaks in middle
    initializations.append(init2)
    
    # 3. Two-bump asymmetric
    init3 = np.ones(N)
    init3[:N//4] *= 1.5
    init3[3*N//4:] *= 0.5
    initializations.append(init3)
    
    # 4. Gaussian-like centered
    init4 = np.exp(-20 * t**2)
    initializations.append(init4)
    
    # 5. Step function: high on one half
    init5 = np.ones(N)
    init5[:N//2] = 1.5
    init5[N//2:] = 0.5
    initializations.append(init5)
    
    # Evaluate all initializations
    for init in initializations:
        val = _autoconv_peak(init)
        if val < best_val:
            best_val = val
            best_f = init.copy()
    
    # Now do simulated annealing / gradient-based optimization
    # Use a simple coordinate descent / local search approach
    
    f = best_f.copy()
    current_val = best_val
    
    # Simulated annealing with coordinate moves
    T = 0.1  # initial temperature
    cooling_rate = 0.9995
    T_min = 0.001
    
    iteration = 0
    while time.time() < deadline and T > T_min:
        # Pick a random index to modify
        idx = np.random.randint(N)
        old_val = f[idx]
        
        # Try a random perturbation
        delta = np.random.uniform(-0.1, 0.1)
        new_val = max(0.0, old_val + delta)
        
        if new_val == old_val:
            continue
            
        f[idx] = new_val
        new_score = _autoconv_peak(f)
        
        # Accept or reject
        if new_score < current_val:
            current_val = new_score
            if current_val < best_val:
                best_val = current_val
                best_f = f.copy()
        else:
            # Metropolis acceptance
            if np.random.random() < np.exp(-(new_score - current_val) / max(T, 1e-10)):
                current_val = new_score
            else:
                f[idx] = old_val
        
        T *= cooling_rate
        iteration += 1
        
        # Periodic check
        if iteration % 100 == 0:
            if time.time() > deadline:
                break
    
    # Final polish: local optimization with smaller steps
    if time.time() < deadline:
        # Do a few rounds of coordinate descent with smaller steps
        for round_num in range(10):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                old_val = f[idx]
                # Try small perturbations
                for delta in [-0.01, 0.01]:
                    new_val = max(0.0, old_val + delta)
                    f[idx] = new_val
                    new_score = _autoconv_peak(f)
                    if new_score < current_val:
                        current_val = new_score
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_f = f.copy()
                    else:
                        f[idx] = old_val
                # Try zeroing out
                if f[idx] > 0:
                    f[idx] = 0.0
                    new_score = _autoconv_peak(f)
                    if new_score < current_val:
                        current_val = new_score
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_f = f.copy()
                    else:
                        f[idx] = old_val
    
    return list(best_f)

# Add a time guard at module level to be safe
if __name__ == "__main__":
    result = construct()
    print(f"Result: {len(result)} values, sum={sum(result):.4f}")
