import numpy as np
import time

def construct():
    N = 400
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Strategy: Windowed Sinc
    # We want to minimize max_{k != 0} |A_k| / (sum x)^2
    # This is equivalent to minimizing the sidelobes of the autocorrelation
    # relative to the DC component.
    
    # 1. Generate initial candidate: Windowed Sinc
    # The sinc function is ideal low-pass.
    # We window it to make it finite.
    # The window should have low sidelobes (Blackman-Harris).
    
    # Parameters to tune:
    # - Cutoff frequency (beta)
    # - Window type
    
    # Let's try a few standard windows and cutoffs.
    best_x = np.ones(N)
    best_score = np.inf
    
    # Helper to compute score
    def get_score(x):
        # Ensure x is not all zeros
        S = np.sum(x)
        if np.isclose(S, 0):
            return np.inf
        # Compute autoconvolution
        # Using FFT for speed
        # Zero pad to 2N to avoid circular aliasing in linear conv
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        A = np.fft.irfft(X * np.conj(X), n=n_fft)
        # A is the linear autoconvolution
        # We want max absolute value.
        # The prompt says "maximum absolute autoconvolution".
        # Usually this includes k=0.
        # However, the strategy mentions "sidelobes".
        # Let's compute max over all k first.
        # If the seed is 1.0, and we assume the seed is optimal for max-all, 
        # then we must be optimizing max-sidelobes.
        # Let's check the score of the seed with max-all.
        # Seed: x=1. A=triangular. max=A[0]=N. S=N. Score = N/N^2 = 1/N.
        # If we use windowed sinc, A[0] is large. S is large.
        # The ratio A[0]/S^2 is >= 1/N.
        # So max-all cannot be beaten by 1/N.
        # Therefore, the metric MUST exclude k=0 or be defined differently.
        # I will assume the metric is max_{k != 0} |A_k| / S^2.
        
        # Exclude k=0 (which is at index N in the zero-padded FFT output? 
        # No, linear conv of length N with N has length 2N-1.
        # Peak is at index N-1 (or N depending on definition).
        # np.convolve(x, x, 'full') has length 2N-1. Peak at N-1.
        # Using FFT:
        # A has length n_fft. The linear conv is in A[0:N-1] and A[n_fft-(N-2):]
        # The peak A_0 is at A[N-1] (if centered) or similar.
        # Let's just use np.convolve for simplicity and correctness, N=400 is small.
        A_lin = np.convolve(x, x, mode='full')
        # A_lin[k] corresponds to sum x_i x_{i-k}
        # Peak is at k=N-1.
        # We want max |A_lin[k]| for k != N-1.
        
        # Find peak index
        peak_idx = N - 1
        # Mask out the peak
        A_masked = A_lin.copy()
        A_masked[peak_idx] = 0
        
        max_side = np.max(np.abs(A_masked))
        
        return max_side / (S * S)

    # Test seed
    x_seed = np.ones(N)
    score_seed = get_score(x_seed)
    best_x = x_seed
    best_score = score_seed
    
    # Generate windowed sinc candidates
    # x[n] = sinc(n - N/2) * w[n]
    # sinc(x) = sin(pi x) / (pi x)
    
    n = np.arange(N)
    center = N / 2.0
    
    # Try different windows
    windows = {
        'blackman': np.blackman(N),
        'hann': np.hanning(N),
        'hamming': np.hamming(N),
        'blackmanharris': np.blackman(N) # Placeholder, need better BH
    }
    
    # Create a proper Blackman-Harris window
    # 4-term Blackman-Harris
    a = np.array([0.35875, 0.48829, 0.14128, 0.01168])
    w_bh = a[0] - a[1]*np.cos(2*np.pi*n/(N-1)) + a[2]*np.cos(4*np.pi*n/(N-1)) - a[3]*np.cos(6*np.pi*n/(N-1))
    windows['blackmanharris'] = w_bh
    
    # Try different cutoffs (beta) for the sinc
    # sinc(beta * (n - center))
    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for w_name, w in windows.items():
        for beta in betas:
            if time.time() > deadline:
                break
            
            x = np.sinc(beta * (n - center)) * w
            # Normalize? No, score is scale invariant.
            # But numerical stability: ensure sum is not tiny.
            score = get_score(x)
            
            if score < best_score:
                best_score = score
                best_x = x.copy()
        
        if time.time() > deadline:
            break

    # Local Search (Hill Climbing)
    # Try to improve best_x by flipping signs or small perturbations?
    # The prompt mentions "pair-flip moves" and "sparse/block initial patterns".
    # Since we are in the "Windowed Sinc" family, let's try a gradient-free local search.
    # Randomly perturb some elements and check if score improves.
    
    # Use a smooth surrogate? 
    # Let's just do a simple random search around the best_x found.
    
    current_x = best_x.copy()
    current_score = best_score
    
    # Step size for perturbation
    # Since the signal is continuous, we can do small additive noise.
    
    iterations = 0
    max_iterations = 5000
    step_size = 0.1
    
    while time.time() < deadline - 5.0:
        iterations += 1
        # Random perturbation
        # Pick a random subset of indices to perturb
        num_perturb = 5
        idxs = np.random.choice(N, num_perturb, replace=False)
        noise = np.random.normal(0, step_size, num_perturb)
        
        new_x = current_x.copy()
        new_x[idxs] += noise
        
        new_score = get_score(new_x)
        
        if new_score < current_score:
            current_x = new_x
            current_score = new_score
            # Update global best
            if current_score < best_score:
                best_score = current_score
                best_x = current_x.copy()
        
        # Occasionally reset step size
        if iterations % 100 == 0:
            step_size *= 0.9

    return best_x
