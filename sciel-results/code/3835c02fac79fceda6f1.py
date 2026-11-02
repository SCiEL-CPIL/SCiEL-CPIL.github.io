import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_f = None
    best_score = float('inf')
    
    # Try a few random initializations
    for seed in range(10):
        if time.time() > deadline: break
        
        rng = np.random.default_rng(seed)
        f = rng.choice([-1, 1], size=N)
        
        # Local search
        f = local_search(f, deadline)
        
        # Evaluate
        score = evaluate(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
    # If no search was done or failed, return seed
    if best_f is None:
        return np.ones(N)
        
    return best_f

def evaluate(f):
    # Compute autocorrelation
    # f is real, length N
    # Use FFT
    F = np.fft.fft(f, n=2*N) # Zero pad to avoid circular conv artifacts? 
    # Actually, linear autocorrelation is IFFT(FFT(f) * conj(FFT(f)))
    # But we need to handle the linear vs circular.
    # For linear autocorrelation of length N, we can use:
    F = np.fft.fft(f, n=2*N)
    power = F * np.conj(F)
    R = np.fft.ifft(power).real
    # The first N elements of R (shifted?)
    # Standard numpy ifft of power gives circular autocorrelation.
    # To get linear, we need to shift.
    # R_lin[k] corresponds to lag k.
    # The IFFT result is circular.
    # For zero-padded FFT, the first N elements (after shifting) are the linear autocorrelation?
    # Actually, if we zero pad to 2N, the circular autocorrelation of length 2N contains the linear autocorrelation.
    # R_circ[0] = sum f^2.
    # R_circ[1] = sum f_i f_{i+1}.
    # ...
    # We want sum R[k]^2 for k=0 to N-1.
    
    # Let's just compute R[0:N]
    # Note: R is symmetric.
    metric = np.sum(R[:N]**2)
    return metric

def local_search(f, deadline):
    f = f.copy()
    current_metric = evaluate(f)
    
    while time.time() < deadline - 5:
        improved = False
        # Random order of indices to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline - 5: break
            
            # Flip i
            f[i] *= -1
            
            new_metric = evaluate(f)
            
            if new_metric < current_metric:
                current_metric = new_metric
                improved = True
            else:
                f[i] *= -1 # Revert
        
        if not improved:
            # Try swapping pairs? Or just break?
            # Let's try a few random restarts or pair swaps
            # For simplicity, break if no single bit flip works
            break
            
    return f
