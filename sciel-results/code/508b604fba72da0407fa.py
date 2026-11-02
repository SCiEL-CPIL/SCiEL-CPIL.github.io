import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s out of 120s/150s
    
    # Initialize with random +/- 1 sequence
    # Or start with all ones and perturb?
    # The insight says +/-1 local search improved score.
    # Let's try starting with a random +/-1 sequence and optimizing.
    
    x = np.random.choice([1, -1], size=N)
    
    def score(x):
        # Calculate autoconvolution
        # Using FFT for speed
        X = np.fft.fft(x, n=2*N)
        C = np.fft.ifft(X * X).real
        # The correlation is in the first N elements (and symmetric)
        # C[k] corresponds to sum x[i] x[i+k]
        # Actually, ifft of |FFT|^2 gives circular autocorrelation.
        # For linear, we need zero-padding.
        # n=2*N is sufficient for linear autocorrelation of length N.
        
        # Max absolute value
        max_c = np.max(np.abs(C))
        
        # Squared integral (sum)
        s = np.sum(x)
        s2 = s * s
        
        if s2 == 0:
            return float('inf')
            
        return max_c / s2

    current_x = x.copy()
    current_score = score(current_x)
    
    # Local search: flip random indices
    # To allow 'S' to be large, we might want to bias towards +1?
    # But random start is good for exploring.
    
    steps = 0
    while time.time() < deadline:
        # Flip a random set of bits? Or single bit?
        # Single bit flip changes S by +/- 2.
        # If S is large, flipping to -1 reduces S.
        
        idx = np.random.randint(N)
        new_x = current_x.copy()
        new_x[idx] *= -1
        
        new_score = score(new_x)
        
        if new_score < current_score:
            current_x = new_x
            current_score = new_score
            
        steps += 1
        # Throttle to avoid infinite loop in tight python
        if steps % 1000 == 0:
            time.sleep(0)
            
    return current_x
# EVOLVE-BLOCK-END
