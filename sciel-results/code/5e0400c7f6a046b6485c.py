# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses Logistic Map for initialization and Simulated Annealing for optimization.
    """
    N = 400
    N_pad = 1 << (3 * N - 1).bit_length()
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3 (max absolute value of triple autocorrelation).
        """
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def generate_logistic_signs(seed):
        """Generate signs from Logistic Map x_{n+1} = 4 x_n (1 - x_n)."""
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        return signs

    # 1. Generate diverse initial candidates
    seeds = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.pi % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        0.111111,
        0.222222,
        0.333333
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        signs = generate_logistic_signs(seed)
        bound = compute_c3_bound(signs)
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            
    # 2. Simulated Annealing optimization
    # Time limit: 100 seconds to be safe (total 120s)
    start_time = time.time()
    time_limit = 100.0
    
    signs = best_signs
    current_bound = best_bound
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-5
    
    # Pre-allocate
    s_pad = np.zeros(N_pad)
    s_pad[:N] = signs
    
    # We need to update the FFT efficiently? 
    # FFT is O(N log N). N_pad=2048. 
    # 100 seconds / (2048 * 11 * 1e-9) ~ huge number of evals possible.
    # Let's just loop until time limit.
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Randomly select an index to flip
        i = np.random.randint(N)
        
        # Flip
        signs[i] *= -1.0
        s_pad[i] = signs[i]
        
        # Compute new bound
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        new_bound = np.max(np.abs(C3))
        
        # Acceptance criterion
        delta = new_bound - current_bound
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_bound = new_bound
        else:
            # Reject
            signs[i] *= -1.0
            s_pad[i] = signs[i]
            
        # Cool down
        T *= cooling_rate
        
    return signs
# EVOLVE-BLOCK-END
