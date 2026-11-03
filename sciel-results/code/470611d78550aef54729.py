import numpy as np
import time

N = 400

def compute_psl(seq):
    # seq is an array of +/- 1
    # Compute autocorrelation
    # Using FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.fft.fft(seq, n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real
    # The circular autocorrelation is in acf
    # The linear autocorrelation lags are:
    # acf[0] is lag 0
    # acf[1:N] are lags 1..N-1
    # acf[n_fft-N+1 : n_fft] are lags -(N-1)..-1
    # We need max absolute value of non-zero lags
    # lags 1 to N-1: acf[1:N]
    # lags -1 to -(N-1): acf[n_fft-N+1 : n_fft]
    side1 = np.abs(acf[1:N])
    side2 = np.abs(acf[n_fft-N+1 : n_fft])
    return np.max(np.concatenate([side1, side2]))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with random sequence
    best_seq = np.random.choice([-1, 1], N)
    best_psl = compute_psl(best_seq)
    
    # Local Search: Hill Climbing with random restarts if stuck
    current_seq = best_seq.copy()
    current_psl = best_psl
    
    # Precompute indices for faster access? Not necessary for N=400.
    
    # Simple Hill Climbing
    # Try flipping each bit and see if PSL decreases.
    # To speed up, we can check a random subset or use a more sophisticated neighbor generation.
    # Given N=400, computing PSL takes ~0.1ms. We can do thousands of evaluations per second.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if not improved:
            # Random restart or perturbation
            # Flip a few random bits
            for _ in range(10):
                idx = np.random.randint(N)
                current_seq[idx] *= -1
            current_psl = compute_psl(current_seq)
            if current_psl < best_psl:
                best_psl = current_psl
                best_seq = current_seq.copy()
            improved = True
            no_improve_count = 0
            continue
            
        # Try flipping random bits
        # Sample a batch of candidates
        improved = False
        for _ in range(100): # 100 trials per "iteration"
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            orig_val = current_seq[idx]
            current_seq[idx] = -orig_val
            new_psl = compute_psl(current_seq)
            
            if new_psl < current_psl:
                current_psl = new_psl
                if new_psl < best_psl:
                    best_psl = new_psl
                    best_seq = current_seq.copy()
                improved = True
                no_improve_count = 0
            else:
                current_seq[idx] = orig_val
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                improved = False
                break
                
    return best_seq
