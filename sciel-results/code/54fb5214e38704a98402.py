import numpy as np
import time

N = 400
N_FFT = 1024  # Next power of 2 >= 2*N - 1

def _calc_max_autoconv(x):
    """
    Calculate the maximum absolute value of the aperiodic autocorrelation.
    x: numpy array of length N with values 1 or -1.
    """
    # Zero-pad to N_FFT
    # np.fft.fft expects a complex or real array. 
    # We compute the power spectrum |X|^2 and inverse FFT.
    # The result is the circular autocorrelation of the padded signal.
    # The first N-1 values (lags 1 to N-1) correspond to the aperiodic autocorrelation.
    
    X = np.fft.fft(x, n=N_FFT)
    P = X * np.conj(X)
    r = np.fft.ifft(P).real
    
    # r[0] is the energy N.
    # r[1:N] are the autocorrelation values for lags 1 to N-1.
    # Note: Due to floating point precision, we take the absolute value.
    # We want the max absolute value for lags 1 to N-1.
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = 115.0  # Keep 5s margin for safety
    
    # Helper to generate a random sequence
    def random_seq():
        return np.random.choice([-1, 1], N)
    
    # Initialize best solution
    best_x = random_seq()
    best_val = _calc_max_autoconv(best_x)
    
    # Current solution for search
    current_x = best_x.copy()
    current_val = best_val
    
    # Simulated Annealing / Local Search parameters
    # We use a hybrid approach:
    # 1. Fast local search (steepest descent / random walk)
    # 2. Occasional restarts if stuck
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    no_improve_count = 0
    patience = 500  # Restart if no improvement for 500 steps
    
    while time.time() - start_time < deadline:
        # Determine move
        # Mix of single bit flips and small block flips
        # Block flips can escape local minima faster
        if np.random.rand() < 0.7:
            # Single bit flip
            i = np.random.randint(N)
            current_x[i] *= -1
        else:
            # Block flip
            L = np.random.randint(2, 8)  # Block length 2 to 7
            if L <= N:
                i = np.random.randint(0, N - L + 1)
                current_x[i:i+L] *= -1
            else:
                # Fallback to single bit if L > N (unlikely)
                i = np.random.randint(N)
                current_x[i] *= -1
        
        # Evaluate
        new_val = _calc_max_autoconv(current_x)
        
        # Acceptance Criterion
        if new_val < current_val:
            # Always accept improvement
            current_val = new_val
            no_improve_count = 0
            
            if new_val < best_val:
                best_val = new_val
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            delta = new_val - current_val
            # Probability of accepting worse solution
            if T > T_min:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_val = new_val
                    no_improve_count += 1
                else:
                    # Reject move, revert
                    if np.random.rand() < 0.7:
                        i = np.random.randint(N)
                        current_x[i] *= -1
                    else:
                        L = np.random.randint(2, 8)
                        if L <= N:
                            i = np.random.randint(0, N - L + 1)
                            current_x[i:i+L] *= -1
                        else:
                            i = np.random.randint(N)
                            current_x[i] *= -1
            else:
                # If T is very low, just revert
                if np.random.rand() < 0.7:
                    i = np.random.randint(N)
                    current_x[i] *= -1
                else:
                    L = np.random.randint(2, 8)
                    if L <= N:
                        i = np.random.randint(0, N - L + 1)
                        current_x[i:i+L] *= -1
                    else:
                        i = np.random.randint(N)
                        current_x[i] *= -1
        
        # Cooling
        T *= cooling_rate
        
        # Restart logic
        if no_improve_count > patience:
            # Restart with a new random sequence
            current_x = random_seq()
            current_val = _calc_max_autoconv(current_x)
            no_improve_count = 0
            T = 2.0  # Reset temperature
    
    return best_x

# EVOLVE-BLOCK-END
