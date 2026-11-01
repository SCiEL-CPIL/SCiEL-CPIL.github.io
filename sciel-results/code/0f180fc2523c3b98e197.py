import numpy as np
import time

N = 400

def _autocorr_fft(x):
    """Compute linear autocorrelation of x using FFT."""
    n_fft = 2 * len(x)
    X = np.fft.fft(x, n=n_fft)
    P = X * np.conj(X)
    R = np.fft.ifft(P).real
    return R[:len(x)]

def _psl(x):
    """Peak sidelobe level: max |R(tau)| for tau != 0."""
    R = _autocorr_fft(x)
    return np.max(np.abs(R[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Simulated annealing on binary ±1 sequence
    # Objective: minimize max |R(tau)| for tau=1..N-1
    
    # Initialize with random signs
    best_x = np.random.choice([-1.0, 1.0], N)
    best_psl = _psl(best_x)
    
    # Annealing parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.9995
    
    current_x = best_x.copy()
    current_psl = best_psl
    T = T_start
    
    # Track stagnation for restarts
    stagnation_count = 0
    max_stagnation = 5000
    
    while time.time() < deadline:
        # Pick a random position to flip
        j = np.random.randint(N)
        new_x = current_x.copy()
        new_x[j] *= -1.0
        
        new_psl = _psl(new_x)
        delta = new_psl - current_psl
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_x = new_x
            current_psl = new_psl
            
            if current_psl < best_psl:
                best_x = current_x.copy()
                best_psl = current_psl
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # Cool down
        T = max(T * cooling_rate, T_end)
        
        # If stuck for too long, restart with a new random initialization
        if stagnation_count > max_stagnation:
            current_x = np.random.choice([-1.0, 1.0], N)
            current_psl = _psl(current_x)
            T = T_start
            stagnation_count = 0
            
            # Keep the best found so far
            if current_psl > best_psl:
                current_x = best_x.copy()
                current_psl = best_psl
    
    return best_x
