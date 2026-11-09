import numpy as np
import time

N = 400

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    to minimize the peak autocorrelation sidelobe.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # FFT size for linear autocorrelation
    fft_size = 1 << (2 * N - 1).bit_length()
    
    # Pre-allocate zero-padded array for speed
    xp_template = np.zeros(fft_size, dtype=np.float64)
    
    def peak_sidelobe(seq):
        # Copy sequence into zero-padded array
        xp = xp_template.copy()
        xp[:N] = seq
        # FFT
        X = np.fft.fft(xp)
        # Power spectrum
        P = X * np.conj(X)
        # Inverse FFT to get autocorrelation
        R = np.fft.ifft(P).real
        # R[0] is N. R[1:N] are positive lags.
        # We want max absolute value of R[1:N]
        return np.max(np.abs(R[1:N]))
    
    def greedy_polish(x, deadline):
        """
        Perform greedy local search with single-bit flips.
        Repeats passes until no improvement or deadline reached.
        """
        current_cost = peak_sidelobe(x)
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(N):
                if time.time() >= deadline:
                    return x, current_cost
                x[i] *= -1
                new_cost = peak_sidelobe(x)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    x[i] *= -1
        return x, current_cost
    
    best_x = None
    best_cost = float('inf')
    
    # Phase 1: Iterated Local Search (ILS) with simulated annealing
    # We use multiple restarts with perturbation.
    
    # Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.9995
    min_T = 0.01
    
    # Block sizes for perturbation
    block_sizes = [1, 2, 4, 8, 16, 32]
    
    # Run ILS
    while time.time() < time_limit - 10:
        # Initialize with random sequence
        x = np.random.choice([-1, 1], size=N)
        current_cost = peak_sidelobe(x)
        
        T = T_start
        
        # Simulated Annealing loop
        while time.time() < time_limit - 10:
            if T < min_T:
                break
            
            # Choose move type: mostly single flips, some block flips
            r = np.random.random()
            if r < 0.7:
                # Single bit flip
                idx = np.random.randint(N)
                x[idx] *= -1
                new_cost = peak_sidelobe(x)
                delta = new_cost - current_cost
                
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    x[idx] *= -1  # Revert
            else:
                # Block flip
                block_len = np.random.choice(block_sizes)
                if block_len >= N:
                    block_len = N - 1
                start_idx = np.random.randint(0, N - block_len + 1)
                x[start_idx:start_idx + block_len] *= -1
                new_cost = peak_sidelobe(x)
                delta = new_cost - current_cost
                
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    x[start_idx:start_idx + block_len] *= -1  # Revert
            
            T *= cooling_rate
        
        # After SA, perform greedy polish
        x, current_cost = greedy_polish(x, time_limit - 5)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
    
    # Final greedy polish on the best found so far
    if best_x is not None:
        best_x, best_cost = greedy_polish(best_x, time_limit)
    
    if best_x is None:
        best_x = np.random.choice([-1, 1], size=N)
    
    return best_x
