"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill

    def compute_initial_autocorr(h):
        """Compute full autocorrelation R where R[k] is lag k. R[0] is energy."""
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        R = np.zeros(N)
        R[0] = acf[0]
        R[1:N] = acf[1:N]
        return R

    def get_max_autocorr(R):
        """Get the maximum absolute autocorrelation for lags 1..N-1."""
        return np.max(np.abs(R[1:N]))

    def local_search(h, max_passes=200):
        """
        Perform local search using incremental O(N) updates.
        """
        h = h.copy()
        R = compute_initial_autocorr(h)
        current_val = get_max_autocorr(R)
        
        k = np.arange(1, N)
        
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            if time.time() > deadline - 15.0:
                break
            improved = False
            passes += 1
            
            # Try flipping each bit in a deterministic or slightly randomized order
            # Randomizing order can help escape some local minima, but deterministic is faster
            order = np.arange(N)
            
            for i in order:
                if time.time() > deadline - 15.0:
                    return h, current_val
                
                # Calculate delta for flipping h[i]
                # Delta R[k] = -2 * h[i] * (h[i+k] + h[i-k])
                
                # term1: h[i+k]
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                if np.any(valid1):
                    term1[valid1] = h[idx1[valid1]]
                
                # term2: h[i-k]
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                if np.any(valid2):
                    term2[valid2] = h[idx2[valid2]]
                
                delta = -2.0 * h[i] * (term1 + term2)
                
                # Apply update
                R[1:N] += delta
                
                new_val = get_max_autocorr(R)
                
                if new_val < current_val:
                    # Accept flip
                    h[i] *= -1.0
                    current_val = new_val
                    improved = True
                else:
                    # Revert
                    R[1:N] -= delta
        
        return h, current_val

    def simulated_annealing(h, max_iters=50000):
        """
        Simulated Annealing to escape local minima.
        """
        h = h.copy()
        R = compute_initial_autocorr(h)
        current_val = get_max_autocorr(R)
        
        k = np.arange(1, N)
        
        T = 5.0 # Initial temperature
        T_min = 0.01
        cooling = 0.9999
        
        for _ in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            
            i = np.random.randint(N)
            
            # Calculate delta
            idx1 = i + k
            valid1 = idx1 < N
            term1 = np.zeros(N-1)
            if np.any(valid1):
                term1[valid1] = h[idx1[valid1]]
            
            idx2 = i - k
            valid2 = idx2 >= 0
            term2 = np.zeros(N-1)
            if np.any(valid2):
                term2[valid2] = h[idx2[valid2]]
            
            delta = -2.0 * h[i] * (term1 + term2)
            R[1:N] += delta
            new_val = get_max_autocorr(R)
            
            delta_val = new_val - current_val
            
            if delta_val < 0:
                # Better
                h[i] *= -1.0
                current_val = new_val
            else:
                # Worse, accept with probability exp(-delta_val / T)
                if np.random.rand() < np.exp(-delta_val / T):
                    h[i] *= -1.0
                    current_val = new_val
                else:
                    # Revert
                    R[1:N] -= delta
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return h, current_val

    rng = np.random.RandomState(42)
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Local Search on random restarts
    # Use fewer restarts but longer local search to exploit the fast incremental updates
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        h = rng.choice([1, -1], N).astype(np.float64)
        h, val = local_search(h, max_passes=100)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Phase 2: Simulated Annealing on the best found so far
    # SA with incremental updates is very fast, so we can run many iterations
    if best_h is not None and time.time() < deadline - 20.0:
        h_sa, val_sa = simulated_annealing(best_h, max_iters=100000)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
            
        # Phase 3: Final Local Search polish
        if time.time() < deadline - 5.0:
            h_final, val_final = local_search(best_h, max_passes=100)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final

    if best_h is None:
        # Fallback to random
        best_h = rng.choice([1, -1], N).astype(np.float64)
        
    return best_h
# EVOLVE-BLOCK-END
