"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h).
    
    overlap(k) = (2/N) * sum_i h[i] * (1 - h[(i+k) % N])
    We return max over k=1..N-1 of overlap(k).
    """
    n = len(h)
    # Use FFT for circular cross-correlation
    # autocorr(k) = sum_i h[i] * h[(i+k) % n]
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # Round to handle floating point
    autocorr = np.round(autocorr)
    # overlap(k) = (2/n) * (sum(h) - autocorr(k)) = (2/n) * (n/2 - autocorr(k)) = 1 - 2*autocorr(k)/n
    # For k=0: overlap(0) = 1 - 2*autocorr(0)/n, but we only care about k>=1
    # autocorr[0] = sum(h^2)
    # For k=1..n-1:
    overlaps = 1.0 - 2.0 * autocorr[1:] / n
    return np.max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initialize with random binary sequence with exactly N/2 ones
    h = np.zeros(N, dtype=np.float64)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    
    # Simulated annealing with swap moves
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    current_h = best_h.copy()
    current_score = best_score
    
    iteration = 0
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        iteration += 1
        
        # Pick two random positions to swap
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
        
        # Only swap if h[i] != h[j]
        if current_h[i] == current_h[j]:
            continue
        
        # Swap
        current_h[i], current_h[j] = current_h[j], current_h[i]
        new_score = _max_overlap(current_h)
        
        delta = new_score - current_score
        if delta <= 0 or np.random.random() < np.exp(-delta / (T + 1e-10)):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Reject: undo swap
            current_h[i], current_h[j] = current_h[j], current_h[i]
            no_improve += 1
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # If stuck too long, restart with perturbation
        if no_improve > max_no_improve:
            current_h = best_h.copy()
            current_score = _max_overlap(current_h)
            # Make several random swaps
            for _ in range(10):
                a = np.random.randint(N)
                b = np.random.randint(N)
                if current_h[a] != current_h[b]:
                    current_h[a], current_h[b] = current_h[b], current_h[a]
            current_score = _max_overlap(current_h)
            T = 1.0
            no_improve = 0
    
    # Final local search polish: try all single swaps
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, N):
                if best_h[i] == best_h[j]:
                    continue
                # Try swap
                best_h[i], best_h[j] = best_h[j], best_h[i]
                new_score = _max_overlap(best_h)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Undo
                    best_h[i], best_h[j] = best_h[j], best_h[i]
    
    return best_h
# EVOLVE-BLOCK-END
