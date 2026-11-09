"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 145 seconds (safety margin for 150s limit)
    start_time = time.time()
    time_limit = 145.0
    
    rng = np.random.default_rng(42)
    
    # Initialize with Thue-Morse sequence
    a = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    
    # Precompute FFT of all possible single-flip candidates for O(1) score updates
    # Actually, let's use the efficient update formula for autocorrelation sum of squares.
    # R_k = sum_i a_i a_{i+k}. Score = sum_{k!=0} R_k^2.
    # Let S = sum_{k!=0} R_k^2.
    # When we flip a_i to -a_i:
    # R_k changes for all k.
    # This is complex. Let's stick to FFT but optimize the loop.
    
    # Alternative: Use the fact that sum_{k} R_k^2 = sum_{i,j} (sum_k a_{i+k} a_{j+k})^2 ... no.
    # sum_{k} R_k^2 = sum_{k} (sum_i a_i a_{i+k})^2 = sum_{i,j} a_i a_j (sum_k a_{i+k} a_{j+k})
    # This is getting complicated. Let's just use FFT but make it fast.
    
    def compute_score(a):
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        return np.sum(R[1:]**2)
    
    current_a = a.copy()
    current_score = compute_score(current_a)
    best_a = current_a.copy()
    best_score = current_score
    
    # Simulated Annealing with Hybrid Moves
    T_start = 50.0
    T_end = 0.1
    num_iterations = 2000000  # Will be limited by time
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        T = T_start * (T_end / T_start) ** (i / num_iterations)
        
        # Hybrid move: 80% flip, 10% swap, 10% adjacent swap
        move_type = rng.random()
        
        if move_type < 0.8:
            # Flip one bit
            idx = rng.integers(0, N)
            candidate_a = current_a.copy()
            candidate_a[idx] *= -1.0
        elif move_type < 0.9:
            # Swap two bits
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            candidate_a = current_a.copy()
            candidate_a[idx1], candidate_a[idx2] = candidate_a[idx2], candidate_a[idx1]
        else:
            # Adjacent swap
            idx1 = rng.integers(0, N-1)
            idx2 = idx1 + 1
            candidate_a = current_a.copy()
            candidate_a[idx1], candidate_a[idx2] = candidate_a[idx2], candidate_a[idx1]
        
        candidate_score = compute_score(candidate_a)
        delta = candidate_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_a = candidate_a
            current_score = candidate_score
            
            if current_score < best_score:
                best_score = current_score
                best_a = current_a.copy()
                
    # Final polish: Local search with only flips, no temperature (greedy)
    # Iterate until no improvement or time runs out
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx in range(N):
            if (time.time() - start_time) > time_limit:
                break
            candidate_a = current_a.copy()
            candidate_a[idx] *= -1.0
            candidate_score = compute_score(candidate_a)
            if candidate_score < current_score:
                current_a = candidate_a
                current_score = candidate_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_a = current_a.copy()
                
    return best_a
# EVOLVE-BLOCK-END
