"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 145 seconds (safety margin for 150s limit)
    start_time = time.time()
    time_limit = 145.0
    
    def compute_score(a):
        """Compute sum_{k!=0} R_k^2 where R_k = sum_i a_i a_{i+k}"""
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        return np.sum(R[1:]**2)
    
    def generate_m_sequence(n):
        """Generate a maximum length sequence of length n (n = 2^m - 1)."""
        # Find the smallest m such that 2^m - 1 >= n
        m = 1
        while (2**m - 1) < n:
            m += 1
        # Use a known primitive polynomial for m up to about 10
        # For m=9: x^9 + x^4 + 1 (primitive)
        # For m=10: x^10 + x^7 + 1 (primitive)
        primitive_polys = {
            9: (1 << 9) | (1 << 4) | 1,  # x^9 + x^4 + 1
            10: (1 << 10) | (1 << 7) | 1,  # x^10 + x^7 + 1
        }
        if m not in primitive_polys:
            # Fallback: use Thue-Morse
            return np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(n)])
        
        poly = primitive_polys[m]
        # Generate m-sequence using LFSR
        state = 1  # Non-zero initial state
        seq = []
        full_len = 2**m - 1
        for _ in range(full_len):
            seq.append(state)
            # LFSR step
            lsb = state & 1
            state = state >> 1
            if lsb:
                state ^= poly
        # Convert to ±1 and take first n elements
        seq = np.array(seq[:n], dtype=float)
        # Map to ±1: 0 -> +1, 1 -> -1 (or vice versa)
        # The actual values don't matter much, we'll optimize
        return np.where(seq > 0, -1.0, 1.0)
    
    def compute_score_fast_update(a, flip_idx):
        """Compute new score after flipping a[flip_idx], in O(N) time."""
        # R_k = sum_i a_i a_{i+k}
        # When we flip a[j] from a[j] to -a[j]:
        # R_k changes by: -2 * a[j] * sum_{i: i+k=j} a_i = -2 * a[j] * a[j-k] for the term where i=j-k
        # Wait, let me think more carefully.
        # R_k = sum_i a_i a_{i+k}
        # After flipping a[j] to -a[j]:
        # New R_k = sum_i a_i a_{i+k} - 2*a[j]*(a_{j-k} if 0<=j-k<N else 0) - 2*a[j]*(a_{j+k} if 0<=j+k<N else 0) + ... 
        # This is getting complex. Let's just use FFT for simplicity but optimize the search.
        pass
    
    # Generate multiple candidate initializations
    best_a = None
    best_score = float('inf')
    
    # Candidate 1: m-sequence
    m_seq = generate_m_sequence(N)
    s = compute_score(m_seq)
    if s < best_score:
        best_score = s
        best_a = m_seq.copy()
    
    # Candidate 2: Thue-Morse
    tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    s = compute_score(tm)
    if s < best_score:
        best_score = s
        best_a = tm.copy()
    
    # Candidate 3: Random
    rng = np.random.default_rng(42)
    rand = rng.choice([-1.0, 1.0], size=N)
    s = compute_score(rand)
    if s < best_score:
        best_score = s
        best_a = rand.copy()
    
    current_a = best_a.copy()
    current_score = best_score
    
    # Simulated Annealing with focus on flips
    T_start = 20.0
    T_end = 0.01
    num_iterations = 5000000  # Will be limited by time
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        T = T_start * (T_end / T_start) ** (i / num_iterations)
        
        # Flip one bit
        idx = rng.integers(0, N)
        candidate_a = current_a.copy()
        candidate_a[idx] *= -1.0
        
        candidate_score = compute_score(candidate_a)
        delta = candidate_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_a = candidate_a
            current_score = candidate_score
            
            if current_score < best_score:
                best_score = current_score
                best_a = current_a.copy()
    
    # Final polish: Greedy local search
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

