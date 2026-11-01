import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill

    # 1. Generate an M-Sequence of length 511 (2^9 - 1)
    # Primitive polynomial for degree 9: x^9 + x^4 + 1
    # Taps: 9 and 4 (0-indexed from LSB, or 1-indexed from MSB depending on convention)
    # Let's use a standard LFSR.
    def generate_m_sequence(m):
        # Find a primitive polynomial
        # For m=9, x^9 + x^4 + 1 is primitive.
        # Feedback taps (indices of 1s in poly, excluding x^m):
        # x^9 + x^4 + 1 -> taps at 4 and 0 (if we consider x^0 as input)
        # Let's use a known list.
        # m=9: 0b100010001 (x^9 + x^4 + 1)
        poly = 0b100010001 
        # LFSR state of m bits
        state = 1  # Non-zero initial state
        seq = []
        for _ in range(2**m - 1):
            seq.append(state & 1)
            # Feedback
            fb = ((state & 1) ^ ((state >> 4) & 1)) # Taps at 0 and 4? 
            # Wait, standard Fibonacci LFSR:
            # new_bit = XOR of bits at tap positions.
            # If poly is x^9 + x^4 + 1, taps are at 9, 4, 0.
            # Usually we shift in new bit.
            # Let's just use a reliable generator or hardcode a good sequence?
            # Hardcoding is risky if I make a mistake.
            # Let's use a simple recursive construction or just random?
            # Random init is weak. M-seq is strong.
            # Let's use a known M-sequence of length 511.
            # Actually, for the purpose of the search, a random init with *many* iterations might work, 
            # but M-seq is much better.
            # Let's generate it properly.
            pass

    # Correct M-Sequence Generation
    def gen_m_seq(m, poly_taps):
        # poly_taps: list of indices (0-based) where taps are, including 0?
        # Standard: x^m + sum x^i + 1.
        # For m=9, x^9 + x^4 + 1. Taps at 4 and 0 (relative to LSB shift).
        # State is m bits.
        state = 1
        seq = []
        for _ in range(2**m - 1):
            seq.append(state & 1)
            # Calculate feedback
            fb = 0
            for tap in poly_taps:
                if tap == 0:
                    fb ^= (state & 1)
                else:
                    fb ^= (state >> tap) & 1
            # Shift right, insert fb at MSB
            state = ((state >> 1) | (fb << (m - 1))) & ((1 << m) - 1)
        return np.array(seq, dtype=np.float64) * 2 - 1 # Map 0,1 to -1,1

    # Taps for x^9 + x^4 + 1: indices 4 and 0.
    m_seq_511 = gen_m_seq(9, [4, 0])
    
    # Truncate to N=400
    h = m_seq_511[:N].copy()
    
    # Function to compute autocorrelation
    def compute_acf(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # acf[0] is energy, acf[1:N] are lags 1..N-1
        # We only care about lags 1..N-1 for the metric
        return acf[1:N]

    def get_metric(acf):
        return np.max(np.abs(acf))

    # Initial metric
    acf = compute_acf(h)
    best_val = get_metric(acf)
    best_h = h.copy()
    
    # Local Search: Greedy Flip
    # We iterate through indices. If flipping reduces max |acf|, we flip.
    # To speed up, we can restart the scan if an improvement is found.
    
    k = np.arange(1, N)
    
    def local_search_greedy(h, max_passes=500):
        acf = compute_acf(h)
        current_val = get_metric(acf)
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 20.0:
                break
            improved = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 20.0:
                    return h, current_val
                
                # Calculate delta for flipping h[i]
                # delta[k] = -2 * h[i] * (h[i+k] + h[i-k])
                # We need to handle boundaries.
                # h is 0-indexed.
                # For lag k (1..N-1):
                # term1: h[i+k] if i+k < N
                # term2: h[i-k] if i-k >= 0
                
                # Vectorized calculation
                # Create index arrays for valid parts
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
                
                new_acf = acf + delta
                new_val = get_metric(new_acf)
                
                if new_val < current_val:
                    # Accept
                    h[i] *= -1.0
                    acf = new_acf
                    current_val = new_val
                    improved = True
                    # Restart loop to check other bits with new state?
                    # Or continue? Continuing is faster, restarting is more thorough.
                    # Let's continue, but the next pass will catch it.
                    # Actually, restarting the inner loop is often better for greedy.
                    # But for speed, let's just finish the pass.
                # else:
                #    # Revert (not needed since we didn't modify h)
                #    pass
        
        return h, current_val

    # Run Greedy Local Search
    h, val = local_search_greedy(h, max_passes=200)
    if val < best_val:
        best_val = val
        best_h = h.copy()

    # Simulated Annealing to escape local minima
    def simulated_annealing(h, max_iters=20000):
        acf = compute_acf(h)
        current_val = get_metric(acf)
        
        T = 2.0
        T_min = 0.1
        cooling = 0.9999
        
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            i = np.random.randint(N)
            
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
            
            new_acf = acf + delta
            new_val = get_metric(new_acf)
            
            delta_val = new_val - current_val
            
            if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
                h[i] *= -1.0
                acf = new_acf
                current_val = new_val
            else:
                # Revert
                pass # acf is not modified in place if we reject? 
                # Wait, I did `new_acf = acf + delta`. `acf` is unchanged.
                # So no revert needed for `acf`.
                # But `h` is not modified.
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return h, current_val

    # Run SA
    h, val = simulated_annealing(best_h, max_iters=50000)
    if val < best_val:
        best_val = val
        best_h = h.copy()
        
    # Final Polish
    h, val = local_search_greedy(best_h, max_passes=100)
    if val < best_val:
        best_val = val
        best_h = h.copy()

    return best_h
