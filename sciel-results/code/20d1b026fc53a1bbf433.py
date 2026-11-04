import numpy as np
import time

N = 400

def construct():
    """
    Construct a signed step function (400 intervals) to minimize the upper bound on C3.
    We assume the metric is the L1 norm of the autocorrelation sidelobes (sum of absolute values).
    We use an m-sequence fragment as a starting point and refine it with Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    rng = np.random.RandomState(42)
    
    def get_l1_sidelobe(h):
        # Compute linear autocorrelation
        # h is +/- 1
        # r(tau) = sum h[i] h[i+tau]
        # Use FFT
        H = np.fft.fft(h, n=2*N)
        R = np.fft.ifft(H * np.conj(H)).real
        # R[0] is r(0) = N
        # R[1:N] are r(1) to r(N-1)
        # R[N+1:] are r(-(N-1)) to r(-1)
        # For real h, r(-tau) = r(tau)
        # We sum |r(tau)| for tau != 0
        # Due to symmetry, sum = 2 * sum_{tau=1}^{N-1} |r(tau)|
        sidelobes = np.abs(R[1:N])
        return 2.0 * np.sum(sidelobes)

    def get_peak_sidelobe(h):
        H = np.fft.fft(h, n=2*N)
        R = np.fft.ifft(H * np.conj(H)).real
        sidelobes = np.abs(R[1:N])
        return np.max(sidelobes)

    # 1. Generate an m-sequence of length 511 (2^9 - 1)
    # Primitive polynomial for LFSR: x^9 + x^4 + 1 (0x021)
    # Or use a known one.
    # Let's use a simple LFSR implementation.
    def generate_m_sequence(length, poly, init_state):
        # poly is integer, e.g., 0x021 for x^9+x^4+1
        # init_state is 9-bit integer
        seq = []
        state = init_state
        # We need to generate 'length' bits
        # LFSR feedback: bit = (state >> 0) ^ (state >> 4)
        # Actually, depends on convention.
        # Let's use numpy for speed if possible, or just loop.
        # Loop is fine for 511 bits.
        for _ in range(length):
            # Get LSB
            bit = state & 1
            seq.append(bit)
            # Feedback
            # x^9 + x^4 + 1 -> new_bit = bit0 ^ bit4
            feedback = (state & 1) ^ ((state >> 4) & 1)
            state = (state >> 1) | (feedback << 8)
        
        # Map 0 -> -1, 1 -> 1
        h = np.array(seq, dtype=np.float64)
        h[h == 0] = -1.0
        h[h == 1] = 1.0
        return h

    # Generate m-sequence of length 511
    m_seq = generate_m_sequence(511, 0x021, 1)
    # Take first 400
    h0 = m_seq[:400].copy()
    
    # Also try a random start
    h_rand = rng.choice([-1.0, 1.0], size=N)
    
    # Evaluate both
    score_m = get_l1_sidelobe(h0)
    score_r = get_l1_sidelobe(h_rand)
    
    best_h = h0 if score_m < score_r else h_rand
    best_score = min(score_m, score_r)
    
    # Simulated Annealing
    # Temperature schedule
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.01
    
    # We will use L1 norm as the primary objective
    # But we might also check peak to avoid pathological cases
    
    while time.time() < deadline - 5.0 and T > min_T:
        # Perform a few steps per temperature
        steps = 1000
        for _ in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            # Random flip
            idx = rng.randint(0, N)
            old_val = best_h[idx]
            best_h[idx] *= -1.0
            
            new_score = get_l1_sidelobe(best_h)
            delta = new_score - best_score
            
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                best_score = new_score
            else:
                best_h[idx] = old_val
        
        T *= cooling_rate
        
        # Periodically check peak to ensure we are not drifting to high peak
        # If peak is too high, we might prefer a sequence with lower peak even if L1 is slightly higher?
        # But L1 is our assumed metric.
        
    return best_h
