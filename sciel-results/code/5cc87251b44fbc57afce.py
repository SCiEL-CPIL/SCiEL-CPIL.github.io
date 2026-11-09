"""Program for the third autocorrelation inequality task.
Uses a binary sequence derived from a maximal-length LFSR (m-sequence)
to minimize the peak sidelobe autocorrelation, which provides a strong
upper bound on C3.
"""
import numpy as np
import time

N = 400

def _compute_score(bits):
    """
    Compute the upper bound on C3 given a binary sequence of signs (+1/-1).
    The score is the maximum absolute value of the aperiodic autocorrelation
    at non-zero lags, divided by N (or a similar normalization).
    For the task, we want to minimize this value.
    The 'score' returned by construct() is the upper bound on C3.
    """
    # Convert bits (0/1) to signs (-1/+1)
    signs = 1 - 2 * bits
    # Compute aperiodic autocorrelation
    # R[k] = sum_{n=0}^{N-1-k} signs[n] * signs[n+k] for k=0..N-1
    # We want to minimize max_{k != 0} |R[k]|
    
    # Efficient computation using FFT or direct convolution
    # Using np.correlate is O(N^2) which is 160,000 ops, very fast.
    # But let's use FFT for safety and speed if N were larger.
    # For N=400, direct is fine.
    
    # autocorr = np.correlate(signs, signs, mode='full')
    # The aperiodic autocorrelation is in the middle part.
    # Let's just compute the lags 1 to N-1.
    
    # Actually, the standard definition for this type of problem often involves
    # the periodic autocorrelation or the peak sidelobe of the aperiodic one.
    # Let's assume the score is max_{k=1}^{N-1} |R_k| where R_k is the aperiodic autocorrelation.
    
    # Let's compute all aperiodic lags
    # R_k = sum_{i=0}^{N-1-k} s_i s_{i+k}
    
    # Using FFT for speed
    f = np.fft.fft(signs, 2 * N)
    autocorr_full = np.fft.ifft(f * np.conj(f)).real
    # The aperiodic autocorrelation lags 0 to N-1 are in autocorr_full[0:N]
    # Lags -(N-1) to -1 are in autocorr_full[N+1:]
    # We care about lags 1 to N-1.
    # autocorr_full[1:N] gives R_1 ... R_{N-1}
    
    sidelobes = np.abs(autocorr_full[1:N])
    peak_sidelobe = np.max(sidelobes)
    
    # The task says "score is the upper bound on C3".
    # In many contexts, C3 is related to the normalized autocorrelation.
    # Let's assume the score is peak_sidelobe / N.
    # The seed program returns np.ones(N), which has peak_sidelobe = N (for lag 1, it's N-1, etc).
    # Wait, if signs are all +1, R_k = N - k. Max is N-1.
    # Score = (N-1)/N = 399/400 = 0.9975.
    # The seed score is 1.0. This implies the metric might be different.
    
    # Let's re-read: "score is the upper bound on C3".
    # "search score is seed/value, so 1.0 equals the seed program".
    # If seed returns ones, and score is 1.0, then value(seed) = seed_value.
    # If the metric is raw value, and seed value is V_seed, then score = V_seed / V_candidate.
    # If candidate is better (lower value), score > 1.
    
    # Let's assume the value to minimize is the peak sidelobe autocorrelation.
    # For all-ones, peak sidelobe is N-1.
    # So if I find a sequence with peak sidelobe P, the score would be (N-1)/P.
    # An m-sequence has peak sidelobe roughly sqrt(N) or similar small constant?
    # For N=400, an m-sequence of length 511 truncated to 400?
    # Or just use a random sequence? Random has peak sidelobe ~ sqrt(N) * log(N).
    # sqrt(400) = 20.
    # So score could be 399/20 ~ 20.
    
    # Let's implement the m-sequence generation.
    # We need a primitive polynomial of degree m such that 2^m - 1 >= 400.
    # m=9 -> 511.
    # Primitive polynomial for m=9: x^9 + x^4 + 1 (0b100010001 = 273)
    # Let's verify or just use a known one.
    # x^9 + x^4 + 1 is primitive.
    
    # Generate m-sequence
    # LFSR state: 9 bits.
    # Polynomial: x^9 + x^4 + 1.
    # Feedback tap at x^4.
    
    # Let's just generate a random binary sequence first to see if it's better than ones.
    # But the directive says "LFSR with a primitive polynomial".
    
    # Let's implement LFSR.
    
    def generate_m_sequence(n_bits, poly, seed_state):
        """
        Generate an m-sequence using an LFSR.
        poly: integer representing the polynomial (e.g., 273 for x^9+x^4+1)
        seed_state: initial state (non-zero)
        """
        m = poly.bit_length() - 1
        # Extract taps
        # For x^m + ... + 1, the feedback is sum of bits at specific positions.
        # Standard Fibonacci LFSR:
        # new_bit = XOR of bits at positions corresponding to terms in poly (excluding x^m)
        
        seq = []
        state = seed_state
        for _ in range(n_bits):
            # Output bit
            seq.append(state & 1)
            # Feedback bit
            # The polynomial is x^m + c_{m-1}x^{m-1} + ... + c_0
            # Feedback = sum c_i * state_bit_i
            # Here state is a bit string.
            # Let's use the standard Galois or Fibonacci form.
            # Fibonacci:
            # new_bit = 0
            # for i in range(m):
            #     if (poly >> i) & 1:
            #         new_bit ^= (state >> i) & 1
            # But we must exclude the x^m term in the feedback sum?
            # Actually, the recurrence is s_{n+m} = sum_{i=0}^{m-1} c_i s_{n+i}
            # So feedback = XOR of (state >> i) & 1 for all i where (poly >> i) & 1 is 1, for i < m.
            
            fb = 0
            for i in range(m):
                if (poly >> i) & 1:
                    fb ^= (state >> i) & 1
            
            state = (state >> 1) | (fb << (m - 1))
        
        return np.array(seq)

    # Poly for m=9: x^9 + x^4 + 1
    # Binary: 1 0001 0001 -> 100010001
    # Value: 256 + 16 + 1 = 273
    poly = 273
    m = 9
    
    # Generate 400 bits
    # We can try different seeds to find the best one, but m-sequences are all equivalent up to shift.
    # So the peak sidelobe of the first 400 bits depends on the shift.
    # We can try all shifts (or a subset) to find the one with minimal peak sidelobe.
    
    # Generate a full period of 511 bits
    full_seq = generate_m_sequence(511, poly, 1)
    
    # Try all 511 shifts
    best_peak = np.inf
    best_shift = 0
    
    # Precompute FFT of the full sequence? No, we need the truncated one.
    # N=400 is small, so we can just loop.
    
    for shift in range(511):
        # Take 400 bits starting from shift
        # Wrap around? The sequence is periodic.
        bits = full_seq[(np.arange(400) + shift) % 511]
        
        signs = 1 - 2 * bits
        f = np.fft.fft(signs, 800)
        autocorr_full = np.fft.ifft(f * np.conj(f)).real
        sidelobes = np.abs(autocorr_full[1:400])
        peak = np.max(sidelobes)
        
        if peak < best_peak:
            best_peak = peak
            best_shift = shift
            best_bits = bits.copy()
            
    return best_peak / N, best_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial guess: m-sequence
    peak, bits = _compute_score(None) # This will do the search inside
    
    # Check time
    if time.time() > deadline:
        return bits
    
    # The score returned is the "upper bound on C3".
    # If the metric is peak_sidelobe / N, then we return that.
    # But wait, the seed returns np.ones(N).
    # If I return bits, the evaluator computes the score.
    # The evaluator measures the "raw value".
    # So I just need to return the sequence that minimizes the raw value.
    
    return bits

# Wait, _compute_score returns peak/N and bits.
# I should just return bits.

# Let's refine _compute_score to just return the best bits found.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate m-sequence
    poly = 273
    m = 9
    full_seq = generate_m_sequence(511, poly, 1)
    
    best_peak = np.inf
    best_bits = None
    
    for shift in range(511):
        if time.time() > deadline:
            break
            
        bits = full_seq[(np.arange(400) + shift) % 511]
        
        signs = 1 - 2 * bits
        f = np.fft.fft(signs, 800)
        autocorr_full = np.fft.ifft(f * np.conj(f)).real
        sidelobes = np.abs(autocorr_full[1:400])
        peak = np.max(sidelobes)
        
        if peak < best_peak:
            best_peak = peak
            best_bits = bits.copy()
            
    if best_bits is None:
        # Fallback to random or ones
        best_bits = np.random.randint(0, 2, N)
        
    return best_bits

# Helper function needs to be defined before construct or inside.
# Let's put it inside or at module level.

def generate_m_sequence(n_bits, poly, seed_state):
    m = poly.bit_length() - 1
    seq = []
    state = seed_state
    for _ in range(n_bits):
        seq.append(state & 1)
        fb = 0
        for i in range(m):
            if (poly >> i) & 1:
                fb ^= (state >> i) & 1
        state = (state >> 1) | (fb << (m - 1))
    return np.array(seq)
