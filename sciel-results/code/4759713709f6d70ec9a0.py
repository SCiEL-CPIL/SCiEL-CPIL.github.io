import numpy as np
import time

N = 400

def _score(x):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation.
    """
    # Use FFT to compute autocorrelation efficiently
    # Pad to 2*N to avoid circular convolution artifacts
    fft_len = 2 * N
    X = np.fft.fft(x, n=fft_len)
    # Power spectrum
    P = np.conj(X) * X
    # Inverse FFT to get autocorrelation
    ac = np.real(np.fft.ifft(P))
    # Take the first N values (lags 0 to N-1)
    ac = ac[:N]
    return np.max(np.abs(ac))

def _generate_m_sequence(m, poly):
    """
    Generate an m-sequence of length 2^m - 1.
    poly is an integer representing the primitive polynomial (excluding x^m term).
    e.g., for x^9 + x^4 + 1, poly = (1<<4) | 1 = 17.
    """
    L = (1 << m) - 1
    state = 1  # Initial state 1
    seq = np.zeros(L, dtype=np.int8)
    for i in range(L):
        seq[i] = 1 if (state & 1) == 1 else -1
        # Feedback
        feedback = (state & 1) ^ ((state >> 4) & 1) # Example for x^9+x^4+1
        # Shift left
        state = (state << 1) & ((1 << m) - 1)
        state |= feedback
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialize with a segment of an m-sequence
    # m=9, L=511. Poly x^9 + x^4 + 1 is primitive.
    # Bits: 000010001 (17)
    # Let's use a standard primitive poly for m=9.
    # x^9 + x^4 + 1 is primitive.
    m_seq = _generate_m_sequence(9, 17)
    x = m_seq[:N].copy()
    
    best_x = x.copy()
    best_score = _score(x)
    
    # 2. Local Search
    # We try flipping bits one by one. If a flip improves the score, we keep it.
    # To avoid getting stuck, we randomize the order of indices in each pass.
    # If no improvement in a full pass, we perform a perturbation.
    
    while time.time() < deadline:
        improved = False
        # Randomize the order of indices to explore different local minima
        indices = np.random.permutation(N)
        
        for i in indices:
            # Flip bit i
            x[i] *= -1
            
            # Calculate new score
            current_score = _score(x)
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
                improved = True
            else:
                # Revert if not improved
                x[i] *= -1
        
        if not improved:
            # Perturbation: flip a small number of random bits to escape local minimum
            num_flips = 10
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            x[flip_indices] *= -1
            # Update best if perturbation accidentally improved (unlikely but possible)
            s = _score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()

    return best_x
