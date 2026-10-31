import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _autoconv_max_abs(f):
    """Compute max |autoconvolution| of f using FFT."""
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, n=size)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=size)
    return np.max(np.abs(ac[:2 * n - 1]))


def _squared_integral(f):
    """Compute (sum f)^2."""
    return float(np.sum(f)) ** 2


def _objective(f):
    """Compute max|autoconv| / (sum f)^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    return _autoconv_max_abs(f) / (S * S)


def _local_sign_flip(f, n_trials=2000):
    """
    Try flipping signs of random subsets to reduce the objective.
    Returns improved f or original if no improvement.
    """
    best_f = f.copy()
    best_obj = _objective(best_f)
    n = len(f)
    
    for _ in range(n_trials):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        # Random subset of indices to flip
        k = np.random.randint(1, n // 4 + 1)
        indices = np.random.choice(n, size=k, replace=False)
        candidate = best_f.copy()
        candidate[indices] *= -1
        
        obj = _objective(candidate)
        if obj < best_obj:
            best_f = candidate
            best_obj = obj
    
    return best_f, best_obj


def construct():
    """
    Returns a signed step function minimizing max|autoconv| / (sum f)^2.
    
    Strategy:
    1. Start with all-ones (achieves 1/N).
    2. Use structured patterns based on known low-autocorrelation sequences.
    3. Local sign-flip optimization.
    4. Return the best found.
    """
    best_f = np.ones(N)
    best_obj = _objective(best_f)
    
    # Candidate 1: All ones (baseline)
    # best_obj should be ~1/N = 0.0025
    
    # Candidate 2: Alternating +1/-1
    alt = np.ones(N)
    alt[1::2] = -1
    obj = _objective(alt)
    if obj < best_obj:
        best_f = alt
        best_obj = obj
    
    # Candidate 3: Blocks of 4: ++++----++++----...
    blocks = np.ones(N)
    for i in range(0, N, 8):
        blocks[i+4:i+8] = -1
    obj = _objective(blocks)
    if obj < best_obj:
        best_f = blocks
        best_obj = obj
    
    # Candidate 4: m-sequence-like pattern (Gold code style)
    # Use a simple LFSR to generate pseudo-random ±1 sequence
    # This can have better autocorrelation properties
    def lfsr_sequence(n, seed=7):
        """Generate pseudo-random ±1 sequence using LFSR."""
        x = np.random.RandomState(seed)
        # Simple approach: use random bits but structured
        bits = x.randint(0, 2, size=n)
        return 1.0 - 2.0 * bits
    
    rand_seq = lfsr_sequence(N, seed=42)
    obj = _objective(rand_seq)
    if obj < best_obj:
        best_f = rand_seq
        best_obj = obj
    
    # Candidate 5: Different seeds for random sequences
    for seed in [1, 7, 13, 99, 123, 456, 789, 2024]:
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        seq = lfsr_sequence(N, seed=seed)
        obj = _objective(seq)
        if obj < best_obj:
            best_f = seq.copy()
            best_obj = obj
    
    # Candidate 6: Structured low-autocorrelation pattern
    # Try: + + + - - - - + + + - - - - (period 12)
    period_12 = np.ones(N)
    for i in range(N):
        pos = i % 12
        if pos in [3, 4, 5, 6, 7, 8, 9]:
            period_12[i] = -1
    obj = _objective(period_12)
    if obj < best_obj:
        best_f = period_12.copy()
        best_obj = obj
    
    # Now do local optimization from the best candidate
    if time.time() - _start_time < _TIME_LIMIT - 15:
        best_f, best_obj = _local_sign_flip(best_f, n_trials=5000)
    
    # Try local optimization from all-ones too if we have time
    if time.time() - _start_time < _TIME_LIMIT - 10:
        ones = np.ones(N)
        f2, obj2 = _local_sign_flip(ones, n_trials=3000)
        if obj2 < best_obj:
            best_f = f2
            best_obj = obj2
    
    # Final: try more local flips if time permits
    if time.time() - _start_time < _TIME_LIMIT - 3:
        best_f, best_obj = _local_sign_flip(best_f, n_trials=5000)
    
    return best_f


# EVOLVE-BLOCK-END
