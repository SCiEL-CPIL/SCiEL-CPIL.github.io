"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct a signed step function with 400 intervals to minimize autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def compute_sidelobe_max(h):
        """Compute max |autoconv| at non-zero lags."""
        ac = np.correlate(h, h, mode='full')
        center = N - 1
        nonzero = np.concatenate([ac[:center], ac[center+1:]])
        return np.max(np.abs(nonzero))

    def compute_sidelobe_l2(h):
        """Compute L2 norm of autoconv at non-zero lags."""
        ac = np.correlate(h, h, mode='full')
        center = N - 1
        nonzero = np.concatenate([ac[:center], ac[center+1:]])
        return np.sqrt(np.sum(nonzero**2))

    best_h = None
    best_score = np.inf

    # Try many random sequences
    for seed in range(200):
        if time.time() > deadline - 10.0:
            break
        rng = np.random.RandomState(seed)
        h = rng.choice([-1.0, 1.0], size=N)
        # Use max sidelobe as primary metric
        s = compute_sidelobe_max(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Try quadratic residue sequence (good autocorrelation properties)
    if time.time() < deadline - 10.0:
        p = 401  # prime
        legendre = np.zeros(p)
        for a in range(1, p):
            val = pow(a, (p-1)//2, p)
            legendre[a] = -1.0 if val == p-1 else 1.0
        legendre[0] = 1.0
        h_qr = legendre[:N].copy()
        s = compute_sidelobe_max(h_qr)
        if s < best_score:
            best_score = s
            best_h = h_qr

    # Try m-sequence based construction
    if time.time() < deadline - 10.0:
        # Generate an m-sequence of length 2^m - 1 where 2^m - 1 >= N
        # 2^9 - 1 = 511 >= 400
        m = 9
        L = (1 << m) - 1  # 511
        # Simple LFSR to generate m-sequence
        # Use polynomial x^9 + x^4 + 1 (primitive)
        state = 1
        seq = []
        for i in range(L):
            seq.append(1.0 if (state & 1) else -1.0)
            # Tap positions for x^9 + x^4 + 1: bits 0 and 4 (0-indexed)
            feedback = ((state >> 0) ^ (state >> 4)) & 1
            state = (state >> 1) | (feedback << (m - 1))
        h_mseq = np.array(seq[:N])
        s = compute_sidelobe_max(h_mseq)
        if s < best_score:
            best_score = s
            best_h = h_mseq

    # If no good candidate found, use a random one
    if best_h is None:
        rng = np.random.RandomState(0)
        best_h = rng.choice([-1.0, 1.0], size=N)

    return best_h
# EVOLVE-BLOCK-END
