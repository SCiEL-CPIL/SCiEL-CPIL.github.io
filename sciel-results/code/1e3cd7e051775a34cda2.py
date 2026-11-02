"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400


def _fft_autocorr_max_abs(s):
    """Compute max |autocorr| over non-zero lags for a length-N integer sequence.

    Returns the maximum of |r[k]| for k=1..N-1 where r = ifft(|fft(s, n_fft)|^2).real
    and r[k] is the circular autocorrelation at lag k (with n_fft large enough to
    capture the linear autocorrelation).
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.fft(s_f, n=n_fft)
    acf = np.fft.ifft(F * np.conj(F)).real[:N]
    # Normalize by N so that lag-0 is ~sum(s^2)/N
    acf /= N
    return float(np.max(np.abs(acf[1:])))


def _score(s):
    """Score = upper bound on C3; lower is better."""
    return _fft_autocorr_max_abs(s)


def _random_walk_sequence():
    """Generate a random +/-1 sequence."""
    return np.random.choice([-1, 1], size=N)


def _construct_seed():
    """Construct the all-ones seed (score 1.0 baseline)."""
    return np.ones(N, dtype=np.int8)


def _flip_block(seq, start, length):
    """Flip a contiguous block of `length` positions starting at `start` (circular)."""
    idx = np.arange(start, start + length) % N
    seq[idx] = -seq[idx]


def construct():
    """
    Search for a signed step function (400 intervals of +/-1) that minimizes
    the max absolute off-peak autocorrelation, which upper-bounds C3.

    Strategy:
    - Start with a random +/-1 sequence (better initial landscape than all-ones).
    - Run simulated annealing with block-flip moves (contiguous segments of
      varying length) to escape local minima.
    - Temperature decays geometrically; accept uphill moves with probability
      exp(-delta/T).
    - Keep the best sequence found within the time budget.
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = np.random.default_rng(42)

    # Start with a random sequence (much better initial score than all-ones)
    best_seq = _random_walk_sequence().astype(np.int8)
    best_score = _score(best_seq)

    seq = best_seq.copy()
    current_score = best_score

    # Simulated annealing parameters
    T_start = 0.5
    T_min = 1e-6
    # We'll run many iterations; compute cooling rate so we hit T_min near deadline
    # First, estimate how many iterations we can do
    t0 = time.time()
    # Quick benchmark: time one score evaluation
    _score(seq)
    eval_time = time.time() - t0
    # Rough estimate: ~50us per eval, plus overhead
    # We'll adaptively choose number of iterations
    est_iters = max(1000, int(100 / max(eval_time, 1e-6)))
    cooling_rate = (T_min / T_start) ** (1.0 / est_iters)

    T = T_start
    block_lengths = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

    while time.time() < deadline:
        # Choose a block length (biased toward shorter blocks)
        # Use a power-law-like distribution: shorter blocks more likely
        r = rng.random()
        if r < 0.3:
            blen = rng.integers(1, 6)  # 1-5
        elif r < 0.6:
            blen = rng.integers(6, 17)  # 6-16
        elif r < 0.8:
            blen = rng.integers(17, 49)  # 17-48
        elif r < 0.95:
            blen = rng.integers(49, 129)  # 49-128
        else:
            blen = rng.integers(129, N)  # 129-399

        start = rng.integers(0, N)
        _flip_block(seq, start, blen)
        new_score = _score(seq)

        delta = new_score - current_score
        if delta < 0:
            # Accept improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_seq = seq.copy()
        else:
            # Accept uphill move with probability exp(-delta/T)
            if T > T_min:
                accept_prob = np.exp(-delta / T)
                if rng.random() < accept_prob:
                    current_score = new_score
                else:
                    # Revert
                    _flip_block(seq, start, blen)
            else:
                # Revert
                _flip_block(seq, start, blen)

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_seq
