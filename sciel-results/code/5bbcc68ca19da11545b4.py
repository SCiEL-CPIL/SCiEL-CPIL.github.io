"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE_FRACTION = 0.90  # use 90% of the 150s budget as safety margin


def _autocorr_peak(seq):
    """Compute the maximum absolute off-diagonal autocorrelation value."""
    # Use FFT for efficient autocorrelation
    n = len(seq)
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(seq, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)[:n]
    # ac[0] is the peak (lag 0), we want max |ac[k]| for k != 0
    peak = np.max(np.abs(ac[1:]))
    return peak


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n mapped to ±1."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count bits in i
        bits = bin(i).count('1')
        seq[i] = 1.0 if bits % 2 == 0 else -1.0
    return seq


def _quasi_periodic(n, freq1=0.31, freq2=0.17):
    """Generate quasi-periodic signed sequence."""
    t = np.arange(n)
    x = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)
    return np.sign(x)


def _random_seq(n, seed):
    """Generate random ±1 sequence."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=n)


def _greedy_local_search(seq, max_rounds=50, deadline=None):
    """Greedy local search: flip individual elements to reduce peak autocorrelation."""
    n = len(seq)
    current_peak = _autocorr_peak(seq)
    seq = seq.copy()
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping element i
            seq[i] *= -1
            new_peak = _autocorr_peak(seq)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                seq[i] *= -1  # revert
        if not improved:
            break
    return seq, current_peak


def _simulated_annealing(seq, max_iters=2000, t_start=1.0, t_end=0.001, deadline=None):
    """Simulated annealing to escape local minima."""
    n = len(seq)
    current_peak = _autocorr_peak(seq)
    best_seq = seq.copy()
    best_peak = current_peak
    seq = seq.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        T = t_start * (t_end / t_start) ** (it / max_iters)
        # Random flip
        i = np.random.randint(n)
        seq[i] *= -1
        new_peak = _autocorr_peak(seq)
        delta = new_peak - current_peak
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_seq = seq.copy()
        else:
            seq[i] *= -1  # revert
    
    return best_seq, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION
    
    best_seq = None
    best_score = np.inf
    
    # Multiple diverse initializations
    initializations = [
        ("thue_morse", _thue_morse(N)),
        ("quasi_periodic_1", _quasi_periodic(N, 0.31, 0.17)),
        ("quasi_periodic_2", _quasi_periodic(N, 0.23, 0.11)),
        ("quasi_periodic_3", _quasi_periodic(N, 0.41, 0.29)),
        ("random_1", _random_seq(N, 42)),
        ("random_2", _random_seq(N, 123)),
        ("random_3", _random_seq(N, 999)),
    ]
    
    for name, init_seq in initializations:
        if time.time() > deadline:
            break
        
        # Greedy local search
        seq, score = _greedy_local_search(init_seq, max_rounds=30, deadline=deadline)
        if score < best_score:
            best_score = score
            best_seq = seq.copy()
        
        # If we have time, try simulated annealing from the best found
        if time.time() < deadline - 20:
            sa_seq, sa_score = _simulated_annealing(seq, max_iters=1000, deadline=deadline)
            if sa_score < best_score:
                best_score = sa_score
                best_seq = sa_seq.copy()
                
                # Follow up with greedy from SA result
                if time.time() < deadline - 10:
                    g_seq, g_score = _greedy_local_search(sa_seq, max_rounds=20, deadline=deadline)
                    if g_score < best_score:
                        best_score = g_score
                        best_seq = g_seq.copy()
    
    if best_seq is None:
        # Fallback: return all ones (shouldn't happen)
        return np.ones(N)
    
    return best_seq
