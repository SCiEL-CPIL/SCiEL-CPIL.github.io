"""Binary-sequence construction for the Erdos minimum-overlap task.

We model h_i as a binary sequence (0 or 1) with exactly N/2 ones (duty cycle 1/2).
For binary h, the overlap O(s) = sum_i h_i * (1 - h_{i-s}) counts the number of
positions where h has a 1 and the shifted (1-h) has a 1, i.e., where h_i=1 and
h_{i-s}=0. This equals the number of "10" pairs at lag s.

For a balanced binary sequence, O(s) is related to the autocorrelation C(s) =
sum_i h_i * h_{i-s}. Specifically, O(s) = (N/2) - C(s) for s != 0, since
sum h_i(1-h_{i-s}) = sum h_i - sum h_i h_{i-s} = N/2 - C(s).

Thus minimizing max O(s) is equivalent to maximizing min C(s) for s != 0,
or equivalently, making the autocorrelation as flat as possible (low sidelobes).

We use m-sequences (maximal-length LFSR sequences) which have ideal periodic
autocorrelation: C(s) = -1 for all s != 0 (up to the DC term). For a balanced
sequence with N/2 ones, we get C(s) = -1 for s != 0, so O(s) = N/2 - (-1) = N/2 + 1.

However, m-sequences have length 2^m - 1, which is odd. We need N = 100 (even).
Instead, we can use a different approach: construct a binary sequence with
balanced property and low autocorrelation sidelobes via greedy or structured
construction.

Actually, let's use a simpler structural approach: a "balanced" binary sequence
where we alternate in a pattern that minimizes autocorrelation. One good candidate
is the Thue-Morse sequence, which has low autocorrelation.

But the directive suggests using rational duty cycle binary sequences. Let's try:
1. Start with a random balanced binary sequence
2. Use simulated annealing to minimize max overlap

Or better: use a structured approach with a specific permutation that has
good autocorrelation properties.

Let me try a different angle: for N=100, use a binary sequence based on
quadratic residues or a similar algebraic construction.

Actually, the simplest structural improvement: use a binary sequence and
optimize it via local search (flipping pairs to maintain balance).
"""

import time
import numpy as np
from scipy.fft import fft, ifft

N = 100
DEADLINE_S = 145.0


def _time_remaining():
    return DEADLINE_S - time.time()


def _max_overlap(h):
    """Maximum circular overlap of h with shifted (1-h), via FFT."""
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    return float(np.max(np.real(ifft(F))))


def _max_overlap_binary(h):
    """Fast max overlap for binary h using direct computation."""
    # O(s) = sum_i h_i * (1 - h_{i-s})
    # For binary h, this is the number of i where h_i=1 and h_{i-s}=0
    best = 0.0
    for s in range(1, N):
        count = 0
        for i in range(N):
            if h[i] == 1 and h[(i - s) % N] == 0:
                count += 1
        if count > best:
            best = count
    return float(best)


def _binary_init():
    """Initialize a balanced binary sequence with N/2 ones."""
    h = np.zeros(N, dtype=np.float64)
    h[:N // 2] = 1.0
    # Random shuffle
    np.random.shuffle(h)
    return h


def _local_search(h0, max_iters=5000):
    """Local search on binary sequences: flip pairs (1->0, 0->1) to reduce max overlap."""
    h = h0.copy()
    best_v = _max_overlap(h)
    best_h = h.copy()
    
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Try a random flip pair
        if len(ones) < 2 or len(zeros) < 2:
            break
        
        i1 = ones[np.random.randint(len(ones))]
        i2 = zeros[np.random.randint(len(zeros))]
        
        # Apply flip
        h[i1] = 0.0
        h[i2] = 1.0
        
        v = _max_overlap(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
        else:
            # Revert
            h[i1] = 1.0
            h[i2] = 0.0
    
    return best_h, best_v


def _structured_binary():
    """Construct a binary sequence with good autocorrelation properties.
    
    Uses a pattern based on quadratic residues for N=100.
    """
    h = np.zeros(N, dtype=np.float64)
    
    # Use a balanced pattern: place ones at positions that are quadratic residues mod 100
    # or use a simpler structured pattern
    
    # Try: alternating blocks of length 2 (11001100...)
    for i in range(N):
        if (i // 2) % 2 == 0:
            h[i] = 1.0
    
    # Ensure exactly N/2 ones
    num_ones = int(np.sum(h))
    if num_ones > N // 2:
        # Remove excess ones
        excess = num_ones - N // 2
        ones_idx = np.where(h == 1.0)[0]
        for idx in ones_idx[:excess]:
            h[idx] = 0.0
    elif num_ones < N // 2:
        # Add needed ones
        needed = N // 2 - num_ones
        zeros_idx = np.where(h == 0.0)[0]
        for idx in zeros_idx[:needed]:
            h[idx] = 1.0
    
    return h


def construct():
    t0 = time.time()
    
    # Try multiple initializations and pick the best
    best_h = None
    best_v = float('inf')
    
    # 1. Structured binary initialization
    h_struct = _structured_binary()
    v_struct = _max_overlap(h_struct)
    if v_struct < best_v:
        best_v = v_struct
        best_h = h_struct.copy()
    
    # 2. Random binary initialization with local search
    for trial in range(3):
        if _time_remaining() < 10.0:
            break
        
        h_rand = _binary_init()
        h_ls, v_ls = _local_search(h_rand, max_iters=2000)
        if v_ls < best_v:
            best_v = v_ls
            best_h = h_ls.copy()
    
    # 3. If time allows, do more local search on the best solution
    if _time_remaining() > 15.0 and best_h is not None:
        h_final, v_final = _local_search(best_h, max_iters=10000)
        if v_final < best_v:
            best_v = v_final
            best_h = h_final.copy()
    
    # Ensure valid output
    if best_h is None:
        best_h = _binary_init()
    
    # Final projection to ensure sum is exactly N/2 and values in [0,1]
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = best_h.sum() - 0.5 * N
    if abs(diff) > 1e-12:
        if diff > 0:
            order = np.argsort(best_h)[::-1]
            for i in order:
                take = min(diff, best_h[i])
                best_h[i] -= take
                diff -= take
                if diff <= 1e-12:
                    break
        else:
            diff = -diff
            order = np.argsort(best_h)
            for i in order:
                take = min(diff, 1.0 - best_h[i])
                best_h[i] += take
                diff -= take
                if diff <= 1e-12:
                    break
    
    return best_h
