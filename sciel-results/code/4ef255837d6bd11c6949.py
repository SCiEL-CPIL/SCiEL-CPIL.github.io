import numpy as np
import time
from scipy.optimize import linprog

# Problem: N step heights h_i in [0,1], sum h_i = N/2.
# Overlap at shift k (circular): O_k = sum_{i=0}^{N-1} h_i * (1 - h_{i+k mod N})
# We want to minimize max_k O_k.
#
# Key insight: This is a combinatorial problem. The optimal solution is 
# a binary sequence (h_i in {0,1}) with exactly N/2 ones, which minimizes 
# the maximum autocorrelation. This is equivalent to finding a sequence 
# with minimum peak autocorrelation (apart from the DC term).
#
# Strategy: Use simulated annealing on binary sequences to find one with 
# low peak autocorrelation, then verify/improve with local search.

N = 100

def _overlap_max(h):
    """Compute max overlap for a given h array."""
    n = len(h)
    max_ov = 0.0
    for k in range(1, n):
        # O_k = sum h_i * (1 - h_{i+k})
        shifted = np.roll(h, -k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _binary_overlap_max(bits):
    """Compute max overlap for a binary sequence (0/1 array)."""
    n = len(bits)
    max_ov = 0.0
    for k in range(1, n):
        shifted = np.roll(bits, -k)
        # For binary: h_i*(1-h_{i+k}) = 1 if h_i=1 and h_{i+k}=0
        ov = np.sum(bits * (1 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _local_search(bits, max_iter=50000, time_limit=10.0):
    """Local search: flip pairs of bits (one 1->0, one 0->1) to maintain sum=N/2."""
    n = len(bits)
    bits = bits.copy()
    best_score = _binary_overlap_max(bits)
    ones = np.where(bits == 1)[0]
    zeros = np.where(bits == 0)[0]
    
    rng = np.random.default_rng(42)
    start = time.time()
    
    for _ in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Pick a random 1 and a random 0 to swap
        i1 = rng.integers(len(ones))
        i0 = rng.integers(len(zeros))
        pos1 = ones[i1]
        pos0 = zeros[i0]
        
        # Try the swap
        bits[pos1] = 0
        bits[pos0] = 1
        
        new_score = _binary_overlap_max(bits)
        
        if new_score < best_score:
            best_score = new_score
            # Update indices
            ones[i1] = pos0
            zeros[i0] = pos1
            # Keep sorted for consistency (not strictly needed but cleaner)
            ones = np.sort(ones)
            zeros = np.sort(zeros)
        else:
            # Revert
            bits[pos1] = 1
            bits[pos0] = 0
    
    return bits, best_score

def _simulated_annealing(bits, time_limit=20.0, T_start=50.0, T_end=0.01):
    """Simulated annealing on binary sequences."""
    n = len(bits)
    bits = bits.copy()
    best_bits = bits.copy()
    best_score = _binary_overlap_max(bits)
    current_score = best_score
    
    ones = np.where(bits == 1)[0]
    zeros = np.where(bits == 0)[0]
    
    rng = np.random.default_rng(123)
    start = time.time()
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / 100000)
    
    while time.time() - start < time_limit:
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        # Pick random 1 and 0 to swap
        i1 = rng.integers(len(ones))
        i0 = rng.integers(len(zeros))
        pos1 = ones[i1]
        pos0 = zeros[i0]
        
        bits[pos1] = 0
        bits[pos0] = 1
        
        new_score = _binary_overlap_max(bits)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            ones[i1] = pos0
            zeros[i0] = pos1
            if new_score < best_score:
                best_score = new_score
                best_bits = bits.copy()
        else:
            bits[pos1] = 1
            bits[pos0] = 0
        
        T = max(T * cooling, T_end)
    
    return best_bits, best_score

def construct():
    """Construct step heights minimizing max overlap with shifted 1-h."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with a balanced binary sequence
    bits = np.zeros(N, dtype=np.int8)
    bits[:N//2] = 1
    
    # Phase 1: Local search from multiple random starts
    best_bits = bits.copy()
    best_score = _binary_overlap_max(bits)
    
    # Try several random initializations
    rng = np.random.default_rng(99)
    for trial in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Random balanced binary sequence
        b = np.zeros(N, dtype=np.int8)
        idx = rng.permutation(N)
        b[idx[:N//2]] = 1
        
        # Quick local search
        b, score = _local_search(b, max_iter=10000, time_limit=3.0)
        if score < best_score:
            best_score = score
            best_bits = b.copy()
    
    # Phase 2: Simulated annealing from the best found
    if time.time() < deadline - 20.0:
        sa_bits, sa_score = _simulated_annealing(best_bits, time_limit=min(30.0, deadline - time.time() - 5.0))
        if sa_score < best_score:
            best_score = sa_score
            best_bits = sa_bits
    
    # Phase 3: More local search from the SA result
    if time.time() < deadline - 10.0:
        ls_bits, ls_score = _local_search(best_bits, max_iter=50000, time_limit=min(15.0, deadline - time.time() - 5.0))
        if ls_score < best_score:
            best_score = ls_score
            best_bits = ls_bits
    
    # Convert to float h in [0,1]
    h = best_bits.astype(np.float64)
    
    # Verify constraints
    assert np.all(h >= 0) and np.all(h <= 1), "h out of bounds"
    assert abs(np.sum(h) - N/2) < 1e-10, f"sum mismatch: {np.sum(h)}"
    
    return h
