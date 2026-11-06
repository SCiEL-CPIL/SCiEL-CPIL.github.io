"""
Size-efficient sorting network for 18 wires.

Strategy:
  1. Start with a known-good odd-even transposition network (153 comparators).
  2. Use a bit-parallel verifier over all 2^18 Boolean inputs to certify correctness.
  3. Perform local search (comparator deletion, replacement, insertion) to minimize count.
  4. Time-budgeted: return best found before deadline.
"""

import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# Precompute all Boolean inputs as a bit matrix: shape (NUM_INPUTS, N)
# Each row is a distinct 18-bit pattern.
def _build_inputs():
    # Use numpy for speed
    arr = np.arange(NUM_INPUTS, dtype=np.uint32)
    # Extract bits
    bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = (arr >> i) & 1
    return bits

def _apply_network(net, inputs):
    """
    Apply the compare-exchange network to all inputs simultaneously.
    net: list of (i, j)
    inputs: (NUM_INPUTS, N) uint8 array
    Returns: sorted (NUM_INPUTS, N) uint8 array
    """
    x = inputs.copy()
    for (i, j) in net:
        # min goes to i, max goes to j
        # For each input row, compare x[row, i] and x[row, j]
        a = x[:, i]
        b = x[:, j]
        # min = a & b (for bits: min is 1 only if both are 1)
        # max = a | b
        x[:, i] = a & b
        x[:, j] = a | b
    return x

def _is_sorted(x):
    """Check if each row is sorted (non-decreasing). For binary, this means all 0s then all 1s."""
    # A row is sorted if there is no index k where x[k] = 0 and x[k+1] = 1
    # i.e., for each row, the sequence of bits is 0...01...1
    # Equivalent: for each adjacent pair (k, k+1), NOT (x[k]==0 AND x[k+1]==1)
    # x[:, k] == 0 and x[:, k+1] == 1
    for k in range(N - 1):
        bad = (x[:, k] == 0) & (x[:, k+1] == 1)
        if np.any(bad):
            return False
    return True

def _verify(net):
    """Verify network sorts all 2^18 Boolean inputs."""
    x = _apply_network(net, _INPUTS)
    return _is_sorted(x)

# Precompute inputs once at module level for speed
_INPUTS = _build_inputs()

def _odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _random_net(n_comps):
    """Generate a random comparator network of given size."""
    net = []
    for _ in range(n_comps):
        i, j = sorted(random.sample(range(N), 2))
        net.append((i, j))
    return net

def construct():
    """
    Construct a sorting network for 18 wires with as few comparators as possible.
    Returns: ordered list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with odd-even transposition
    best_net = _odd_even_transposition()
    best_len = len(best_net)
    
    # Verify initial network
    if not _verify(best_net):
        # Should not happen, but just in case
        return best_net
    
    # Local search: try to delete comparators
    # We'll do multiple passes of deletion attempts
    
    def try_delete(net):
        """Try to delete each comparator one at a time, keep if still valid."""
        improved = True
        while improved:
            improved = False
            for idx in range(len(net)):
                if time.time() > deadline:
                    return net
                # Try removing comparator at idx
                candidate = net[:idx] + net[idx+1:]
                if _verify(candidate):
                    net = candidate
                    improved = True
                    break  # restart loop since indices shifted
        return net
    
    # Phase 1: Greedy deletion
    best_net = try_delete(best_net)
    best_len = len(best_net)
    
    if time.time() > deadline:
        return best_net
    
    # Phase 2: Try swapping/replacing comparators to enable further deletions
    # Random restarts with perturbation
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Perturb: take current best, randomly modify a few comparators
        perturbed = list(best_net)
        
        # Randomly change 1-3 comparators
        for _ in range(random.randint(1, 3)):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                i, j = sorted(random.sample(range(N), 2))
                perturbed[idx] = (i, j)
        
        # Try to delete from perturbed
        candidate = try_delete(perturbed)
        
        if len(candidate) < best_len:
            best_net = candidate
            best_len = len(candidate)
    
    if time.time() > deadline:
        return best_net
    
    # Phase 3: More aggressive local search - try replacing pairs
    for _ in range(20):
        if time.time() > deadline:
            break
        
        candidate = list(best_net)
        
        # Try replacing one comparator with another
        idx = random.randint(0, len(candidate) - 1)
        i, j = sorted(random.sample(range(N), 2))
        candidate[idx] = (i, j)
        
        # Then try deletion
        candidate = try_delete(candidate)
        
        if len(candidate) < best_len:
            best_net = candidate
            best_len = len(candidate)
    
    return best_net
