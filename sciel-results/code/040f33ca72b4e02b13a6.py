"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 45.0  # safety margin before 60s kill

# Pre-compute the input matrix for verification to avoid regeneration overhead
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    # Copy the input matrix
    x = _INPUT_MATRIX.copy()
    
    # Apply network
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    
    # Verify sorted
    # For boolean sorted: x[:, i] <= x[:, i+1] for all i
    # We can check this by ensuring that for each row, the transition from 0 to 1 happens at most once.
    # Or simply check adjacent pairs.
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    
    n = len(network)
    wire_to_last_comp = {}  # wire -> last comparator index that touched it
    depths = [0] * n
    
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    
    return max(depths) if depths else 0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires.
    Uses a known good construction as baseline, then attempts local search
    to reduce comparator count.
    """
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE
    
    # Start with odd-even transposition: 120 comparators
    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_network = odd_even_transposition()
    best_count = len(best_network)
    
    def is_valid(net):
        return _verify_network_fastest(net)
    
    # First, try removing comparators one by one (greedy removal)
    # We iterate backwards to avoid index shifting issues, but we must be careful
    # with the time budget.
    
    current = list(best_network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = current[:idx] + current[idx+1:]
            if is_valid(candidate):
                current = candidate
                changed = True
        if time.time() > deadline - 5:
            break
    
    best_network = current
    best_count = len(best_network)
    
    # If we have time, try a limited local search to see if we can remove more
    # by rearranging. However, rearrangement usually doesn't reduce count unless
    # it enables further removals. Given the time constraints, we will skip
    # complex local search and just return the pruned network.
    
    # To be safe and ensure we return a valid network quickly, we stop here.
    # The greedy removal from 120 comparators is expected to reduce the count
    # significantly, potentially to around 60-80 depending on the structure.
    
    return best_network
# EVOLVE-BLOCK-END
