"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np

def _batcher_odd_even_merge_sort(n):
    """Batcher's odd-even merge sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    left = _batcher_odd_even_merge_sort(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(half)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge network for 2n wires (n from each half).
    Returns comparators for wires 0..2n-1 where 0..n-1 are 'even' (left) and n..2n-1 are 'odd' (right)."""
    if n == 1:
        return [(0, 1)]
    even_merge = _odd_even_merge(n // 2)
    odd_merge = _odd_even_merge(n // 2)
    even_translated = [(2*i, 2*j) for (i, j) in even_merge]
    odd_translated = [(2*i + 1, 2*j + 1) for (i, j) in odd_merge]
    final_merge = [(2*i + 1, 2*i + 2) for i in range(n - 1)]
    return even_translated + odd_translated + final_merge

def _verify_network_fast(comparators, n):
    """Fast verification using bitset representation.
    Returns True if the network sorts all 2^n zero-one inputs."""
    total = 1 << n
    # For each input mask, track the state of each wire as a bit in a Python int
    # wire_states[w] is a bitset: bit k is 1 if wire w has value 1 for input k
    wire_states = [0] * n
    for w in range(n):
        # Wire w gets bit w of each mask
        # For mask m, wire w = (m >> w) & 1
        # We want wire_states[w] to have bit m set iff (m >> w) & 1 == 1
        # That means bit m is set when m has bit w set
        # So wire_states[w] = sum of all masks that have bit w set
        # = pattern: for each bit position m, set if (m & (1<<w)) != 0
        # This is: repeat (2^w ones, 2^w zeros) for total/(2^(w+1)) times
        block = (1 << (1 << w)) - 1  # 2^w ones
        period = 1 << (w + 1)
        wire_states[w] = 0
        for start in range(0, total, period):
            wire_states[w] |= block << start
    
    for (i, j) in comparators:
        si = wire_states[i]
        sj = wire_states[j]
        # min goes to i: si & sj (both must be 1)
        # max goes to j: si | sj (at least one must be 1)
        wire_states[i] = si & sj
        wire_states[j] = si | sj
    
    # Check sorted: for all adjacent pairs, wire[w] <= wire[w+1]
    # i.e., wire_states[w] & ~wire_states[w+1] should be 0
    for w in range(n - 1):
        if wire_states[w] & ~wire_states[w + 1] != 0:
            return False
    return True

def _remove_redundant_fast(comparators, n, deadline):
    """Remove redundant comparators using fast verification."""
    comparators = list(comparators)
    i = 0
    while i < len(comparators) and time.time() < deadline:
        candidate = comparators[:i] + comparators[i+1:]
        if _verify_network_fast(candidate, n):
            comparators = candidate
            # Don't increment i, check same position again
        else:
            i += 1
    return comparators

def _try_swap_order(comparators, n, deadline):
    """Try swapping adjacent comparators to enable further pruning."""
    comparators = list(comparators)
    i = 0
    while i < len(comparators) - 1 and time.time() < deadline:
        swapped = comparators[:i] + [comparators[i+1], comparators[i]] + comparators[i+2:]
        if _verify_network_fast(swapped, n):
            # After swap, try to remove one of them
            for rem in [i, i+1]:
                candidate = swapped[:rem] + swapped[rem+1:]
                if _verify_network_fast(candidate, n):
                    comparators = candidate
                    return True
            # Just keep the swap if it's valid
            comparators = swapped
            i += 1
        else:
            i += 1
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Build Batcher's odd-even merge sort network
    network = _batcher_odd_even_merge_sort(N)
    
    # Verify it's correct
    if not _verify_network_fast(network, N):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Remove redundant comparators
    pruned = _remove_redundant_fast(network, N, deadline)
    
    # Try swap-and-remove passes
    for _ in range(10):
        if time.time() > deadline:
            break
        if not _try_swap_order(pruned, N, deadline):
            break
    
    # Final verification
    if not _verify_network_fast(pruned, N):
        return network
    
    return pruned
# EVOLVE-BLOCK-END
