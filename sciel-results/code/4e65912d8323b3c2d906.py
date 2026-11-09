"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _generate_bitonic_network(n):
    """Generate Bitonic Sort network for n wires."""
    def _bitonic_sort(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        network = []
        network.extend(_bitonic_sort(start, half, 1))
        network.extend(_bitonic_sort(start + half, half, -1))
        network.extend(_bitonic_merge(start, length, 1))
        return network

    def _bitonic_merge(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        network = []
        for i in range(half):
            left = start + i
            right = start + i + half
            if direction == 1:
                network.append((left, right))
            else:
                network.append((right, left))
        network.extend(_bitonic_merge(start, half, direction))
        network.extend(_bitonic_merge(start + half, half, direction))
        return network

    return _bitonic_sort(0, n, 1)

def _verify_network_fast(network):
    """
    Ultra-fast verification using bit-parallelism over all 65536 inputs.
    Each wire's state is packed into uint64 words (64 inputs per word).
    Returns 0 if network is correct, else number of unsorted inputs.
    """
    # Pre-compute bit masks for each wire
    # For wire k, the pattern of which of 64 consecutive masks have bit k set
    # is determined by k's position in the 6-bit counter
    # We'll use numpy to create the initial state efficiently
    
    # Create all 65536 masks
    all_masks = np.arange(1 << N, dtype=np.uint64)
    
    # For each wire k, extract bit k from all masks
    # Pack into groups of 64: wire k has 1024 uint64 values
    # bit k of mask m is (m >> k) & 1
    # For 64 consecutive masks starting at base, the bits form a known pattern
    
    # Efficient approach: create state as (16, 1024) uint64 array
    # state[k, g] = 64-bit word where bit b is the value of wire k for input (g*64 + b)
    state = np.zeros((N, 1024), dtype=np.uint64)
    
    # For wire k, input m has bit k = (m >> k) & 1
    # For group g, inputs are g*64 to g*64+63
    # (m >> k) & 1 for m = g*64 + b
    # = ((g*64 + b) >> k) & 1
    # = ((g << 6 + b) >> k) & 1
    # = ((g << (6-k) + (b >> k)) & 1) if k < 6, but this gets complex
    # Simpler: just compute directly for the 64 bits in each group
    
    for k in range(N):
        # For each group of 64, compute the 64-bit word
        # bit b of word = ((g*64 + b) >> k) & 1 for b = 0..63
        # This is a periodic pattern with period 2^(k+1)
        # For k < 6: the pattern repeats every 2^(k+1) bits within the 64-bit word
        # For k >= 6: depends on g
        
        if k < 6:
            # Pattern has period 2^(k+1), and within each period, lower k bits are 0, upper 2^k bits are 1
            # Actually: (b >> k) & 1, so for b in [0, 2^k-1]: 0, [2^k, 2^(k+1)-1]: 1, repeat
            period = 1 << (k + 1)
            half_period = 1 << k
            # Create one period: half_period zeros followed by half_period ones
            pattern_word = 0
            for b in range(period):
                if (b >> k) & 1:
                    pattern_word |= (1 << b)
            # Tile this pattern to fill 64 bits
            # 64 / period repetitions
            reps = 64 // period
            word = 0
            for r in range(reps):
                word |= (pattern_word << (r * period))
            # All groups have the same word
            state[k, :] = np.uint64(word)
        else:
            # k >= 6: (m >> k) & 1 where m = g*64 + b
            # = ((g*64 + b) >> k) & 1
            # = (g >> (k-6)) & 1  (since b < 64 = 2^6, and k >= 6)
            # So for group g, all 64 bits are the same: (g >> (k-6)) & 1
            for g in range(1024):
                bit_val = (g >> (k - 6)) & 1
                if bit_val:
                    state[k, g] = np.uint64(0xFFFFFFFFFFFFFFFF)
                else:
                    state[k, g] = np.uint64(0)
    
    # Apply network
    for i, j in network:
        # Compare-exchange: if state[i] > state[j] (bit-wise), swap
        # "state[i] > state[j]" means: for each input, wire i has 1 and wire j has 0
        # mask = state[i] & ~state[j]  (bit is 1 where i=1 and j=0)
        mask = state[i] & ~state[j]
        # Where mask has 1s: set i to 0, j to 1
        state[i] &= ~mask
        state[j] |= mask
    
    # Check if sorted: for all adjacent pairs, state[k] <= state[k+1]
    # i.e., no input has state[k]=1 and state[k+1]=0
    # violation = state[k] & ~state[k+1]
    violations = 0
    for k in range(N - 1):
        viol = state[k] & ~state[k + 1]
        if np.any(viol != 0):
            # Count the number of violating inputs
            violations += int(np.sum(np.unpackbits(viol.view(np.uint8).reshape(-1), axis=1).sum(axis=1)))
    
    return violations

def _verify_network_simple(network):
    """Simple verification using numpy array (slower but correct)."""
    all_masks = np.arange(1 << N, dtype=np.uint64)
    inputs_arr = ((all_masks[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
    
    for i, j in network:
        mask = inputs_arr[:, i] > inputs_arr[:, j]
        if np.any(mask):
            tmp = inputs_arr[mask, i].copy()
            inputs_arr[mask, i] = inputs_arr[mask, j]
            inputs_arr[mask, j] = tmp
    
    diffs = np.diff(inputs_arr, axis=1)
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def _construct_impl():
    start_time = time.time()
    DEADLINE = 87  # seconds, with safety margin for 90s kill
    
    # Start with Bitonic Sort (119 comparators for n=16)
    best_net = _generate_bitonic_network(N)
    
    # Verify initial network using fast method
    if _verify_network_fast(best_net) != 0:
        # Fallback: verify with simple method
        if _verify_network_simple(best_net) != 0:
            # Fallback to odd-even transposition
            best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    random.seed(42)
    
    # Phase 1: Aggressive removal via fast verification
    # Try removing each comparator one at a time (systematic)
    improved = True
    while improved and (time.time() - start_time < DEADLINE - 10):
        improved = False
        for idx in range(len(best_net)):
            if time.time() - start_time > DEADLINE - 10:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network_fast(candidate) == 0:
                best_net = candidate
                improved = True
                break
    
    # Phase 2: Random removal and swap attempts
    while time.time() - start_time < DEADLINE - 5:
        if len(best_net) <= 55:
            break
        action = random.random()
        if action < 0.6:
            # Try removing a random comparator
            idx = random.randint(0, len(best_net) - 1)
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network_fast(candidate) == 0:
                best_net = candidate
        elif action < 0.8:
            # Try swapping adjacent comparators
            if len(best_net) >= 2:
                idx = random.randint(0, len(best_net) - 2)
                candidate = best_net[:idx] + [best_net[idx+1], best_net[idx]] + best_net[idx+2:]
                if _verify_network_fast(candidate) == 0:
                    best_net = candidate
        else:
            # Try replacing a comparator with a different one
            idx = random.randint(0, len(best_net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = best_net[:idx] + [(i, j)] + best_net[idx+1:]
            if _verify_network_fast(candidate) == 0:
                best_net = candidate
    
    # Phase 3: Try to find a shorter network by adding then removing
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Add 1-2 random comparators
            candidate = list(best_net)
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                idx = random.randint(0, len(candidate))
                candidate.insert(idx, (i, j))
            # Try to remove random comparators
            for _ in range(15):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ridx = random.randint(0, len(candidate) - 1)
                reduced = candidate[:ridx] + candidate[ridx+1:]
                if _verify_network_fast(reduced) == 0:
                    candidate = reduced
            if len(candidate) < len(best_net):
                best_net = candidate
    
    # Final verification
    if _verify_network_fast(best_net) != 0:
        # Double-check with simple method
        if _verify_network_simple(best_net) != 0:
            # Fallback to a known good network
            best_net = _generate_bitonic_network(N)
    
    return best_net

def construct():
    try:
        return _construct_impl()
    except Exception:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
