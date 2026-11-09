"""
Optimized sorting network construction for 16 wires.
Starts from a known efficient construction and aggressively removes redundant comparators.
"""
N = 16

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _generate_bitonic_network(n):
    """Generate Bitonic Sort network for n wires (119 comparators for n=16)."""
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

def _generate_odd_even_merge_network(n):
    """Generate Odd-Even Merge Sort network (72 comparators for n=16)."""
    def _oe_merge(a, b):
        # a and b are start indices, length is len(a)=len(b)=2^k
        length = n
        # This is a simplified version; let's use a recursive approach
        pass
    
    # Actually, let's just use the known construction
    # For 16 wires, odd-even merge sort uses 72 comparators
    # But let's implement it properly
    
    def _merge_odd_even(A, B):
        # A and B are lists of wire indices, each of length 2^k
        n = len(A)
        if n == 1:
            return [(A[0], B[0])]
        
        # Split into even and odd
        A_even = A[0::2]
        A_odd = A[1::2]
        B_even = B[0::2]
        B_odd = B[1::2]
        
        result = []
        # Merge evens
        result.extend(_merge_odd_even(A_even, B_even))
        # Merge odds
        result.extend(_merge_odd_even(A_odd, B_odd))
        # Interleave comparators
        for i in range(n // 2):
            result.append((A[2*i+1], B[2*i]))
        
        return result
    
    def _sort_odd_even(wires, direction):
        n = len(wires)
        if n <= 1:
            return []
        
        half = n // 2
        left = wires[:half]
        right = wires[half:]
        
        result = []
        # Sort left ascending
        result.extend(_sort_odd_even(left, 1))
        # Sort right descending
        result.extend(_sort_odd_even(right, -1))
        # Merge
        if direction == 1:
            result.extend(_merge_odd_even(left, right))
        else:
            # For descending, swap the order
            result.extend(_merge_odd_even(right, left))
        
        return result
    
    return _sort_odd_even(list(range(n)), 1)

def _verify_network_fast(network):
    """
    Fast verification using bit-parallelism.
    Returns 0 if network is correct, else number of unsorted inputs.
    """
    # State: for each wire k, a numpy array of uint64 words
    # Each word represents 64 consecutive input masks
    # Bit b of word g represents input (g*64 + b)
    
    state = np.zeros((N, 1024), dtype=np.uint64)
    
    # Initialize state
    for k in range(N):
        if k < 6:
            # Pattern repeats every 2^(k+1) bits
            period = 1 << (k + 1)
            half_period = 1 << k
            # Create one period
            pattern_word = 0
            for b in range(period):
                if (b >> k) & 1:
                    pattern_word |= (1 << b)
            # Tile to fill 64 bits
            reps = 64 // period
            word = 0
            for r in range(reps):
                word |= (pattern_word << (r * period))
            state[k, :] = np.uint64(word)
        else:
            # k >= 6: depends on group
            for g in range(1024):
                bit_val = (g >> (k - 6)) & 1
                if bit_val:
                    state[k, g] = np.uint64(0xFFFFFFFFFFFFFFFF)
    
    # Apply network
    for i, j in network:
        mask = state[i] & ~state[j]
        state[i] &= ~mask
        state[j] |= mask
    
    # Check sortedness
    for k in range(N - 1):
        viol = state[k] & ~state[k + 1]
        if np.any(viol != 0):
            return -1  # Unsorted
    
    return 0

def _local_search_remove(network, time_limit):
    """
    Aggressively remove redundant comparators.
    Returns the shortest valid network found within time_limit.
    """
    start_time = time.time()
    best = list(network)
    random.seed(12345)
    
    # First: systematic single removal (greedy)
    improved = True
    while improved and (time.time() - start_time < time_limit - 2):
        improved = False
        for idx in range(len(best)):
            if time.time() - start_time > time_limit - 2:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_fast(candidate) == 0:
                best = candidate
                improved = True
                break
    
    # Second: random removals and swaps
    while time.time() - start_time < time_limit - 1:
        if len(best) <= 58:
            break
        action = random.random()
        if action < 0.5:
            # Remove random comparator
            idx = random.randint(0, len(best) - 1)
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_fast(candidate) == 0:
                best = candidate
        elif action < 0.7:
            # Swap adjacent
            if len(best) >= 2:
                idx = random.randint(0, len(best) - 2)
                candidate = best[:idx] + [best[idx+1], best[idx]] + best[idx+2:]
                if _verify_network_fast(candidate) == 0:
                    best = candidate
        else:
            # Replace a comparator
            idx = random.randint(0, len(best) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = best[:idx] + [(i, j)] + best[idx+1:]
            if _verify_network_fast(candidate) == 0:
                best = candidate
    
    return best

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin
    
    # Start with odd-even merge sort (72 comparators)
    # Actually, let's use bitonic sort first, it's more reliable
    best_net = _generate_bitonic_network(N)
    
    # Verify
    if _verify_network_fast(best_net) != 0:
        # Fallback
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Try odd-even merge sort as well
    try:
        oem_net = _generate_odd_even_merge_network(N)
        if _verify_network_fast(oem_net) == 0 and len(oem_net) < len(best_net):
            best_net = oem_net
    except:
        pass
    
    # Local search to reduce comparator count
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        best_net = _local_search_remove(best_net, remaining)
    
    # Final verification
    if _verify_network_fast(best_net) != 0:
        # Fallback to bitonic
        best_net = _generate_bitonic_network(N)
    
    return best_net

# EVOLVE-BLOCK-END
