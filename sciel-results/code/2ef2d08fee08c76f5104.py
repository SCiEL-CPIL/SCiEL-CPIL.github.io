"""Size-efficient sorting networks on 18 wires."""
import time
import itertools
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _build_bitset_wires():
    """Pre-compute the 2^N boolean input wires as Python integers (bitsets)."""
    wires = []
    # We need an integer W_i where bit k is 1 if bit i of k is 1.
    # k ranges from 0 to 2^N - 1.
    # We can build this by iterating or using patterns.
    # Pattern for bit i:
    # Blocks of 2^i zeros, 2^i ones.
    # Example N=3, i=0: 01010101 (binary) -> 0b01010101
    # i=1: 00110011
    # i=2: 00001111
    
    for i in range(N):
        # Construct the bitset for wire i
        # The bit at position k is (k >> i) & 1
        # We can build this by repeating a pattern.
        # Pattern length is 2^(i+1).
        # First 2^i bits are 0, next 2^i bits are 1.
        # This pattern repeats 2^(N - i - 1) times.
        
        # Create the base pattern as an integer
        # 2^i zeros, 2^i ones.
        # Value is (2^(2*i) - 2^i) ? No.
        # Bits 0..2^i-1 are 0. Bits 2^i..2^(i+1)-1 are 1.
        # Mask for ones: ((1 << (2*i)) - 1) ^ ((1 << i) - 1)
        # This is (1 << (2*i)) - (1 << i)
        
        pattern_len = 1 << (i + 1)
        # The pattern value
        # Lower i+1 bits: 0...0 1...1
        # Actually, let's just build it.
        
        # Efficient construction:
        # The bitset for wire i is a periodic sequence of 0s and 1s.
        # We can generate the integer directly.
        
        # Let's use a helper to build the integer.
        # For N=18, 2^18 bits.
        
        # We can use the property:
        # W_i = sum( 1 << k for k in range(2^N) if (k >> i) & 1 )
        
        # Faster:
        # Block size = 2^(i+1)
        # Ones in block = 2^i
        # Number of blocks = 2^(N - i - 1)
        
        # Value of one block of ones at position offset:
        # (1 << 2^i) - 1 is a mask of 2^i ones.
        # Shifted by i: ((1 << 2^i) - 1) << i
        
        block_ones_mask = ((1 << (1 << i)) - 1) << i
        # Wait, 1 << i is the count of bits.
        # If i=0, count=1. Mask=1. Shift=0. Val=1. Pattern 01. Correct.
        # If i=1, count=2. Mask=3 (11). Shift=1. Val=110. Pattern 0011. Correct.
        
        # Now repeat this block 2^(N - i - 1) times.
        # We can do this by shifting and ORing.
        
        w_val = 0
        shift = 0
        num_blocks = 1 << (N - i - 1)
        for _ in range(num_blocks):
            w_val |= block_ones_mask << shift
            shift += pattern_len
            
        wires.append(w_val)
        
    return wires

def _apply_network_bitset(network, wires=None):
    """Apply network to all 2^N boolean inputs using bitsets. Returns True if all sorted."""
    if wires is None:
        wires = _build_bitset_wires()
    
    # Work on a copy
    w = wires[:]
    
    for (a, b) in network:
        wa = w[a]
        wb = w[b]
        # min is AND, max is OR
        w[a] = wa & wb
        w[b] = wa | wb
    
    # Check if sorted: wire[i] <= wire[i+1] for all i
    # Violation if wire[i] has 1 and wire[i+1] has 0.
    # i.e., (w[i] & ~w[i+1]) != 0
    for i in range(N - 1):
        if (w[i] & ~w[i + 1]) != 0:
            return False
    return True

def _verify_network(network):
    """Verify network sorts all boolean inputs. Returns True if correct."""
    return _apply_network_bitset(network)

def _batcher_odd_even_merge_sort(n):
    """Build Batcher's odd-even merge sort network for n wires. Returns list of (i,j)."""
    if n <= 1:
        return []
    
    network = []
    
    def odd_even_merge(k, offset, direction):
        if k == 1:
            return
        half = k // 2
        odd_even_merge(half, offset + 1, direction)
        odd_even_merge(half, offset, direction)
        for i in range(half):
            a = offset + i
            b = offset + half + i
            if a < b:
                network.append((a, b))
            else:
                network.append((b, a))
    
    def odd_even_merge_sort(k, offset):
        if k <= 1:
            return
        half = k // 2
        odd_even_merge_sort(half, offset)
        odd_even_merge_sort(k - half, offset + half)
        odd_even_merge(half, offset, 0)
    
    odd_even_merge_sort(n, 0)
    return network

def _remove_redundant(network, deadline=None):
    """Try to remove comparators that don't affect correctness."""
    if not network:
        return network
    
    current = list(network)
    base_wires = _build_bitset_wires()
    
    changed = True
    while changed:
        changed = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(len(current) - 1, -1, -1):
            candidate = current[:i] + current[i+1:]
            if _apply_network_bitset(candidate, base_wires):
                current = candidate
                changed = True
                break  # restart from beginning since indices changed
    return current

def _local_search_reduce(network, time_limit):
    """Try to reduce network size via local search."""
    current = list(network)
    best = len(current)
    start_time = time.time()
    base_wires = _build_bitset_wires()
    
    # Phase 1: Greedy removal from end to start
    for i in range(len(current) - 1, -1, -1):
        if time.time() - start_time > time_limit * 0.4:
            break
        candidate = current[:i] + current[i+1:]
        if _apply_network_bitset(candidate, base_wires):
            current = candidate
            best = len(current)
    
    # Phase 2: Try swapping adjacent comparators then removing
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        progress = False
        for i in range(len(current) - 1):
            if time.time() - start_time > time_limit:
                break
            candidate = current[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _apply_network_bitset(candidate, base_wires):
                for j in range(len(candidate) - 1, -1, -1):
                    if time.time() - start_time > time_limit:
                        break
                    cand2 = candidate[:j] + candidate[j+1:]
                    if _apply_network_bitset(cand2, base_wires):
                        current = cand2
                        best = len(current)
                        progress = True
                        break
                if progress:
                    break
        if not progress:
            break
            
    return current

def _construct_batcher():
    """Construct Batcher's odd-even merge sort network for N wires."""
    return _batcher_odd_even_merge_sort(N)

def construct():
    deadline = 85.0
    start = time.time()
    
    network = _construct_batcher()
    
    if not _verify_network(network):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5
    if remaining > 10:
        network = _remove_redundant(network, deadline=deadline - 5)
    
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5
    if remaining > 10:
        network = _local_search_reduce(network, remaining)
    
    if not _verify_network(network):
        network = _construct_batcher()
    
    return network

# EVOLVE-BLOCK-END
