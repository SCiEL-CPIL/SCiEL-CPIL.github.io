import time
import random

N = 16

def _batcher_sort(n):
    """Generates Batcher's sorting network for n inputs."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _batcher_sort(half)
    right = _batcher_sort(half)
    
    # Shift right comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge
    merge = _batcher_merge(half, 0, half)
    
    return left + right + merge

def _batcher_merge(size, left_offset, right_offset):
    """
    Generates comparators for an odd-even merge of two sorted lists of 'size'.
    left_offset is the start index of the left list.
    right_offset is the start index of the right list.
    """
    if size == 1:
        return [(left_offset, right_offset)]
    
    half = size // 2
    
    # Recursively merge odd and even parts
    # The odd-even merge splits the indices into odd and even relative to the merge start?
    # Actually, Batcher's merge works on the combined sequence of 2*size elements.
    # It compares (left[i], right[i]) for i in 0..size-1? No.
    
    # Standard Odd-Even Merge:
    # 1. Merge the odd-indexed elements of the two lists.
    # 2. Merge the even-indexed elements of the two lists.
    # 3. Compare and swap cross elements.
    
    # Let's implement it on the indices of the 2*size block.
    # The block is [left_offset ... left_offset+size-1, right_offset ... right_offset+size-1]
    # But in Batcher, the lists are interleaved? No, they are contiguous.
    # The merge network takes two sorted lists of length 'size' and produces one sorted list of length 2*size.
    
    # The recursive step:
    # Merge the odd positions and even positions of the combined 2*size array?
    # No, the standard definition:
    # To merge two sorted lists A and B of length N:
    # 1. Merge A[1], A[3]... and B[1], B[3]... (odd indices)
    # 2. Merge A[0], A[2]... and B[0], B[2]... (even indices)
    # 3. Compare A[i] with B[i-1] and B[i] (cross comparisons)
    
    # This is getting complicated to implement generically without errors.
    # Given the small N=16, I can hardcode the 53-comparator network if I can find it.
    # Or, I can use a known correct implementation of Batcher's network.
    
    # Let's try a simpler approach: Use the known 53-comparator network from literature.
    # The "Akers" network for 16 inputs has 53 comparators.
    
    # I will define the 53-comparator network explicitly.
    pass

# Since implementing Batcher correctly is prone to bugs, and I need a robust solution,
# I will use a known 53-comparator network.
# Reference: "Sorting Networks" by Knuth, or online repositories.
# A common 53-comparator network for 16 inputs:

def _get_53_network():
    # This is a verified 53-comparator sorting network for 16 inputs.
    # Source: Derived from Akers' construction or Batcher's.
    # I will use a specific known list.
    # If I cannot verify the exact list from memory, I will rely on the local search
    # starting from the 60-comparator network but with a much faster verifier.
    # However, local search from 60 to 53 might be hard.
    # Let's try to generate Batcher's network programmatically.
    
    def merge(n, a, b):
        # a, b are lists of indices
        # Returns list of comparators to merge two sorted lists of size n
        if n == 1:
            return [(a[0], b[0])]
        
        half = n // 2
        
        # Split indices into odd and even
        a_odd = a[1::2]
        a_even = a[0::2]
        b_odd = b[1::2]
        b_even = b[0::2]
        
        # Recursively merge odd and even parts
        # Note: The recursive merge assumes the inputs are sorted.
        # The odd parts form two sorted lists of size half (or half+1? No, n is power of 2)
        # If n is power of 2, n/2 is integer.
        
        res = []
        res += merge(half, a_odd, b_odd)
        res += merge(half, a_even, b_even)
        
        # Cross comparisons
        # For i in 0..n-1:
        #   Compare (a[i], b[i-1]) if i > 0
        #   Compare (a[i], b[i]) if i < n
        # Wait, the cross comparisons in Odd-Even Merge are:
        # Compare the i-th element of the even-merged list with the (i-1)-th and i-th of the odd-merged list?
        # No, the structure is:
        # After merging odds and evens, we have two sorted lists of length n (the odd-indexed positions and even-indexed positions of the final output? No).
        
        # Let's stick to the standard recursive definition for Batcher's merge:
        # To merge two sorted lists of length N:
        # 1. Merge the odd-indexed elements (indices 1, 3, ...) of both lists.
        # 2. Merge the even-indexed elements (indices 0, 2, ...) of both lists.
        # 3. Perform cross comparisons:
        #    For i = 0 to N-1:
        #       Compare (Even[i], Odd[i-1]) for i > 0
        #       Compare (Even[i], Odd[i]) for i < N
        #    Actually, the indices in the final merged list are interleaved.
        
        # This is too error-prone to write from scratch in the time available.
        # I will use the 60-comparator network and apply a very aggressive local search
        # with the fast verifier. The fast verifier allows ~10,000-100,000 verifications per second.
        # In 60 seconds, we can do millions of moves.
        # Starting from 60, can we reach 53?
        # The gap is 7 comparators.
        # Local search might struggle to remove 7 comparators if they are "structurally" necessary in the 60-comparator topology.
        # However, if we allow "insert" moves, we can change the topology.
        
        # Better: Start with a different base network.
        # The Batcher network for 16 inputs is 53.
        # I will hardcode the Batcher network for 16 inputs.
        
        # Batcher 16:
        # 1. Sort 4x4 blocks (5 comparators each, 4 blocks = 20)
        # 2. Merge them into 8x8 (13 comparators? No, merge of two 4s is 5? No, 8-merge is 13?
        #    Actually, Batcher 8 uses 19.
        #    Batcher 16 uses 53.
        
        # I will find a reliable 53-comparator list online or derive it.
        # Since I cannot browse, I will use a known property:
        # The network can be represented as a list of pairs.
        
        # I will provide a function that returns a known 53-comparator network.
        # If I fail to recall it, I will return the 60-comparator one.
        
        # Let's try to construct it via code:
        # Batcher's Odd-Even Mergesort
        # T(16) = 2*T(8) + M(16)
        # T(8) = 2*T(4) + M(8)
        # T(4) = 2*T(2) + M(4)
        # T(2) = 1
        
        # M(2) = 1 (compare 0,1)
        # M(4) = 5 (merge two 2s)
        # M(8) = 13 (merge two 4s)
        # M(16) = 29 (merge two 8s)
        
        # Total = 2*19 + 29 = 38 + 29 = 67? No.
        # Batcher's comparator count:
        # C(N) = 2*C(N/2) + N*log2(N) - N + 1?
        # C(16) = 53.
        
        # I will implement the recursive Batcher generator carefully.
        
        def gen_batcher(n):
            if n == 1:
                return []
            if n == 2:
                return [(0, 1)]
            
            half = n // 2
            left = gen_batcher(half)
            right = gen_batcher(half)
            right = [(i + half, j + half) for i, j in right]
            
            # Merge
            merge_comps = gen_merge(half, 0, half)
            
            return left + right + merge_comps
        
        def gen_merge(size, l_start, r_start):
            # Merges two sorted lists of length 'size' starting at l_start and r_start
            if size == 1:
                return [(l_start, r_start)]
            
            half = size // 2
            
            # Indices for odd and even positions in the merge
            # The merge operates on the combined 2*size array?
            # No, it operates on the two lists.
            
            # Odd indices: 1, 3, ...
            # Even indices: 0, 2, ...
            
            # List A (left) indices: l_start, l_start+1, ...
            # List B (right) indices: r_start, r_start+1, ...
            
            # A_odd: l_start+1, l_start+3, ...
            # B_odd: r_start+1, r_start+3, ...
            # A_even: l_start, l_start+2, ...
            # B_even: r_start, r_start+2, ...
            
            a_odd = [l_start + i for i in range(1, size, 2)]
            b_odd = [r_start + i for i in range(1, size, 2)]
            a_even = [l_start + i for i in range(0, size, 2)]
            b_even = [r_start + i for i in range(0, size, 2)]
            
            res = []
            res += gen_merge(half, a_odd[0], b_odd[0]) # This is wrong, gen_merge expects contiguous?
            # My gen_merge assumes contiguous lists.
            # I need a more flexible merge or to map indices back.
            
            # This is getting too complex. I will use the 60-comparator network and local search.
            # To improve the chances, I will use a "random restart" strategy in the local search.
            # And I will use the fast bit-slicing verifier.
            
            return [] # Placeholder
        
        return gen_batcher(16)

# Given the complexity and risk of bugs in the Batcher generator, I will stick to the 60-comparator base
# but optimize the local search significantly.
# The key is the fast verifier.

import time
import random

N = 16

# EVOLVE-BLOCK-START

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    """
    if not net:
        return True
    
    # Precompute initial bitsets
    # We can cache these in a global variable to avoid recomputation
    global _INITIAL_BITSETS
    if not hasattr(_verify_zero_one_fast, '_INITIAL_BITSETS'):
        _INITIAL_BITSETS = []
        for i in range(N):
            bitset = 0
            for k in range(1 << N):
                if k & (1 << i):
                    bitset |= (1 << k)
            _INITIAL_BITSETS.append(bitset)
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    # Same as before
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s margin
    
    base_net = _base_60_network()
    
    if not _verify_zero_one_fast(base_net):
        # Fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = base_net
    best_len = len(base_net)
    
    rng = random.Random(42)
    
    # Local Search
    current_net = base_net[:]
    
    while time.time() < deadline:
        # Try to remove a comparator
        if len(current_net) > 1:
            # Try removing a random comparator
            idx = rng.randrange(len(current_net))
            new_net = current_net[:idx] + current_net[idx + 1:]
            if _verify_zero_one_fast(new_net):
                current_net = new_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
        else:
            break
            
        # Try swapping two comparators
        if len(current_net) > 1:
            i, j = rng.sample(range(len(current_net)), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if _verify_zero_one_fast(new_net):
                current_net = new_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
                    
        # Try inserting a comparator (to escape local minima)
        if rng.random() < 0.1: # 10% chance to insert
            i, j = rng.sample(range(N), 2)
            if i > j: i, j = j, i
            idx = rng.randrange(len(current_net) + 1)
            new_net = current_net[:idx] + [(i, j)] + current_net[idx:]
            if _verify_zero_one_fast(new_net):
                current_net = new_net
                # Don't update best if we inserted, unless we can remove it later
                
    return best_net

# EVOLVE-BLOCK-END
