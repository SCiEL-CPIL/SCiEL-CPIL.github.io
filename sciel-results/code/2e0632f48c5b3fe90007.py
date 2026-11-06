"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _batcher_sort(n):
    """Return Batcher odd-merge sorting network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    m = n // 2
    left = _batcher_sort(m)
    right = _batcher_sort(n - m)
    
    # Shift right network by m
    right_shifted = [(i + m, j + m) for (i, j) in right]
    
    # Merge: odd-even merge of two sorted lists of sizes m and n-m
    merged = _odd_even_merge(m, n - m)
    
    return left + right_shifted + merged

def _odd_even_merge(a, b):
    """Odd-even merge two sorted sequences of lengths a and b (a <= b)."""
    if a == 0:
        return []
    if a == 1 and b == 1:
        return [(0, 1)]
    if a == 1:
        # Merge single element into sorted list of length b
        # The single element is at position 0, list at 1..b
        # We need to insert it - this requires b comparators in worst case
        # But for odd-even merge structure:
        # Split the b elements into even and odd positions
        # Actually for a=1, b=k: compare (0,1), (2,3), ..., then recurse
        # Standard odd-even merge for (1, b):
        # Compare positions (0,1), (2,3), ..., (b-2, b-1) [if b even]
        # Then recursively merge
        result = []
        # First phase: compare (0,1), (2,3), ...
        for i in range(0, b, 2):
            if i + 1 < b + 1:  # i+1 < total length
                result.append((i, i + 1))
        # Now we have two sorted sublists
        # Actually this is getting complex. Let me use a cleaner recursive approach.
        # For odd-even merge of (a, b) where a <= b:
        # Split each into even and odd indexed elements
        # Recursively merge the evens and odds
        # Then compare adjacent pairs
        
        # Let me restart with a cleaner implementation
        pass
    
    # Standard odd-even merge algorithm
    # Inputs: two sorted lists of length a and b (a <= b), interleaved
    # Step 1: Split each list into even and odd indexed elements
    # Step 2: Recursively merge the even-indexed elements
    # Step 3: Recursively merge the odd-indexed elements  
    # Step 4: Compare-exchange adjacent pairs (0,1), (2,3), ...
    
    if a == 0:
        return []
    if a == 1 and b == 1:
        return [(0, 1)]
    
    # For the general case, we interleave the two lists
    # List A has elements at positions 0, 2, 4, ... (a elements)
    # List B has elements at positions 1, 3, 5, ... (b elements)
    # Wait, that's not right either.
    
    # Let me use a different, well-known formulation:
    # Odd-even merge of two sorted sequences X[0..a-1] and Y[0..b-1]
    # where a <= b, total length = a + b
    # 
    # The merged sequence is formed by:
    # 1. Split X into X_even and X_odd
    # 2. Split Y into Y_even and Y_odd  
    # 3. Recursively merge X_even and Y_even -> Z_even
    # 4. Recursively merge X_odd and Y_odd -> Z_odd
    # 5. Interleave Z_even and Z_odd, then compare-exchange adjacent pairs
    
    # But we need to track wire positions carefully.
    # 
    # Alternative: use the standard Batcher network construction
    # which I'll implement more carefully.
    
    # For simplicity and correctness, let me use a known-good approach:
    # Generate the network by simulating what Batcher's algorithm does
    # on indices.
    
    def _merge_indices(a, b):
        """Return list of (i,j) comparators to merge two sorted halves.
        First half is wires 0..a-1, second half is wires a..a+b-1.
        Both halves are assumed sorted. Returns comparators to fully sort."""
        if a == 0 or b == 0:
            return []
        if a == 1 and b == 1:
            return [(0, 1)]
        
        # Odd-even merge:
        # The two sorted lists are at positions [0, a) and [a, a+b)
        # We interleave them conceptually and apply odd-even merge
        
        # Split first list into evens and odds (by index within the list)
        # Split second list into evens and odds
        # Recursively merge evens, recursively merge odds
        # Then do final compare-exchanges
        
        a_evens = (a + 1) // 2
        a_odds = a // 2
        b_evens = (b + 1) // 2
        b_odds = b // 2
        
        # The evens of the first list are at positions 0, 2, 4, ... < a
        # The odds of the first list are at positions 1, 3, 5, ... < a
        # Similarly for the second list starting at position a
        
        # After recursive merges, we need to interleave and compare
        # This is getting complex with position tracking.
        
        # Let me use a simpler but correct approach:
        # Just use a known compact construction.
        
        # Actually, let me just use a straightforward insertion-sort-like 
        # network for the merge, which is correct but not optimal.
        # For n=17 total, this should still be much better than 136.
        
        # Merge by inserting each element of the smaller list into the larger
        # This gives O(a*b) comparators for the merge
        
        result = []
        # We have sorted [0, a) and [a, a+b)
        # Insert elements from [0,a) into [a, a+b)
        # Actually, let's just do a simple merge network:
        # For each pair (i, j) where i < a <= j, we need enough 
        # comparators to ensure sorting.
        
        # Simplest correct merge: use a "shuffle" network
        # Compare (i, a+i) for i in range(min(a,b))
        # Then recursively sort the result... 
        # This is getting complicated.
        
        # Let me just use a known result: for merging two sorted lists 
        # of total length n, we need at most n*log2(n) - n + 1 comparators
        # but a simple construction uses more.
        
        # Practical approach: use a "brick wall" merge
        # Compare all pairs (i, j) where i < j and they're in different halves
        # This is O(a*b) but correct.
        
        for i in range(a):
            for j in range(a, a + b):
                if i < j:
                    result.append((i, j))
        
        return result
    
    return _merge_indices(a, b)

def construct():
    """Construct a sorting network for N=17 wires."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Use Batcher's odd-merge sort network
    # For n=17: split 8+9, sort each, merge
    net = _batcher_sort(N)
    
    # Remove duplicate comparators and sort for determinism
    net = list(set(net))
    net.sort()
    
    # Verify correctness on all 2^17 Boolean inputs
    def verify(network):
        for mask in range(1 << N):
            # Create input: 1s where bit is set
            wires = [(mask >> i) & 1 for i in range(N)]
            # Apply network
            for (i, j) in network:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            # Check sorted: non-decreasing
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    if verify(net):
        return net
    
    # Fallback: odd-even transposition
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
