import time
import random
import numpy as np
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For k=5, g=9, we attempt to find a cyclic Cayley graph.
    # If not found, fallback to LUW.
    
    if k == 5 and g == 9:
        # Try to find a cyclic Cayley graph with n < 1250
        # Known best is 1116, LUW is 1250.
        # We search n in [1116, 1249]
        
        best_n = 1250 # Default to LUW size
        best_edges = None
        
        # We will try to find a valid cyclic graph
        # Iterate n from 1116 to 1249
        # Step 2 to keep n even (required for n/2 to be integer in Z_n)
        
        # To save time, we don't try all n. We try a subset.
        # Or we try all but with limited trials per n.
        
        for n in range(1116, 1250, 2):
            if time.time() > deadline:
                break
                
            # Try a few random pairs (s1, s2)
            # s1, s2 in [1, n-1], s1 != n/2, s2 != n/2
            # s1 != s2, s1 != -s2
            
            trials = 5
            for _ in range(trials):
                if time.time() > deadline:
                    break
                    
                # Heuristic: pick s1, s2 from a range that avoids small sums
                # e.g., n/4 < s < 3n/4
                low = n // 4
                high = 3 * n // 4
                
                # Exclude n/2
                valid_range = list(range(low, high))
                if n // 2 in valid_range:
                    valid_range.remove(n // 2)
                    
                if len(valid_range) < 2:
                    continue
                    
                s1, s2 = random.sample(valid_range, 2)
                
                # Ensure s2 != -s1
                if s2 == (-s1) % n:
                    continue
                    
                S = [n // 2, s1, (-s1) % n, s2, (-s2) % n]
                # Ensure unique
                if len(set(S)) != 5:
                    continue
                
                # Check girth using NumPy
                if _check_girth_numpy(n, S, g):
                    # Construct edges
                    edges = []
                    for u in range(n):
                        for s in S:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
        # If no cyclic graph found, return LUW
        return _luw_construct(k, g)
        
    # Fallback for other k, g
    return _luw_construct(k, g)

def _check_girth_numpy(n, S, g):
    # Check for cycles of length < g
    # S is a list of 5 integers mod n
    
    # Precompute S as numpy array
    S_arr = np.array(S)
    
    # We check lengths 3 to g-1
    # For efficiency, we can stop early if we find a cycle
    
    for L in range(3, g):
        # Generate all sequences of length L
        # This is 5^L. For L=8, 390625.
        # We can use itertools.product
        
        # Optimization: For large L, this is slow.
        # We can use a BFS approach instead?
        # But NumPy vectorization on sums is fast.
        
        # Let's try to generate the sums directly.
        # We don't need to store all sequences, just the sums?
        # No, we need to check if the path is simple (partial sums distinct).
        # So we need the sequences.
        
        # For L=8, 390k sequences.
        # Creating a list of 390k tuples is ~0.1s?
        # Converting to Numpy array is ~0.1s?
        # Summing is ~0.01s?
        
        # Let's do it.
        
        # If L is large, we might run out of memory or time.
        # 5^8 = 390625.
        # 5^7 = 78125.
        
        # We can break early if we find a cycle.
        
        # Use itertools.product
        # To speed up, we can generate the array in chunks or use a recursive generator?
        # itertools.product is implemented in C, so it's fast.
        
        seqs = list(product(S_arr, repeat=L))
        if not seqs:
            continue
            
        A = np.array(seqs)
        # A is shape (N, L)
        
        # Compute cumulative sums
        # partial_sums = np.cumsum(A, axis=1) % n
        # final_sums = partial_sums[:, -1]
        
        # We only care if final_sum == 0
        # So we can compute full sums first
        sums = A.sum(axis=1) % n
        zero_indices = np.where(sums == 0)[0]
        
        if len(zero_indices) == 0:
            continue
            
        # For those with sum 0, check if partial sums are distinct
        # Extract the sequences
        A_zero = A[zero_indices]
        partial_sums = np.cumsum(A_zero, axis=1) % n
        
        # Check for duplicates in each row
        # A row is a valid simple cycle if all partial sums are distinct.
        # Note: partial_sums[0] is the first step.
        # We need to check if partial_sums[i] == partial_sums[j] for i != j.
        # Also, the start vertex is 0. The end is 0.
        # So we check if any partial sum is 0 (before the end)?
        # If partial_sum[k] == 0 for k < L-1, then we have a cycle of length k+1 < L.
        # But we are checking for cycles of length < g.
        # If we find a cycle of length L, it's invalid.
        # If we find a cycle of length k < L, it would have been caught in the previous iteration.
        # So we just need to check if the path of length L is simple.
        # Simple means: v_0=0, v_1, ..., v_L=0.
        # v_i must be distinct for i=0..L-1.
        
        # partial_sums contains v_1, ..., v_L.
        # We need v_0..v_{L-1} to be distinct.
        # v_0 = 0.
        # So we need 0 not in partial_sums[0:L-1].
        # And partial_sums[0:L-1] to be distinct.
        
        # Let's check:
        # 1. Is 0 in partial_sums[:, :-1]?
        # 2. Are the values in partial_sums[:, :-1] distinct?
        
        # Check 1:
        has_zero = (partial_sums[:, :-1] == 0).any(axis=1)
        
        # Check 2:
        # We can use np.unique or sort.
        # For small L, we can just check pairwise?
        # Or use a set in a Python loop for the few candidates.
        
        # Since len(zero_indices) is likely small, we can loop in Python.
        for i in range(len(zero_indices)):
            ps = partial_sums[i]
            # Check if any ps[j] == 0 for j < L-1
            if (ps[:-1] == 0).any():
                return False
            # Check if any ps[j] == ps[k] for j != k
            # Convert to list and check set size
            if len(set(ps.tolist())) < L:
                return False
                
    return True

def _luw_construct(k, g):
    # Original LUW implementation
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
