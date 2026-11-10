"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of {1, ..., 1000} such that every integer has at most
    two representations a + b with a <= b from the set (B2[2] sets).
    
    Strategy:
    1. Start with a dense random initialization.
    2. Use local search: try to add elements that don't violate the constraint,
       and remove elements that cause violations.
    3. Use NumPy for efficient sum representation counting.
    """
    N = 1000
    max_sums = 2 * N  # max possible sum is 2000
    
    def count_representations(arr):
        """
        Count the number of representations a + b = s with a <= b for each s.
        arr is a boolean array of length N+1 (index 0 unused).
        Returns an array of length 2*N+1 where counts[s] is the number of pairs.
        """
        # Use convolution to count all pairs (a, b) with a, b in set
        # conv = arr * arr gives counts of a+b for all ordered pairs
        # But we need a <= b, so:
        # For a < b: counted twice in convolution (a,b) and (b,a)
        # For a = b: counted once
        # So: representations(s) = (conv[s] + conv_diag[s]) // 2
        # where conv_diag[s] is 1 if s/2 is in set and s is even, else 0
        
        conv = np.convolve(arr, arr, mode='full')
        # conv[s] is the number of ordered pairs (a, b) with a + b = s
        # For a <= b:
        #   if s is even and s/2 in set: a=b=s/2 contributes 1
        #   otherwise: a < b contributes (conv[s] - 1) / 2 ... wait
        
        # Actually:
        # Let A be the set. 
        # Number of pairs (a, b) with a + b = s, a, b in A, a <= b:
        # = (number of ordered pairs (a,b) with a+b=s) + (number of pairs with a=b)
        #   all divided by 2? No.
        # 
        # Let c(s) = number of ordered pairs (a, b) in A x A with a + b = s.
        # Let d(s) = 1 if s is even and s/2 in A, else 0.
        # Then number of unordered pairs with a <= b is (c(s) + d(s)) // 2.
        
        reps = np.zeros(2 * N + 1, dtype=np.int32)
        for s in range(2, 2 * N + 1):
            c = conv[s]
            d = 1 if (s % 2 == 0 and s // 2 <= N and arr[s // 2]) else 0
            reps[s] = (c + d) // 2
        return reps
    
    def is_valid(arr):
        """Check if all sums have at most 2 representations."""
        reps = count_representations(arr)
        return np.max(reps) <= 2
    
    # Greedy construction first to get a good starting point
    def greedy_construct():
        chosen = []
        counts = {}
        for x in range(1, N + 1):
            # Check if adding x would violate the constraint
            new_sums = [x + y for y in chosen] + [2 * x]
            valid = True
            for s in new_sums:
                cnt = counts.get(s, 0) + 1
                if cnt > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        return chosen
    
    # Start with greedy
    chosen = greedy_construct()
    
    # Convert to boolean array
    arr = np.zeros(N + 1, dtype=np.int8)
    for x in chosen:
        arr[x] = 1
    
    # Now try to improve by adding more elements
    # Try adding each element not in the set
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if arr[x]:
                continue
            # Try adding x
            arr[x] = 1
            if is_valid(arr):
                improved = True
            else:
                arr[x] = 0
    
    # Now try local search: swap elements
    # Try removing one element and adding another
    for _ in range(50):  # number of passes
        improved = False
        # Try all pairs of removals and additions
        for x in range(1, N + 1):
            if not arr[x]:
                continue
            for y in range(1, N + 1):
                if arr[y] or y == x:
                    continue
                # Try swapping x with y
                arr[x] = 0
                arr[y] = 1
                if is_valid(arr):
                    # Keep the swap if it doesn't reduce size (it doesn't, size stays same)
                    # But we want to maximize size, so swaps don't help directly
                    # However, they might enable future additions
                    improved = True
                else:
                    arr[y] = 0
                    arr[x] = 1
                if improved:
                    break
            if improved:
                break
    
    # Convert back to list
    result = [i for i in range(1, N + 1) if arr[i]]
    return result

# EVOLVE-BLOCK-END
