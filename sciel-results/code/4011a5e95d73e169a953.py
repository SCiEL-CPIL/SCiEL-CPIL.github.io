"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import sys

def synthesize(matrix, depth_limit):
    try:
        A = np.array(matrix, dtype=int)
    except Exception:
        A = np.array(matrix)
        A = (A != 0).astype(int)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    m, n = A.shape
    
    # Convert rows to frozensets of indices for memoization
    rows = []
    row_indices = []
    for i in range(m):
        row = A[i]
        ones = np.where(row == 1)[0]
        s = frozenset(int(x) for x in ones)
        rows.append(s)
        row_indices.append(i)
    
    # Sort rows by weight (number of 1s) to process simpler ones first
    # This helps populate memo with small sub-expressions
    sorted_indices = sorted(row_indices, key=lambda i: len(rows[i]))
    
    gates = []
    next_wire = [n + 1]
    memo = {}
    
    # Initialize memo with inputs and zero
    for i in range(n):
        memo[frozenset([i])] = i
    memo[frozenset()] = n
    
    def get_candidate_splits(s):
        """
        Yield candidate splits (sub, comp) for set s.
        """
        size = len(s)
        if size <= 10:
            # Try all proper non-empty subsets
            lst = list(s)
            for mask in range(1, (1 << size) - 1):
                sub = frozenset(lst[i] for i in range(size) if mask & (1 << i))
                comp = s ^ sub
                # Avoid duplicates: only consider if sub is "smaller" or equal in some canonical way
                # Or just yield all and let the caller handle duplicates?
                # To save time, we can enforce len(sub) <= len(comp)
                if len(sub) <= len(comp):
                    yield (sub, comp)
        else:
            # Heuristic splits for large sets
            lst = list(s)
            # 1. Balanced split
            mid = size // 2
            sub1 = frozenset(lst[:mid])
            sub2 = frozenset(lst[mid:])
            yield (sub1, sub2)
            if sub1 != sub2:
                yield (sub2, sub1)
            
            # 2. Split by first element
            first = lst[0]
            sub_a = frozenset([first])
            sub_b = s ^ sub_a
            yield (sub_a, sub_b)
            yield (sub_b, sub_a)
            
            # 3. Split by last element
            last = lst[-1]
            sub_c = frozenset([last])
            sub_d = s ^ sub_c
            yield (sub_c, sub_d)
            yield (sub_d, sub_c)
            
            # 4. Try a few more random-ish splits based on index
            # Split into 1/3 and 2/3
            t1 = size // 3
            t2 = (2 * size) // 3
            if t1 > 0 and t2 < size:
                sub_e = frozenset(lst[:t1])
                sub_f = frozenset(lst[t1:])
                yield (sub_e, sub_f)
                sub_g = frozenset(lst[:t2])
                sub_h = frozenset(lst[t2:])
                yield (sub_g, sub_h)

    def estimate_cost(s):
        """
        Estimate the cost to synthesize s.
        """
        if s in memo:
            return 0
        return len(s) - 1

    @lru_cache(maxsize=None)
    def solve(s):
        """
        Returns the wire index for the linear form represented by frozenset s.
        Creates gates as needed.
        """
        if s in memo:
            return memo[s]
        
        size = len(s)
        if size == 0:
            return n
        if size == 1:
            return list(s)[0]
        
        best_cost = float('inf')
        best_split = None
        
        # Generate candidates
        candidates = list(get_candidate_splits(s))
        
        for sub, comp in candidates:
            # Ensure non-empty
            if not sub or not comp:
                continue
                
            c = 1 + estimate_cost(sub) + estimate_cost(comp)
            
            if c < best_cost:
                best_cost = c
                best_split = (sub, comp)
                
        if best_split is None:
            # Fallback: should not happen for size > 1
            lst = list(s)
            best_split = (frozenset([lst[0]]), s ^ frozenset([lst[0]]))
            
        sub, comp = best_split
        w1 = solve(sub)
        w2 = solve(comp)
        
        w_new = next_wire[0]
        next_wire[0] += 1
        gates.append([w1, w2])
        memo[s] = w_new
        return w_new

    outputs = []
    for idx in sorted_indices:
        r = rows[idx]
        w = solve(r)
        outputs.append(w)
        
    # Re-order outputs to match original matrix row order
    final_outputs = [0] * m
    for i, idx in enumerate(sorted_indices):
        final_outputs[idx] = outputs[i]
        
    return {'gates': gates, 'outputs': final_outputs}
# EVOLVE-BLOCK-END
