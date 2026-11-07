"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    divisors = DIVISORS
    n = len(divisors)
    L = reduce(_lcm, divisors)
    
    # Bipartite matching helper
    # Left: divisors (0..n-1), Right: candidate values
    def has_matching(candidates):
        """
        candidates: list of lists. candidates[i] is list of values for divisor i.
        Returns True if there is a system of distinct representatives.
        """
        # Map values to indices for the right side
        all_vals = set()
        for c in candidates:
            all_vals.update(c)
        if len(all_vals) < n:
            return False
            
        val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
        m = len(all_vals)
        
        # match_to[j] = i (left node matched to right node j)
        match_to = [-1] * m
        # match_from[i] = j (right node matched to left node i)
        match_from = [-1] * n
        
        def try_kun(u, seen):
            for v_val in candidates[u]:
                v_idx = val_to_idx[v_val]
                if seen[v_idx]:
                    continue
                seen[v_idx] = True
                if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                    match_to[v_idx] = u
                    match_from[u] = v_idx
                    return True
            return False
            
        for i in range(n):
            seen = [False] * m
            if not try_kun(i, seen):
                return False
        return True

    def get_assignment(candidates):
        """
        Returns the list of values if matching exists, else None.
        Re-runs matching to get the actual values.
        """
        all_vals = set()
        for c in candidates:
            all_vals.update(c)
        if len(all_vals) < n:
            return None
            
        val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
        idx_to_val = {i: v for v, i in val_to_idx.items()}
        m = len(all_vals)
        
        match_to = [-1] * m
        match_from = [-1] * n
        
        def try_kun(u, seen):
            for v_val in candidates[u]:
                v_idx = val_to_idx[v_val]
                if seen[v_idx]:
                    continue
                seen[v_idx] = True
                if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                    match_to[v_idx] = u
                    match_from[u] = v_idx
                    return True
            return False
            
        for i in range(n):
            seen = [False] * m
            if not try_kun(i, seen):
                return None
                
        # Extract values
        result = [0] * n
        for i in range(n):
            result[i] = idx_to_val[match_from[i]]
        return result

    # Search for minimal W
    # Minimum span for 10 distinct integers is 9
    for W in range(9, 500):  # Cap W at 500, should be found much earlier
        if time.time() > deadline:
            break
            
        # Iterate X from 0 to L-1
        # Optimization: We can step X, but 30k steps is fast.
        for X in range(L):
            if time.time() > deadline:
                break
                
            candidates = []
            valid = True
            for i in range(n):
                d = divisors[i]
                # Find multiples of d in [X, X+W]
                # First multiple >= X
                if X % d == 0:
                    first = X
                else:
                    first = X + (d - X % d)
                
                if first > X + W:
                    valid = False
                    break
                    
                cands = []
                v = first
                while v <= X + W:
                    cands.append(v)
                    v += d
                    
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
                
            if not valid:
                continue
                
            # Check for SDR
            if has_matching(candidates):
                # Get the actual assignment
                assignment = get_assignment(candidates)
                if assignment:
                    # Ensure types are int
                    return [int(v) for v in assignment]
            
    # Fallback if no solution found within time or W limit
    # This should not happen, but return divisors themselves as valid (though span might be large)
    return [int(d) for d in divisors]
# EVOLVE-BLOCK-END
