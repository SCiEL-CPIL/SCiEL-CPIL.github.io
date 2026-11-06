"""Divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # For each target n, find minimal antichain representation
    # Strategy: enumerate antichains of size k (starting from 1) and check if any sums to n
    
    answers = [None] * 129
    
    # Precompute for each target the minimal number of terms needed
    # Use dynamic programming or direct search with early termination
    
    # For efficiency, precompute all pairs (a, b) with a < b, both smooth, a+b <= 128, and neither divides the other
    valid_pairs = []
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a + b > 128:
                continue
            # Check antichain: neither divides the other
            if b % a != 0 and a % b != 0:  # a%b is always 0 if a<b, so just check b%a
                valid_pairs.append((a, b, a + b))
    
    # Sort by sum for quick lookup
    pair_by_sum = {}
    for a, b, s in valid_pairs:
        if s not in pair_by_sum:
            pair_by_sum[s] = []
        pair_by_sum[s].append((a, b))
    
    # For triples, precompute is harder; do on-the-fly with pruning
    # But first, let's handle 1-term and 2-term cases
    
    for n in range(1, 129):
        if time.time() > deadline - 5:
            break
        
        # Try 1 term
        if n in smooth_set:
            answers[n] = [n]
            continue
        
        # Try 2 terms
        if n in pair_by_sum:
            # Find any valid pair
            for a, b in pair_by_sum[n]:
                answers[n] = [a, b]
                break
            if answers[n] is not None:
                continue
        
        # Try 3 terms: n = a + b + c
        # Enumerate pairs (a, b) with a < b, then c = n - a - b
        # c must be smooth, c > b (or at least distinct), and form antichain
        found = False
        # To minimize, we want the smallest valid triple
        # Iterate over possible a values (smooth, < n)
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            remaining = n - a
            # Now need b + c = remaining, b < c, both smooth, b > a (to avoid duplicates)
            # Actually b just needs to be > a to avoid re-enumerating same set
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if b >= remaining:
                    break
                c = remaining - b
                if c <= b:
                    continue
                if c not in smooth_set:
                    continue
                # Check antichain: no element divides another
                # a, b, c: check all pairs
                # a divides b? a divides c? b divides c?
                if b % a == 0 or c % a == 0 or c % b == 0:
                    continue
                answers[n] = [a, b, c]
                found = True
                break
            if found:
                break
        
        if not found:
            # Try 4+ terms - this is more complex
            # Use a backtracking search with pruning
            # For now, fall back to a general search
            found = _find_antichain_rep(n, smooth, can, n_smooth, max_terms=30)
            if found:
                answers[n] = found
            else:
                # Should not happen for targets 1..128, but handle gracefully
                answers[n] = [n] if n in smooth_set else None
    
    # Verify all answers are valid
    for n in range(1, 129):
        if answers[n] is None:
            # Fallback: use a simple valid representation if exists
            # For smooth n, [n] works
            if n in smooth_set:
                answers[n] = [n]
    
    return answers[1:129]


def _find_antichain_rep(n, smooth, can, n_smooth, max_terms=30):
    """Find an antichain representation of n using smooth numbers."""
    # Backtracking search with pruning
    # Sort smooth numbers
    smooth_sorted = sorted(smooth)
    
    def backtrack(remaining, start_idx, current):
        if remaining == 0:
            if len(current) >= 1:
                return list(current)
            return None
        if remaining < 0:
            return None
        if len(current) >= max_terms:
            return None
        
        for i in range(start_idx, n_smooth):
            val = smooth_sorted[i]
            if val > remaining:
                break
            
            # Check if adding val maintains antichain property
            valid = True
            for existing in current:
                if val % existing == 0 or existing % val == 0:
                    valid = False
                    break
            if not valid:
                continue
            
            current.append(val)
            result = backtrack(remaining - val, i + 1, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    return backtrack(n, 0, [])
# EVOLVE-BLOCK-END
