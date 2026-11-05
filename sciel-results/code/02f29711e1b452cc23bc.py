"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

# Precompute smooth numbers up to 128
def _generate_smooth(max_val):
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

SMOOTH = _generate_smooth(128)
SMOOTH_SET = set(SMOOTH)

# Precompute divisibility: divides[i][j] = True if SMOOTH[i] divides SMOOTH[j]
def _precompute_divisibility():
    n = len(SMOOTH)
    divides = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and SMOOTH[j] % SMOOTH[i] == 0:
                divides[i][j] = True
    return divides

DIVIDES = _precompute_divisibility()

def _is_antichain(terms):
    """Check if no term in terms divides another."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _find_min_antichain(n, max_terms=30):
    """Find representation of n as sum of distinct smooth numbers forming an antichain,
    minimizing the number of terms."""
    
    # Try 1 term first
    if n in SMOOTH_SET:
        return [n]
    
    # Try 2 terms
    for i in range(len(SMOOTH)):
        a = SMOOTH[i]
        if a >= n:
            break
        b = n - a
        if b in SMOOTH_SET and b > a and b % a != 0:
            return [a, b]
    
    # Try 3 terms using DFS with pruning
    smooth_le_n = [x for x in SMOOTH if x <= n]
    
    # For 3 terms: a + b + c = n, a < b < c, none divides another
    found = None
    for i in range(len(smooth_le_n)):
        a = smooth_le_n[i]
        if a >= n:
            break
        remaining = n - a
        # Need b + c = remaining, b > a, c > b
        # b ranges from a+1 to remaining//2
        for j in range(i + 1, len(smooth_le_n)):
            b = smooth_le_n[j]
            if b >= remaining:
                break
            c = remaining - b
            if c <= b:
                continue
            if c not in SMOOTH_SET:
                continue
            if c > n:
                continue
            # Check antichain: a doesn't divide b, a doesn't divide c, b doesn't divide c
            if b % a == 0 or c % a == 0 or c % b == 0:
                continue
            return [a, b, c]
    
    # Try 4 terms using DFS
    def dfs_4(remaining, start_idx, chosen, depth):
        """Find terms to sum to remaining, starting from smooth_le_n[start_idx], 
        with len(chosen) + depth more terms needed."""
        if depth == 0:
            if remaining == 0:
                return list(chosen)
            return None
        if depth == 1:
            if remaining in SMOOTH_SET:
                # Check antichain with all chosen
                candidate = chosen + [remaining]
                if _is_antichain(candidate):
                    return candidate
            return None
        
        # Upper bound on how many more we can pick
        for idx in range(start_idx, len(smooth_le_n)):
            val = smooth_le_n[idx]
            if val > remaining:
                break
            # Pruning: if we pick val, the remaining must be achievable with (depth-1) terms
            # each >= next smooth number after val
            chosen.append(val)
            result = dfs_4(remaining - val, idx + 1, chosen, depth - 1)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    result = dfs_4(n, 0, [], 4)
    if result is not None:
        return result
    
    # Try 5 terms
    def dfs_k(remaining, start_idx, chosen, depth):
        if depth == 0:
            if remaining == 0:
                return list(chosen)
            return None
        if depth == 1:
            if remaining in SMOOTH_SET:
                candidate = chosen + [remaining]
                if _is_antichain(candidate):
                    return candidate
            return None
        
        for idx in range(start_idx, len(smooth_le_n)):
            val = smooth_le_n[idx]
            if val > remaining:
                break
            # Simple pruning: minimum possible sum of 'depth' terms starting from idx+1
            # is just the next 'depth' smallest smooth numbers > val
            # But that's expensive; use a simpler bound
            chosen.append(val)
            result = dfs_k(remaining - val, idx + 1, chosen, depth - 1)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    result = dfs_k(n, 0, [], 5)
    if result is not None:
        return result
    
    # Try up to 30 terms with a more systematic approach
    # For larger term counts, use a greedy/iterative approach
    # This is a fallback; hopefully not needed for n <= 128
    
    # Try increasing term counts from 6 up to min(30, n) since all terms >= 1
    max_k = min(30, n)
    for k in range(6, max_k + 1):
        result = dfs_k(n, 0, [], k)
        if result is not None:
            return result
    
    # Should not reach here for n <= 128, but as a safety net
    raise ValueError(f"No antichain representation found for n={n}")

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 second deadline for safety
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Return what we have so far with None for remaining
            # But this should not happen
            while len(answers) < 128:
                answers.append(None)
            return answers
        
        result = _find_min_antichain(n)
        answers.append(result)
    
    return answers
# EVOLVE-BLOCK-END
