import math
import time
from collections import defaultdict

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM_ALL = 1
for d in DIVISORS:
    _LCM_ALL = _lcm(_LCM_ALL, d)

def _find_sdr(domains):
    """
    Find a System of Distinct Representatives for the given list of domains.
    domains[i] is a list of integers.
    Returns a list of chosen integers or None if impossible.
    Uses backtracking. N is small (10).
    """
    n = len(domains)
    assignment = [None] * n
    used = set()
    
    # Sort domains by size to prune earlier
    indexed_domains = [(len(d), i, d) for i, d in enumerate(domains)]
    indexed_domains.sort()
    
    def backtrack(idx):
        if idx == n:
            return True
        _, orig_idx, domain = indexed_domains[idx]
        for val in domain:
            if val not in used:
                used.add(val)
                assignment[orig_idx] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assignment[orig_idx] = None
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    deadline = time.time() + 50.0
    
    # Generate candidates up to LCM
    # This ensures we capture all unique "patterns" of relative positions
    limit = _LCM_ALL
    candidates = []
    for i, d in enumerate(DIVISORS):
        k = 1
        while k * d <= limit:
            candidates.append((k * d, i))
            k += 1
            
    # Sort by value
    candidates.sort(key=lambda x: x[0])
    
    # Sliding window to find smallest span intervals covering all divisors
    # We will collect the top K smallest spans
    K = 200
    best_windows = [] # (span, left_idx, right_idx)
    
    left = 0
    counts = [0] * N
    covered = 0
    
    for right in range(len(candidates)):
        val, div_idx = candidates[right]
        if counts[div_idx] == 0:
            covered += 1
        counts[div_idx] += 1
        
        while covered == N:
            # Current window [left, right] covers all divisors
            span = candidates[right][0] - candidates[left][0]
            
            # Add to heap or list
            # To keep top K smallest, we can use a heap or just append and sort later
            # Given N candidates is ~300k, and windows are many, 
            # but we only care about small spans.
            # We can just append and prune if list gets too big?
            # Or use a max-heap of size K.
            if len(best_windows) < K:
                best_windows.append((span, left, right))
            else:
                # Find max span in best_windows
                max_idx = 0
                max_span = best_windows[0][0]
                for j in range(1, K):
                    if best_windows[j][0] > max_span:
                        max_span = best_windows[j][0]
                        max_idx = j
                if span < max_span:
                    best_windows[max_idx] = (span, left, right)
            
            # Shrink window from left
            l_val, l_div_idx = candidates[left]
            counts[l_div_idx] -= 1
            if counts[l_div_idx] == 0:
                covered -= 1
            left += 1
            
            if time.time() > deadline:
                break
                
    if time.time() > deadline:
        # Fallback
        return DIVISORS[:]

    # Sort best_windows by span
    best_windows.sort(key=lambda x: x[0])
    
    # Try to find a valid SDR in these windows
    for span, l_idx, r_idx in best_windows:
        # Extract domains for this window
        domains = [[] for _ in range(N)]
        for i in range(l_idx, r_idx + 1):
            val, div_idx = candidates[i]
            domains[div_idx].append(val)
            
        result = _find_sdr(domains)
        if result is not None:
            # Verify distinctness and divisibility just in case
            # (Should be guaranteed by _find_sdr and construction)
            if len(set(result)) == N:
                for i, val in enumerate(result):
                    if val % DIVISORS[i] != 0:
                        continue
                return [int(x) for x in result]
                
    # If no valid SDR found in top K windows, fallback to a known valid solution
    # The divisors themselves form a valid solution with span 33
    return DIVISORS[:]

# EVOLVE-BLOCK-END
