"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

_N = 256

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S without violating constraints."""
    n = len(S)
    # S is a list, x is an int
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            # Check distinctness: a, b, x are distinct.
            # S contains distinct elements. x is not in S (checked by caller or assumed).
            # If x is in S, we shouldn't be calling this.
            
            # 1. x | a*b
            if (a * b) % x == 0:
                return False
            # 2. a | x*b
            if (x * b) % a == 0:
                return False
            # 3. b | x*a
            if (x * a) % b == 0:
                return False
    return True

def _conflict_count(S, x):
    """Count how many pairs in S conflict with x."""
    count = 0
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    # 1. Initialize with primes
    S = [a for a in range(2, _N + 1) if _is_prime(a)]
    S_set = set(S)
    
    # 2. Candidates: composites
    candidates = [a for a in range(2, _N + 1) if not _is_prime(a)]
    
    # 3. Greedy Addition
    # We repeatedly pick the candidate with the fewest conflicts.
    # If conflicts > 0, we can't add it. So we only consider candidates with 0 conflicts.
    # Among those, which one to pick?
    # Heuristic: Pick the one that minimizes future conflicts?
    # Or just pick the smallest?
    # Let's try picking the one with the "least impact" on other candidates?
    # Simpler: Just pick the smallest valid candidate first?
    # Or: Calculate conflict count for ALL candidates. 
    # If any have 0, pick the one with min "potential conflict" with other candidates?
    # Let's just pick the smallest 0-conflict candidate.
    
    while time.time() < deadline:
        # Filter candidates that can be added
        valid_candidates = []
        for x in candidates:
            if _can_add(S, x):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
            
        # Pick the best candidate
        # Heuristic 1: Smallest value
        # Heuristic 2: Least conflicts with other *remaining* candidates?
        # Let's try: Pick the candidate that has the fewest conflicts with other candidates that are *also* valid?
        # That's expensive.
        
        # Let's try a different heuristic:
        # Score = number of pairs in S that are "close" to conflicting? No, they are 0 conflicts.
        # Let's just sort by value and pick the first.
        # Or maybe pick the one that allows the most other candidates to remain valid?
        
        # Let's try: For each valid candidate x, count how many other candidates y (y!=x) 
        # would become invalid if x is added?
        # This is O(C^2 * |S|^2). C ~ 200. |S| ~ 60. 200^2 * 3600 = 144M ops. Too slow.
        
        # Let's stick to a simpler heuristic:
        # Pick the candidate with the smallest value.
        # Or, pick the candidate that is "most prime-like" (fewest factors)?
        
        # Let's try picking the candidate that minimizes the number of "near misses" in S?
        # Actually, just picking the smallest is a good baseline.
        
        best_x = min(valid_candidates)
        S.append(best_x)
        S_set.add(best_x)
        candidates.remove(best_x)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # 4. Local Search / Swaps
    # Try to improve by swapping.
    # For each element a in S, remove it, then greedily add as many as possible.
    # If the resulting set is larger, keep it.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        current_size = len(S)
        
        # Try removing each element
        for i, a in enumerate(list(S)):
            if time.time() > deadline - 2.0:
                break
            
            # Remove a
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            # Candidates are all numbers not in S_temp
            # But we can restrict to those that were in candidates + a
            # Actually, just all numbers 2..256 not in S_temp
            # To be fast, use the remaining candidates from the greedy phase + a
            # But some candidates might have been removed in previous swaps?
            # Let's just use all numbers 2..256 not in S_temp.
            
            # Greedy add on S_temp
            # To speed up, only consider numbers that were in the original candidates + a
            # But we might have missed some?
            # Let's just use the global list of composites + primes?
            # No, primes are mostly in S.
            
            # Let's define a pool of candidates for this local step
            # Pool = all x in 2..256 such that x not in S_temp
            # This is O(255 * |S|^2) per swap attempt.
            # 255 * 3600 = ~1M ops per swap.
            # We have ~60 elements in S. 60 * 1M = 60M ops.
            # This might be too slow if we do many iterations.
            
            # Optimization: Only try swapping if we can add at least 2 elements to compensate for removing 1?
            # Or just try to add as many as possible.
            
            # Let's restrict the pool to:
            # 1. The element 'a' we removed
            # 2. The elements from 'candidates' that were not added in the greedy phase
            #    (i.e., the ones that failed to add)
            #    Plus 'a' itself.
            #    This is a smaller set.
            
            # Let's reconstruct the "failed" candidates list?
            # We don't have it stored globally.
            
            # Alternative: Just try to add ANY number not in S_temp.
            # But check time carefully.
            
            # Let's try a limited local search:
            # Try to add elements from the remaining candidates list (global) + a
            # We need to maintain a global list of "unadded" numbers.
            
            # Let's just restart the greedy on the new S_temp with the full pool of non-members.
            # But limit the number of additions to, say, 5?
            
            # Better:
            # Keep a global list of "available" numbers.
            # Initially: all composites.
            # When we add x, remove it from available.
            # When we remove a (if a was composite), add it back to available.
            # If a was prime, it's not in available, so we just add it to a temporary pool for this swap.
            
            # Let's implement this properly.
            
            pass # Placeholder for actual logic

    # Fallback: Return current S
    return S

# EVOLVE-BLOCK-END
