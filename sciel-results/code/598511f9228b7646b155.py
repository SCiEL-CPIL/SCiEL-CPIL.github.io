"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys

def _check_candidate_fast(x, S, used_sums):
    """Check if x can be added to S without triple-sum collisions using bytearray.
    Returns True if valid, and updates used_sums if valid."""
    n = len(S)
    
    # Check all new triple sums involving x
    # (x,x,x)
    s = 3 * x
    if used_sums[s]:
        return False
    
    # (x,x,y) for each y in S
    for y in S:
        s = 2 * x + y
        if used_sums[s]:
            return False
    
    # (x,y,y) for each y in S
    for y in S:
        s = x + 2 * y
        if used_sums[s]:
            return False
    
    # (x,y,z) for each pair y<=z in S
    for i in range(n):
        for j in range(i, n):
            s = x + S[i] + S[j]
            if used_sums[s]:
                return False
    
    # All checks passed, mark sums as used
    used_sums[3 * x] = 1
    for y in S:
        used_sums[2 * x + y] = 1
        used_sums[x + 2 * y] = 1
    for i in range(n):
        for j in range(i, n):
            used_sums[x + S[i] + S[j]] = 1
    
    return True

def _unmark_sums(x, S, used_sums):
    """Remove all triple sums involving x from used_sums."""
    n = len(S)
    
    used_sums[3 * x] = 0
    for y in S:
        used_sums[2 * x + y] = 0
        used_sums[x + 2 * y] = 0
    for i in range(n):
        for j in range(i, n):
            used_sums[x + S[i] + S[j]] = 0

def _greedy_backtrack(deadline, max_depth=80):
    """Backtracking search that tries smallest valid numbers first."""
    S = []
    used_sums = bytearray(12289)  # max sum = 3*4096 = 12288
    start_time = time.time()
    
    def backtrack(min_val, target_depth):
        nonlocal S, used_sums
        if time.time() > deadline:
            return False
        if len(S) >= target_depth:
            return True
        
        for x in range(min_val, 4097):
            if time.time() > deadline:
                return False
            
            if _check_candidate_fast(x, S, used_sums):
                S.append(x)
                if backtrack(x + 1, target_depth):
                    return True
                S.pop()
                _unmark_sums(x, S, used_sums)
        
        return False
    
    # Try to build up to target_depth
    for target in range(1, max_depth + 1):
        if time.time() > deadline:
            break
        if backtrack(1, target):
            continue
        else:
            # Can't reach this target, return what we have
            break
    
    return S

def _greedy_increasing_fast(deadline):
    """Simple greedy in increasing order using bytearray."""
    S = []
    used_sums = bytearray(12289)
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if _check_candidate_fast(x, S, used_sums):
            S.append(x)
    
    return S

def _greedy_with_backtrack_from_base(base_S, deadline, max_improvement=5):
    """Start from a base set, try to improve by backtracking from the end."""
    S = base_S[:]
    used_sums = bytearray(12289)
    
    # Rebuild used_sums
    for i, x in enumerate(S):
        for j in range(i, len(S)):
            for k in range(j, len(S)):
                used_sums[S[i] + S[j] + S[k]] = 1
    
    start_size = len(S)
    start_time = time.time()
    
    # Try to extend the set
    for target in range(start_size + 1, start_size + max_improvement + 1):
        if time.time() > deadline:
            break
        
        # Try backtracking from the last element
        def backtrack(idx, min_val):
            nonlocal S, used_sums
            if time.time() > deadline:
                return False
            if len(S) >= target:
                return True
            
            for x in range(min_val, 4097):
                if time.time() > deadline:
                    return False
                
                if _check_candidate_fast(x, S, used_sums):
                    S.append(x)
                    if backtrack(idx + 1, x + 1):
                        return True
                    S.pop()
                    _unmark_sums(x, S, used_sums)
            
            return False
        
        if backtrack(len(S), S[-1] + 1 if S else 1):
            continue
        else:
            break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Strategy 1: Fast greedy in increasing order
    S1 = _greedy_increasing_fast(deadline)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Backtracking search from scratch
    if time.time() < deadline:
        S2 = _greedy_backtrack(deadline, max_depth=80)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Improve from best found so far
    if time.time() < deadline and best:
        S3 = _greedy_with_backtrack_from_base(best, deadline, max_improvement=5)
        if len(S3) > len(best):
            best = S3[:]
    
    return best
