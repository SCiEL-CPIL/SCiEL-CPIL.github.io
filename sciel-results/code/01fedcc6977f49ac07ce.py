import time
import random

N = 256

def _compute_sums(set_list):
    """Compute all pair sums a+b where a<=b from the set."""
    sums = set()
    for i in range(len(set_list)):
        for j in range(i, len(set_list)):
            sums.add(set_list[i] + set_list[j])
    return sums

def _is_maximal(S, sums):
    """Check if S is inclusion-maximal: no element from [1,N] can be added."""
    S_set = set(S)
    for y in range(1, N + 1):
        if y in S_set:
            continue
        # Check if y can be added
        if 2 * y in sums:
            continue
        can_add = True
        for s in S:
            if y + s in sums:
                can_add = False
                break
        if can_add:
            return False
    return True

def _local_search_reduce(S, deadline):
    """Try to remove elements from S while preserving maximality."""
    S = S[:]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(S)):
            candidate_remove = S[idx]
            # Try removing S[idx]
            S_new = S[:idx] + S[idx+1:]
            sums_new = _compute_sums(S_new)
            if _is_maximal(S_new, sums_new):
                S = S_new
                changed = True
                break
    return S

def construct():
    best_size = N + 1
    best_set = []
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for x in candidates:
            # Check if x can be added
            valid = True
            if 2 * x in sums:
                valid = False
            else:
                for s in S:
                    if x + s in sums:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                sums.add(2 * x)
                for s in S[:-1]:
                    sums.add(x + s)
        
        # Now try to reduce via local search
        S_reduced = _local_search_reduce(S, deadline)
        
        if len(S_reduced) < best_size:
            best_size = len(S_reduced)
            best_set = S_reduced[:]
        
        # Also check if the original greedy set is maximal (it might not be)
        # and if so, record it
        sums_orig = _compute_sums(S)
        if _is_maximal(S, sums_orig):
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Early exit if we find a very small set
        if best_size <= 16:
            break
    
    return best_set
