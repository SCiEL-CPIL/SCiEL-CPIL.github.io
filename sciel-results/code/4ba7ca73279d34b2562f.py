"""Improved program: MODEL-AND-OPTIMIZE for small maximal Sidon sets in [1,256].

Strategy:
1. Start from strong baselines (greedy ascending, descending, random).
2. Use combined local search:
   a. Remove elements one by one if the set remains maximal.
   b. Try swaps: remove one element, add a missing element, check Sidon + maximality.
3. Multiple randomized restarts to escape local optima.
4. Deadline guard with safety margin.
"""

import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(set_list):
    """Check if set_list is a Sidon set. Returns True if all pair sums a+b (a<=b) are distinct."""
    s = set(set_list)
    if len(s) != len(set_list):
        return False
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sm = lst[i] + lst[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _compute_sums(set_list):
    """Compute the set of all pair sums."""
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sums.add(lst[i] + lst[j])
    return sums

def _is_maximal_fast(set_list, sums, N_val=N):
    """Faster maximality check using precomputed sums."""
    s = set(set_list)
    for x in range(1, N_val + 1):
        if x in s:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for elem in set_list:
            if (x + elem) in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_ascending():
    """Original greedy: pick smallest possible element."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_descending():
    """Greedy descending: pick largest possible element."""
    selected = []
    sums = set()
    for x in range(N, 0, -1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _make_maximal(current, deadline):
    """
    Ensure the set is maximal by adding any missing element that doesn't break Sidon.
    Returns the maximal set.
    """
    current = list(current)
    current_set = set(current)
    current_sums = _compute_sums(current)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            # Check if adding x keeps it Sidon
            new_sums = [x + a for a in current] + [2 * x]
            # Check for internal conflicts in new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
            # Check conflict with existing sums
            if current_sums.intersection(set(new_sums)):
                continue
            
            # It is a valid addition. Add it.
            current.append(x)
            current_set.add(x)
            current_sums.update(new_sums)
            changed = True
            break
            
    return current

def _local_search_reduce(start_set, deadline, rng):
    """
    Local search: iteratively removes elements that do not break maximality.
    This is the primary optimization step.
    """
    current = list(start_set)
    
    # Ensure we start with a maximal set? 
    # Actually, if we start with a Sidon set, removing elements keeps it Sidon.
    # But we need the *result* to be maximal.
    # The standard approach: 
    # 1. Start with Sidon set.
    # 2. Try to remove elements. If removal makes it non-maximal, we can't remove it?
    # NO. If we remove an element, the set might become non-maximal (i.e., we could add it back or another element).
    # The goal is to find a *maximal* Sidon set of minimum size.
    # If we have a maximal Sidon set S, and we remove x, S\x might not be maximal.
    # However, if S\x is Sidon, we can try to make it maximal again.
    # But making it maximal might require adding elements back, potentially increasing size.
    # 
    # Correct logic for "Small Maximal Sidon Set":
    # We want a Sidon set S such that for all y not in S, S U {y} is NOT Sidon.
    # This is equivalent to: S is Sidon, and for all y not in S, there exists a,b in S (a<=b) such that a+b = y+c for some c in S? No.
    # S is maximal if for all y not in S, S U {y} is not Sidon.
    # S U {y} is not Sidon if there is a collision in sums.
    # Collision types:
    # 1. y+y = a+b (a,b in S)
    # 2. y+a = b+c (a,b,c in S, a!=b, etc)
    # 
    # So, if we remove x from a maximal set S, the new set S' = S\{x} is Sidon.
    # Is S' maximal? 
    # If S' is not maximal, there exists y not in S' such that S' U {y} is Sidon.
    # If we simply remove x, we might end up with a non-maximal set.
    # The algorithm should be:
    # 1. Start with a maximal Sidon set.
    # 2. Try to remove x.
    # 3. If S\{x} is Sidon, check if it can be made maximal with size <= |S|-1?
    #    Actually, if S\{x} is Sidon, we can just check if S\{x} is maximal.
    #    If it IS maximal, we have reduced the size. Great.
    #    If it is NOT maximal, it means we can add some y back. 
    #    If we add y back, we might end up with a different maximal set of the same size or larger.
    #    But we want to MINIMIZE size.
    #    So, if S\{x} is not maximal, it's not a valid candidate for a *smaller* maximal set unless we can find a maximal subset of S\{x}? No, S\{x} is a specific set.
    #    
    #    Wait. If S is maximal, and we remove x, S' = S\{x}.
    #    If S' is maximal, then |S'| < |S| and we improved.
    #    If S' is NOT maximal, then S' is not a solution (because solution must be maximal).
    #    So we can only accept the removal if S' is maximal.
    #    
    #    Therefore, the check `_is_maximal_fast(candidate, cand_sums)` is correct.
    #    If it returns True, we accept the removal.
    #    If it returns False, we reject the removal (for this iteration).
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            
            cand_sums = _compute_sums(candidate)
            if _is_maximal_fast(candidate, cand_sums):
                current = candidate
                improved = True
                break

    return current

def construct():
    start_time = time.time()
    # Use 28s deadline to be safe against 30s hard limit
    deadline = start_time + 28.0
    
    # Generate diverse starting sets
    candidates = []
    
    # 1. Greedy ascending
    candidates.append(_greedy_ascending())
    
    # 2. Greedy descending
    candidates.append(_greedy_descending())
    
    # 3. Multiple random greedy (increase count for better coverage, but check time)
    # Greedy is O(N^2) roughly, which is fast. 100 greeds should be < 1s.
    for seed in range(100):
        if time.time() > deadline:
            break
        candidates.append(_greedy_random(seed))
    
    # Filter valid Sidon sets (all greedy outputs are Sidon by construction, but check just in case)
    valid_candidates = []
    for c in candidates:
        if _is_sidon(c):
            valid_candidates.append(c)
            
    if not valid_candidates:
        return _greedy_ascending()
        
    # Sort by size to prioritize smaller sets
    valid_candidates.sort(key=len)
    
    best_set = valid_candidates[0]
    best_size = len(best_set)
    
    # Apply local search to top candidates
    # We process more candidates but with a strict deadline
    rng = random.Random(42)
    
    # Process top 30 candidates (increased from 15, but local search is faster now)
    for c in valid_candidates[:30]:
        if time.time() > deadline:
            break
        
        # Ensure the candidate is maximal before reducing?
        # Greedy sets are maximal by construction?
        # Greedy ascending: picks x if no conflict. At the end, no x can be added. So yes, maximal.
        # Greedy descending: same.
        # Greedy random: same.
        # So we can start reduction directly.
        
        improved = _local_search_reduce(c, deadline, rng)
        
        # Verify the result is still a valid maximal Sidon set
        if _is_sidon(improved) and _is_maximal_fast(improved, _compute_sums(improved)):
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    # Final verification
    if not _is_sidon(best_set):
        return _greedy_ascending()
        
    # Ensure maximality (safety check)
    sums = _compute_sums(best_set)
    if not _is_maximal_fast(best_set, sums):
        # This should not happen if local search logic is correct, but fix if needed
        best_set = _make_maximal(best_set, deadline)
        
    return best_set

# EVOLVE-BLOCK-END
