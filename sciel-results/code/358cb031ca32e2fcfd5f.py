"""Improved program: MODEL-AND-OPTIMIZE for small maximal Sidon sets in [1,256].

Strategy:
1. Start from a strong baseline (greedy ascending).
2. Use local search: remove elements one by one if the set remains maximal (i.e., every missing element still conflicts).
   A set is maximal iff for every x not in S, there exists a,b in S such that a+b = x+c for some c in S OR more precisely:
   Actually, maximality means: for every x not in S, adding x creates a duplicate sum.
   This happens iff there exist a,b in S (possibly a=b) and c,d in S (possibly c=d) such that a+b = c+d AND x is involved?
   No: adding x creates a duplicate iff there exists some sum involving x that already exists in the set of sums of S.
   The sums involving x are: x+x=2x, and x+s for s in S.
   So S is maximal iff for every x not in S, either 2x is in sums(S) OR x+s is in sums(S) for some s in S.
   Since sums(S) contains all a+b for a<=b in S, we check: is 2x in sums? Is there s in S such that x+s in sums?

   Actually simpler: S is NOT maximal if there exists x not in S such that all sums {2x} ∪ {x+s : s in S} are NOT in sums(S).

3. We want to minimize |S| while maintaining maximality.
   So we can try to remove elements. After removing an element, we need to re-check maximality.
   If removal preserves maximality, keep it removed.

4. We can also try swaps: remove one element, add another (that wasn't in S), check if still Sidon and maximal.

5. Use deadline guard.
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

def _local_search_reduce(start_set, deadline, rng):
    """
    Reduced local search.
    Iteratively removes elements that do not break maximality.
    """
    current = list(start_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            
            # Check if still maximal
            cand_sums = _compute_sums(candidate)
            if _is_maximal_fast(candidate, cand_sums):
                current = candidate
                improved = True
                break # Restart loop to re-evaluate with new set

    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate multiple diverse starting sets
    candidates = []
    
    # 1. Greedy ascending
    candidates.append(_greedy_ascending())
    
    # 2. Multiple random greedy
    for seed in range(100):
        if time.time() > deadline:
            break
        candidates.append(_greedy_random(seed))
    
    # Filter valid Sidon sets
    valid_candidates = []
    for c in candidates:
        if _is_sidon(c):
            valid_candidates.append(c)
            
    if not valid_candidates:
        return _greedy_ascending()
        
    # Sort by size
    valid_candidates.sort(key=len)
    
    best_set = valid_candidates[0]
    best_size = len(best_set)
    
    # Apply local search reduction to the top candidates
    rng = random.Random(42)
    
    # Process top 20 candidates to ensure we find a good local minimum
    for c in valid_candidates[:20]:
        if time.time() > deadline:
            break
        reduced = _local_search_reduce(c, deadline, rng)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
            
    # Final verification
    if not _is_sidon(best_set):
        return _greedy_ascending()
        
    # Ensure maximality
    sums = _compute_sums(best_set)
    if not _is_maximal_fast(best_set, sums):
        # Fallback: add elements to make it maximal
        current = list(best_set)
        current_set = set(current)
        current_sums = sums
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if x in current_set:
                    continue
                # Check if x is undominated
                if (2 * x) in current_sums:
                    continue
                covered = False
                for elem in current:
                    if (x + elem) in current_sums:
                        covered = True
                        break
                if not covered:
                    # Add x
                    current.append(x)
                    current_set.add(x)
                    # Update sums
                    for elem in current[:-1]:
                        current_sums.add(x + elem)
                    current_sums.add(2 * x)
                    changed = True
                    break
        best_set = current
        
    return best_set

# EVOLVE-BLOCK-END
