"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
import numpy as np

def _is_sidon_addition(x, s, sums_set):
    """
    Check if x can be added to Sidon set s.
    s: list of elements
    sums_set: set of all pairwise sums a+b (a<=b) from s
    """
    # New sums: 2x, and x+a for all a in s
    # Must not collide with existing sums_set
    # Must not collide with each other (2x vs x+a, or x+a1 vs x+a2)
    
    # 1. Check 2x against sums_set
    if 2 * x in sums_set:
        return False
        
    # 2. Check x+a against sums_set and against 2x
    # Also check if x+a1 == x+a2 -> a1 == a2, which is fine since a are distinct.
    # So new sums are distinct among themselves if a are distinct.
    # We just need to ensure none of (2x, x+a) is in sums_set.
    
    for a in s:
        if (x + a) in sums_set:
            return False
            
    return True

def _get_new_sums(x, s):
    """
    Get the set of new sums added by x to s.
    """
    new_sums = {2 * x}
    for a in s:
        new_sums.add(x + a)
    return new_sums

def _is_blocked(x, s, sums_set):
    """
    Check if x is blocked by s. i.e., x cannot be added to s.
    This is the same as not _is_sidon_addition.
    """
    return not _is_sidon_addition(x, s, sums_set)

def _count_unblocked_candidates(x, s, sums_set, candidates):
    """
    If we add x to s, how many candidates y (currently addable) become blocked?
    candidates: set of elements not in s.
    Note: We only count y that are CURRENTLY addable to s.
    """
    if not _is_sidon_addition(x, s, sums_set):
        return -1 # x itself cannot be added
        
    new_sums = _get_new_sums(x, s)
    # The new sums set will be sums_set U new_sums
    # A candidate y is currently addable if _is_sidon_addition(y, s, sums_set) is True.
    # It becomes blocked if _is_sidon_addition(y, s U {x}, sums_set U new_sums) is False.
    
    count = 0
    for y in candidates:
        if y == x:
            continue
        # Check if y is currently addable
        if not _is_sidon_addition(y, s, sums_set):
            continue # Already blocked, no change
            
        # Check if y is blocked by s U {x}
        # This is equivalent to checking if adding y to s U {x} causes collision.
        # Collision if:
        # 1. 2y in (sums_set U new_sums)
        # 2. y + a in (sums_set U new_sums) for some a in s U {x}
        
        # Optimization: 
        # If 2y in sums_set, it was already blocked? No, if 2y in sums_set, 
        # _is_sidon_addition(y, s, sums_set) would have returned False.
        # So 2y is NOT in sums_set.
        # It might be in new_sums.
        
        if 2 * y in new_sums:
            count += 1
            continue
            
        # Check y + a for a in s
        # If y+a in sums_set, it was already blocked. So y+a NOT in sums_set for all a in s.
        # It might be in new_sums.
        
        # Check y + x (new pair)
        # If y+x in sums_set U new_sums?
        # y+x is a new sum from the perspective of s? No, y is not in s.
        # When we add x, the set becomes s U {x}.
        # The sums of s U {x} are sums_set U {2x} U {x+a for a in s}.
        # Now we try to add y.
        # New sums from y: 2y, and y+b for b in s U {x}.
        # Collisions:
        # 1. 2y in sums(s U {x})? 
        #    We checked 2y in new_sums. If not, and 2y not in sums_set, then ok.
        # 2. y+b in sums(s U {x}) for b in s U {x}.
        #    For b in s: y+b in sums_set? No (y was addable). y+b in new_sums?
        #    For b=x: y+x in sums_set? y+x in new_sums?
        
        # So we check:
        # - Is (y+x) in sums_set?
        # - Is (y+x) in new_sums?
        # - Is (y+a) in new_sums for any a in s?
        
        if (y + x) in sums_set:
            count += 1
            continue
            
        if (y + x) in new_sums:
            count += 1
            continue
            
        for a in s:
            if (y + a) in new_sums:
                count += 1
                break
                
    return count

def _greedy_blocking_construct(N):
    """
    Construct a maximal Sidon set by greedily picking elements that block the most candidates.
    """
    s = []
    sums_set = set()
    candidates = set(range(1, N + 1))
    
    while candidates:
        best_x = None
        best_count = -1
        
        # Filter candidates to those that can be added
        addable = []
        for x in candidates:
            if _is_sidon_addition(x, s, sums_set):
                addable.append(x)
                
        if not addable:
            break
            
        # Evaluate blocking power
        # Optimization: If s is large, this is O(|s|^2 * |candidates|).
        # For N=256, it's manageable if we limit iterations or optimize.
        
        for x in addable:
            # Heuristic: If we just want speed, we can sample?
            # But N is small. Let's compute exact count.
            cnt = _count_unblocked_candidates(x, s, sums_set, candidates)
            if cnt > best_count:
                best_count = cnt
                best_x = x
                
        if best_x is None:
            break
            
        # Add best_x
        s.append(best_x)
        new_sums = _get_new_sums(best_x, s[:-1]) # s[:-1] is the old s
        sums_set.update(new_sums)
        candidates.remove(best_x)
        
    return s

def _minimize_set(s, N):
    """
    Remove elements to minimize size while maintaining maximality.
    """
    s = list(s)
    changed = True
    while changed:
        changed = False
        for i in range(len(s) - 1, -1, -1):
            candidate = s[:i] + s[i+1:]
            if not candidate:
                continue
                
            # Check if candidate is maximal
            cand_sums = set()
            n = len(candidate)
            for j in range(n):
                cj = candidate[j]
                for k in range(j, n):
                    cand_sums.add(cj + candidate[k])
                    
            # Check maximality
            is_maximal = True
            cand_set = set(candidate)
            for x in range(1, N + 1):
                if x in cand_set:
                    continue
                # Check if x is blocked
                if 2 * x in cand_sums:
                    continue
                blocked = False
                for a in candidate:
                    if (x + a) in cand_sums:
                        blocked = True
                        break
                if not blocked:
                    is_maximal = False
                    break
                    
            if is_maximal:
                s = candidate
                changed = True
                break
                
    return s

def construct():
    best_size = N + 1
    best_set = None
    start_time = time.time()
    time_limit = 28.0
    
    random.seed(42)
    
    # Try multiple runs of the greedy blocking constructor with different random seeds/orders?
    # The greedy constructor is deterministic if we iterate candidates in sorted order?
    # No, we pick max count. Ties?
    # To diversify, we can shuffle the candidate iteration order or add random tie-breaking.
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Add some randomness to the greedy choice
        # We can shuffle the 'addable' list before evaluating? No, that changes the max.
        # We can add a small random perturbation to the count.
        
        s = _greedy_blocking_construct(N)
        
        # The greedy constructor produces a maximal set?
        # It stops when no addable candidates remain.
        # So it is maximal.
        
        s_min = _minimize_set(s, N)
        
        if len(s_min) < best_size:
            best_size = len(s_min)
            best_set = s_min[:]
            
        # Local search on best_set
        if best_set and time.time() - start_time < time_limit - 2.0:
            # Try randomized removals
            trial = best_set[:]
            # Shuffle indices
            indices = list(range(len(trial)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > time_limit - 1.0:
                    break
                if len(best_set) == 0:
                    break
                # Map shuffled index to current list index?
                # Since we modify best_set, indices shift.
                # Let's just pick a random index from the current best_set.
                r_idx = random.randint(0, len(best_set) - 1)
                candidate = best_set[:r_idx] + best_set[r_idx+1:]
                
                # Check maximality of candidate
                if not candidate:
                    continue
                    
                cand_sums = set()
                n = len(candidate)
                for j in range(n):
                    cj = candidate[j]
                    for k in range(j, n):
                        cand_sums.add(cj + candidate[k])
                        
                is_maximal = True
                cand_set = set(candidate)
                for x in range(1, N + 1):
                    if x in cand_set:
                        continue
                    if 2 * x in cand_sums:
                        continue
                    blocked = False
                    for a in candidate:
                        if (x + a) in cand_sums:
                            blocked = True
                            break
                    if not blocked:
                        is_maximal = False
                        break
                        
                if is_maximal:
                    best_set = candidate
                    best_size = len(best_set)
                    # Restart local search?
                    break
                    
    if best_set is None:
        # Fallback: simple greedy
        s = []
        sums = set()
        for x in range(1, N+1):
            if _is_sidon_addition(x, s, sums):
                s.append(x)
                sums.update(_get_new_sums(x, s[:-1]))
        return s
        
    return best_set
# EVOLVE-BLOCK-END
