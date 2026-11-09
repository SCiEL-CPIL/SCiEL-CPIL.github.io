"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
import numpy as np

def _fast_sidon_check(s):
    """Check if set s is a Sidon set. Returns True if Sidon."""
    sums = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            val = si + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _get_sums_set(s):
    """Compute the set of all pairwise sums a+b for a<=b in s."""
    sums = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            sums.add(si + s[j])
    return sums

def _is_maximal_fast(s, N, sums_set):
    """
    Check if Sidon set s is inclusion-maximal in [1,N].
    s: list of elements
    N: bound
    sums_set: precomputed set of sums from s
    """
    s_set = set(s)
    # For each x not in s, check if x is blocked
    # x is blocked if (2x in sums_set) OR (exists a in s such that x+a in sums_set)
    # Note: sums_set contains a+b for a<=b.
    # If we add x, new sums are 2x and x+a for all a in s.
    # Conflict if 2x in sums_set OR (x+a) in sums_set for some a in s.
    # Note: x+a might equal b+c where b,c in s.
    # Since x not in s, x is distinct from all a.
    
    for x in range(1, N + 1):
        if x in s_set:
            continue
        
        # Check 2x
        if 2 * x in sums_set:
            continue
        
        # Check x + a for a in s
        # Optimization: We can precompute which sums are present.
        # We need to check if any (x+a) is in sums_set.
        blocked = False
        for a in s:
            if (x + a) in sums_set:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _build_greedy_sidon_heuristic(order, N):
    """
    Build a Sidon set.
    Heuristic: Prefer elements that 'block' many others?
    Actually, for minimization, we want a dense set of blockers.
    Standard greedy is fine, but we can try to prioritize 'large' numbers or specific patterns.
    """
    selected = []
    sums = set()
    # Sort order to potentially find better structures?
    # Random shuffle is already done outside.
    for x in order:
        # Check if x can be added
        # New sums: 2x, x+a for a in selected
        # Must not collide with existing sums or each other
        new_sums = [2 * x]
        for a in selected:
            new_sums.append(x + a)
        
        # Check internal collisions in new_sums
        if len(set(new_sums)) != len(new_sums):
            continue
            
        # Check collision with existing sums
        if sums.intersection(new_sums):
            continue
            
        selected.append(x)
        sums.update(new_sums)
        
    return selected

def _minimize_set(s, N):
    """
    Aggressively remove elements to maintain maximality.
    Uses fast checks.
    """
    s = list(s)
    changed = True
    # Repeat until no change
    while changed:
        changed = False
        # Try removing each element
        # We iterate backwards to handle list modification safely if we were modifying in place, 
        # but we create new lists here.
        for i in range(len(s) - 1, -1, -1):
            # Candidate without s[i]
            candidate = s[:i] + s[i+1:]
            
            # Check Sidon property of candidate
            # Since s was Sidon, removing an element keeps it Sidon. 
            # So we don't need to re-check Sidon property!
            
            # Check Maximality of candidate
            # Compute sums for candidate
            cand_sums = _get_sums_set(candidate)
            
            if _is_maximal_fast(candidate, N, cand_sums):
                s = candidate
                changed = True
                break # Restart loop as indices changed
                
    return s

def construct():
    best_size = N + 1
    best_set = None
    start_time = time.time()
    time_limit = 28.0
    
    random.seed(42)
    
    # Pre-compute nothing global, but we can use numpy for faster ops if needed?
    # For N=256, pure Python sets are fast enough if we reduce iterations.
    
    # Strategy: 
    # 1. Generate many random Sidon sets.
    # 2. Minimize them.
    # 3. Keep the smallest.
    
    for attempt in range(1000): # Increase attempts
        if time.time() - start_time > time_limit:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        s = _build_greedy_sidon_heuristic(order, N)
        
        # Quick prune: if current best is small, skip if this one is already bigger
        # Greedy usually produces size ~ sqrt(N) * constant.
        if len(s) >= best_size:
            # It's possible that removing elements from a larger set yields a smaller maximal set?
            # No, if S is Sidon, and S' is a subset of S, S' is Sidon.
            # If S' is maximal, S cannot be maximal (since S' is a subset, S can be extended to S? No, S is larger).
            # Wait. If S is Sidon, any subset is Sidon.
            # If we minimize S to S_min, S_min is a subset of S.
            # If we start with a smaller S, the minimal subset S_min will be <= len(S).
            # So if len(s) >= best_size, the result of minimizing s will have size <= len(s), 
            # but could it be < best_size? 
            # Yes, if best_size was found from a different set.
            # However, usually smaller initial sets lead to smaller minimal sets.
            # Let's just check if len(s) is much larger, maybe skip?
            # For now, keep checking if time permits.
            pass
            
        s_min = _minimize_set(s, N)
        
        if len(s_min) < best_size:
            best_size = len(s_min)
            best_set = s_min[:]
            
        # Local search on best_set: try to swap elements to reduce size further?
        # Or try to remove more? _minimize_set already removes all possible.
        # Can we find a *different* maximal subset of the same size?
        # Or a smaller one via swaps?
        # If we swap x for y, size remains same. We want to reduce size.
        # So we only care about removals. _minimize_set does exhaustive removal search (greedy order).
        # To improve, we can randomize the order of removal attempts.
        
        if best_set and time.time() - start_time < time_limit - 2.0:
             # Try randomized removal order on the current best
             # Create a copy
             trial = best_set[:]
             # Shuffle indices for removal
             indices = list(range(len(trial)))
             random.shuffle(indices)
             improved = False
             for idx in indices:
                 if time.time() - start_time > time_limit - 1.0:
                     break
                 # Remove element at current position idx (note: positions shift)
                 # We need to map original index to current index?
                 # Easier: just try removing each element in random order from the *current* list
                 # Let's do a simpler local search:
                 # Pick a random element from best_set, try to remove it.
                 # If successful, update best_set.
                 if len(best_set) == 0: break
                 idx = random.randint(0, len(best_set) - 1)
                 candidate = best_set[:idx] + best_set[idx+1:]
                 cand_sums = _get_sums_set(candidate)
                 if _is_maximal_fast(candidate, N, cand_sums):
                     best_set = candidate
                     best_size = len(best_set)
                     improved = True
                     # Continue trying to remove more from the new best
                     # The loop above (for idx in indices) is static, so let's just break and let the outer loop handle it?
                     # Or just continue this loop with the updated best_set?
                     # The indices are stale. Let's just break and let the next outer iteration (or a new local search phase) handle it.
                     # Actually, we can just restart the local search phase.
                     break

    if best_set is None:
        # Fallback
        return _build_greedy_sidon_heuristic(list(range(1, N + 1)), N)
        
    return best_set
# EVOLVE-BLOCK-END
