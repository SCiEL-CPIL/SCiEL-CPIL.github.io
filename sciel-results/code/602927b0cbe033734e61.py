"""Construct a small inclusion-maximal Sidon set in [1,256].

Strategy:
1. Generate diverse maximal Sidon sets via randomized greedy with different orderings.
2. Aggressively reduce cardinality via removal: remove an element if the set
   remains maximal (every missing element still conflicts).
3. Use a deadline guard to ensure we finish within 60s.

Key insight: A maximal Sidon set must block every element in [1,N] not in the set.
The most efficient way to block elements is to use elements that create many
conflicts. Randomized greedy with multiple orderings finds diverse maximal sets,
and local search reduction finds smaller maximal subsets.
"""

import time
import random

N = 256

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
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sums.add(lst[i] + lst[j])
    return sums

def _is_maximal(set_list, sums, N_val=N):
    """Check if set_list is inclusion-maximal Sidon in [1, N_val]."""
    s = set(set_list)
    for x in range(1, N_val + 1):
        if x in s:
            continue
        # If 2x is already in sums, then adding x would create duplicate 2x = x+x
        if 2 * x in sums:
            continue
        # Check if x + elem is in sums for any elem in set_list
        conflict = False
        for elem in set_list:
            if (x + elem) in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_with_order(order):
    """Greedy: pick elements in the given order if they can be added."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _local_search_reduce(start_set, deadline, rng):
    """Try to reduce cardinality by removing elements that preserve maximality."""
    current = list(start_set)
    current_sums = _compute_sums(current)
    
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
            if _is_maximal(candidate, cand_sums):
                current = candidate
                current_sums = cand_sums
                improved = True
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    best_set = []
    best_size = N + 1
    
    # Generate multiple diverse maximal Sidon sets
    # Strategy 1: Random orderings
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = _greedy_with_order(order)
        
        if not S:
            continue
        
        # Check if maximal
        sums = _compute_sums(S)
        if not _is_maximal(S, sums):
            continue
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Try to reduce further
        if time.time() < deadline:
            reduced = _local_search_reduce(S, deadline, rng)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
        
        # Early stop if we found a very small set
        if best_size <= 10:
            break
    
    # Strategy 2: Structured orderings - bias towards elements that block many others
    if time.time() < deadline:
        for trial in range(1000):
            if time.time() > deadline:
                break
            
            order = list(range(1, N + 1))
            # Bias towards middle values which tend to create more conflicts
            rng.shuffle(order)
            order.sort(key=lambda x: -abs(x - N/2) + rng.random() * 50)
            
            S = _greedy_with_order(order)
            
            if not S:
                continue
            
            sums = _compute_sums(S)
            if not _is_maximal(S, sums):
                continue
            
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
            
            if time.time() < deadline:
                reduced = _local_search_reduce(S, deadline, rng)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 10:
                break
    
    # Strategy 3: Hybrid - start with spread-out elements, then random
    if time.time() < deadline:
        for trial in range(1000):
            if time.time() > deadline:
                break
            
            order = list(range(1, N + 1))
            rng.shuffle(order)
            spread = [1, N//4, N//2, 3*N//4, N]
            for s in spread:
                if s in order:
                    order.remove(s)
            order = spread + order
            
            S = _greedy_with_order(order)
            
            if not S:
                continue
            
            sums = _compute_sums(S)
            if not _is_maximal(S, sums):
                continue
            
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
            
            if time.time() < deadline:
                reduced = _local_search_reduce(S, deadline, rng)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 10:
                break
    
    # Final reduction on best set found
    if best_set and time.time() < deadline:
        sums_best = _compute_sums(best_set)
        S_final = _local_search_reduce(best_set, deadline, rng)
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = S_final[:]
    
    # If no maximal set found (shouldn't happen), fall back to greedy ascending
    if not best_set:
        best_set = _greedy_with_order(list(range(1, N + 1)))
        # Ensure maximality
        sums = _compute_sums(best_set)
        if not _is_maximal(best_set, sums):
            current = list(best_set)
            current_sums = _compute_sums(current)
            current_set = set(current)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if x in current_set:
                        continue
                    if 2 * x in current_sums:
                        continue
                    conflict = False
                    for elem in current:
                        if (x + elem) in current_sums:
                            conflict = True
                            break
                    if not conflict:
                        current.append(x)
                        current_set.add(x)
                        current_sums.update([x + x] + [x + e for e in current[:-1]])
                        changed = True
                        break
            best_set = current
    
    # Final verification
    if not _is_sidon(best_set):
        # Fallback to simple greedy
        return _greedy_with_order(list(range(1, N + 1)))
    
    return best_set
