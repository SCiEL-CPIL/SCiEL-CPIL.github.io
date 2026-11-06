"""Construct a small inclusion-maximal Sidon set in [1,256].

Strategy:
1. Precompute all "conflict hyperedges": for each missing element x, the set of
   elements y such that adding x would create a sum collision with {y, z} in S.
2. Use a hitting-set / set-cover approach: we need to pick a Sidon set S such that
   for every x not in S, S intersects the conflict set of x.
3. Start with small random cores, then use a greedy set-cover to add elements that
   block the most unblocked missing elements while maintaining the Sidon property.
4. Local search: try removing elements and swapping to reduce cardinality.
"""

import time
import random

N = 256

def _is_sidon(set_list):
    """Check if set_list is a Sidon set."""
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
        # If 2x is already in sums, x is blocked
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

def _unblocked_elements(set_list, sums):
    """Return list of missing elements that are NOT blocked (could be added)."""
    s = set(set_list)
    unblocked = []
    for x in range(1, N + 1):
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
            unblocked.append(x)
    return unblocked

def _can_add(x, current, sums):
    """Check if x can be added to current Sidon set."""
    new_sums = [x + x] + [x + a for a in current]
    if len(set(new_sums)) != len(new_sums):
        return False
    if sums.intersection(new_sums):
        return False
    return True

def _greedy_complete(core, deadline, rng):
    """Starting from a core Sidon set, greedily add elements to block all missing ones."""
    current = list(core)
    current_set = set(current)
    sums = _compute_sums(current)
    
    while time.time() < deadline:
        unblocked = _unblocked_elements(current, sums)
        if not unblocked:
            break
        
        # Find the element that blocks the most unblocked elements
        best_elem = None
        best_count = 0
        # Shuffle for tie-breaking diversity
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if x in current_set:
                continue
            if not _can_add(x, current, sums):
                continue
            
            # Count how many unblocked elements this would block
            count = 0
            new_sums = sums | {x + x} | {x + a for a in current}
            for y in unblocked:
                if 2 * y in new_sums:
                    count += 1
                elif any((y + a) in new_sums for a in current + [x]):
                    count += 1
            
            if count > best_count:
                best_count = count
                best_elem = x
        
        if best_elem is None:
            # Can't add any more elements while maintaining Sidon
            break
        
        current.append(best_elem)
        current_set.add(best_elem)
        sums.update([best_elem + best_elem] + [best_elem + a for a in current[:-1]])
    
    return current

def _local_search_reduce(start_set, deadline, rng):
    """Try to reduce cardinality by removing elements that preserve maximality."""
    current = list(start_set)
    
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
            if not _is_sidon(candidate):
                continue
            cand_sums = _compute_sums(candidate)
            if _is_maximal(candidate, cand_sums):
                current = candidate
                improved = True
                break
    
    return current

def _local_search_swap(start_set, deadline, rng):
    """Try to reduce cardinality by swapping elements."""
    current = list(start_set)
    current_set = set(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one element and adding another (net zero or -1)
        for idx in list(range(len(current))):
            if time.time() > deadline:
                break
            to_remove = current[idx]
            remaining = current[:idx] + current[idx+1:]
            rem_sums = _compute_sums(remaining)
            
            # Try adding each missing element
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if x in remaining or x == to_remove:
                    continue
                if not _can_add(x, remaining, rem_sums):
                    continue
                candidate = remaining + [x]
                cand_sums = _compute_sums(candidate)
                if _is_maximal(candidate, cand_sums):
                    current = candidate
                    current_set = set(current)
                    improved = True
                    break
            if improved:
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    best_set = []
    best_size = N + 1
    
    # Phase 1: Try small random cores and complete them
    for core_size in range(3, 12):
        if time.time() > deadline:
            break
        
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random core
            core = rng.sample(range(1, N + 1), core_size)
            
            # Check if core is Sidon
            if not _is_sidon(core):
                continue
            
            # Complete greedily
            completed = _greedy_complete(core, deadline, rng)
            
            if not completed:
                continue
            
            sums = _compute_sums(completed)
            if not _is_maximal(completed, sums):
                continue
            
            if len(completed) < best_size:
                best_size = len(completed)
                best_set = completed[:]
            
            # Try to reduce
            if time.time() < deadline:
                reduced = _local_search_reduce(completed, deadline, rng)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 12:
                break
    
    # Phase 2: Structured cores - spread out elements
    if time.time() < deadline:
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Create a spread-out core
            base = [1, N//4, N//2, 3*N//4, N]
            extra = rng.sample([x for x in range(1, N+1) if x not in base], rng.randint(0, 5))
            core = base + extra
            rng.shuffle(core)
            
            if not _is_sidon(core):
                continue
            
            completed = _greedy_complete(core, deadline, rng)
            
            if not completed:
                continue
            
            sums = _compute_sums(completed)
            if not _is_maximal(completed, sums):
                continue
            
            if len(completed) < best_size:
                best_size = len(completed)
                best_set = completed[:]
            
            if time.time() < deadline:
                reduced = _local_search_reduce(completed, deadline, rng)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 12:
                break
    
    # Phase 3: Random greedy with various orderings
    if time.time() < deadline:
        for trial in range(1000):
            if time.time() > deadline:
                break
            
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            selected = []
            sums = set()
            for x in order:
                added = [x + a for a in selected] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    selected.append(x)
                    sums.update(added)
            
            if not selected:
                continue
            
            if not _is_maximal(selected, sums):
                continue
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
            
            if time.time() < deadline:
                reduced = _local_search_reduce(selected, deadline, rng)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 12:
                break
    
    # Phase 4: Final reduction on best set
    if best_set and time.time() < deadline:
        S_final = _local_search_reduce(best_set, deadline, rng)
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = S_final[:]
        
        if time.time() < deadline:
            S_swap = _local_search_swap(best_set, deadline, rng)
            if len(S_swap) < best_size:
                best_size = len(S_swap)
                best_set = S_swap[:]
    
    # Fallback
    if not best_set:
        best_set = _greedy_complete([], deadline, rng)
        if not _is_maximal(best_set, _compute_sums(best_set)):
            # Force maximality by adding blocking elements
            sums = _compute_sums(best_set)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                unblocked = _unblocked_elements(best_set, sums)
                for x in unblocked:
                    if _can_add(x, best_set, sums):
                        best_set.append(x)
                        sums.update([x+x] + [x+a for a in best_set[:-1]])
                        changed = True
                        break
    
    # Final verification
    if not _is_sidon(best_set):
        return _greedy_complete([], deadline, rng)
    
    return best_set
