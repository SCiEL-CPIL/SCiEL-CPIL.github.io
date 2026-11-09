import time
import random
import numpy as np
from itertools import combinations

N = 256
DEADLINE = time.time() + 55

# EVOLVE-BLOCK-START

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a valid Sidon set."""
    n = len(selected)
    expected = n * (n + 1) // 2
    sums = _build_sums(selected)
    return len(sums) == expected

def _is_maximal(selected, sums):
    """
    Check if the set is inclusion-maximal.
    A set S is maximal if for every x in [1, N] \ S, adding x violates Sidon property.
    """
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        # Check if x can be added
        if 2 * x in sums:
            continue
        
        # Check if x conflicts with any existing sum via x+a
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _greedy_conflict_graph(perm, rng):
    """
    Greedy using conflict graph model.
    Process elements in order, add if it doesn't create conflicts.
    """
    selected = []
    sums = set()
    for x in perm:
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if conflict:
            continue
            
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
            
    return selected, sums

def _greedy_dominating_set(perm, rng):
    """
    Alternative greedy: prioritize elements that block many others.
    This targets inclusion-maximality directly.
    """
    selected = []
    sums = set()
    remaining = set(perm)
    
    while remaining:
        # For each candidate, count how many other elements it would block
        best_candidate = None
        best_block_count = -1
        
        # Sample a subset if too many remaining for speed
        candidates = list(remaining)
        if len(candidates) > 30:
            candidates = rng.sample(candidates, 30)
        
        for x in candidates:
            if x in set(selected):
                continue
                
            # Check if x can be added
            if 2 * x in sums:
                continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Count how many remaining elements this would block
            block_count = 0
            new_sums = sums | {2 * x}
            for a in selected:
                new_sums.add(x + a)
            
            for y in remaining:
                if y == x or y in set(selected):
                    continue
                # Check if y would be blocked by adding x
                if 2 * y in new_sums:
                    block_count += 1
                else:
                    for a in selected + [x]:
                        if y + a in new_sums:
                            block_count += 1
                            break
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_candidate = x
        
        if best_candidate is None:
            # No valid candidate found, break
            break
        
        # Add best candidate
        x = best_candidate
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
        
        remaining.discard(x)
        
        # Remove elements that are now blocked
        still_remaining = set()
        for y in remaining:
            if 2 * y in sums:
                continue
            blocked = False
            for a in selected:
                if y + a in sums:
                    blocked = True
                    break
            if not blocked:
                still_remaining.add(y)
        remaining = still_remaining
    
    return selected, sums

def _greedy_weighted(perm, rng, weights):
    """
    Greedy with weighted selection: at each step, pick the available element
    with highest weight among those that can be added.
    """
    selected = []
    sums = set()
    remaining = list(perm)
    
    while remaining:
        best_x = None
        best_w = -1
        
        for x in remaining:
            if 2 * x in sums:
                continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict:
                continue
            
            w = weights[x]
            if w > best_w:
                best_w = w
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
        
        remaining.remove(best_x)
    
    return selected, sums

def _local_search(selected, sums, rng, time_limit):
    """
    Local search to minimize size while maintaining maximality.
    Tries single removals, double removals, and swaps.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        # --- Single Removal Pass ---
        for i in range(len(cur_sel)):
            if time.time() > DEADLINE:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            removed_sums = {2 * e}
            for a in new_sel:
                removed_sums.add(e + a)
            
            new_sums = cur_sums - removed_sums
            
            if _is_maximal(new_sel, new_sums):
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break
        
        if not improved:
            # --- Double Removal Pass ---
            for i in range(len(cur_sel)):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                for j in range(i + 1, len(cur_sel)):
                    new_sel = [x for k, x in enumerate(cur_sel) if k != i and k != j]
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    if _is_maximal(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            # --- Swap Pass: try replacing one element with another ---
            sset = set(cur_sel)
            for i in range(len(cur_sel)):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                e = cur_sel[i]
                
                # Try replacing e with a random missing element
                for _ in range(5):
                    if time.time() > DEADLINE:
                        return best_size, best_sel, best_sums
                    
                    # Pick a random element not in cur_sel
                    candidates = [x for x in range(1, N + 1) if x not in sset]
                    if not candidates:
                        break
                    new_e = rng.choice(candidates)
                    
                    new_sel = [x for x in cur_sel if x != e] + [new_e]
                    new_sel.sort()
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    # Check if valid Sidon
                    valid = len(new_sums) == len(new_sel) * (len(new_sel) + 1) // 2
                    if valid and _is_maximal(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            break

    return best_size, best_sel, best_sums

def _triple_removal_search(selected, sums, rng, time_limit):
    """
    Try removing 3 elements at once to find smaller maximal sets.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE:
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        found = False
        n = len(cur_sel)
        
        # Try all triples of removals (only feasible for small n)
        if n <= 25:
            for i in range(n):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                for j in range(i + 1, n):
                    if time.time() > DEADLINE:
                        return best_size, best_sel, best_sums
                    for k in range(j + 1, n):
                        new_sel = [x for idx, x in enumerate(cur_sel) 
                                   if idx != i and idx != j and idx != k]
                        
                        new_sums = _build_sums(new_sel)
                        
                        if _is_maximal(new_sel, new_sums):
                            selected = new_sel
                            sums = new_sums
                            found = True
                            if len(selected) < best_size:
                                best_size = len(selected)
                                best_sel = list(selected)
                                best_sums = set(sums)
                            break
                    if found:
                        break
                if found:
                    break
        
        if not found:
            break
    
    return best_size, best_sel, best_sums

def construct():
    rng = random.Random(42)
    best_size = N + 1
    best_result = None
    best_sums = None
    
    start = time.time()
    
    # Phase 1: Find good initial maximal Sidon sets using different greedy strategies
    iterations = 0
    while time.time() < DEADLINE - 12:
        iterations += 1
        
        # Alternate between different greedy strategies
        strategy = iterations % 4
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        if strategy == 0:
            selected, sums = _greedy_conflict_graph(perm, rng)
        elif strategy == 1:
            selected, sums = _greedy_dominating_set(perm, rng)
        elif strategy == 2:
            # Reverse order greedy
            perm.reverse()
            selected, sums = _greedy_conflict_graph(perm, rng)
        else:
            # Weighted greedy: prefer smaller numbers (they create smaller sums, blocking more)
            weights = {x: 1.0 / (x + 1) for x in range(1, N + 1)}
            selected, sums = _greedy_weighted(perm, rng, weights)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)
            best_sums = set(sums)
            
        if best_size <= 18:
            break

    # Phase 2: Local search on the best found
    if best_result is not None:
        remaining = DEADLINE - time.time()
        if remaining > 3:
            t_limit = remaining - 3
            new_size, new_sel, new_sums = _local_search(best_result, best_sums, rng, t_limit)
            if new_size < best_size:
                best_size = new_size
                best_result = new_sel
                best_sums = new_sums

    # Phase 3: Triple removal search if time remains
    if best_result is not None and time.time() < DEADLINE - 8:
        t_rem = DEADLINE - time.time() - 8
        if t_rem > 2 and len(best_result) <= 25:
            new_size, new_sel, new_sums = _triple_removal_search(best_result, best_sums, rng, t_rem)
            if new_size < best_size:
                best_size = new_size
                best_result = new_sel
                best_sums = new_sums

    # Phase 4: More restarts if time remains
    if time.time() < DEADLINE - 10 and best_size > 18:
        while time.time() < DEADLINE - 5:
            perm = list(range(1, N + 1))
            rng.shuffle(perm)
            
            # Randomly choose strategy
            r = rng.random()
            if r < 0.3:
                selected, sums = _greedy_conflict_graph(perm, rng)
            elif r < 0.6:
                selected, sums = _greedy_dominating_set(perm, rng)
            else:
                weights = {x: 1.0 / (x + 1) for x in range(1, N + 1)}
                selected, sums = _greedy_weighted(perm, rng, weights)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
                
            if len(selected) <= best_size + 1:
                t_rem = DEADLINE - time.time()
                if t_rem > 5:
                    new_size, new_sel, new_sums = _local_search(selected, sums, rng, min(2.0, t_rem - 5))
                    if new_size < best_size:
                        best_size = new_size
                        best_result = new_sel
                        best_sums = new_sums

    if best_result is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            if 2 * x in sums: continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict: continue
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
