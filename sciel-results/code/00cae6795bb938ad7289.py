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

def _can_add(x, selected, sums):
    """Check if x can be added to the Sidon set."""
    if 2 * x in sums:
        return False
    for a in selected:
        if x + a in sums:
            return False
    return True

def _is_maximal_fast(selected, sums):
    """
    Fast check if the set is inclusion-maximal.
    Uses vectorized numpy operations where possible.
    """
    sset = set(selected)
    n = len(selected)
    
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        # Check if x can be added - if yes, not maximal
        if 2 * x in sums:
            continue
        
        # Check all x+a for a in selected
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _greedy_dominating(perm, rng, max_candidates=50):
    """
    Greedy: prioritize elements that block many others.
    Targets inclusion-maximality directly with fast blocking count.
    """
    selected = []
    sums = set()
    remaining = set(perm)
    sset = set()
    
    while remaining and time.time() < DEADLINE - 3:
        best_candidate = None
        best_block_count = -1
        best_new_sums = None
        
        # Sample candidates for speed
        candidates = list(remaining)
        if len(candidates) > max_candidates:
            candidates = rng.sample(candidates, max_candidates)
        
        for x in candidates:
            if not _can_add(x, selected, sums):
                continue
            
            # Compute new sums
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
            
            # Count how many remaining elements this blocks
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    block_count += 1
                else:
                    # Check if y+a is in new_sums for any a in selected+[x]
                    blocked = False
                    for a in selected:
                        if y + a in new_sums:
                            blocked = True
                            break
                    if not blocked and y + x in new_sums:
                        blocked = True
                    if blocked:
                        block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_candidate = x
                best_new_sums = new_sums
        
        if best_candidate is None:
            break
        
        x = best_candidate
        selected.append(x)
        sset.add(x)
        sums = best_new_sums
        remaining.discard(x)
        
        # Remove all blocked elements from remaining
        still_remaining = set()
        for y in remaining:
            if _can_add(y, selected, sums):
                still_remaining.add(y)
        remaining = still_remaining
    
    return selected, sums

def _greedy_lookahead(perm, rng):
    """
    Greedy with 1-step lookahead: when choosing next element,
    consider which choice leads to more future blocks.
    """
    selected = []
    sums = set()
    remaining = set(perm)
    
    while remaining and time.time() < DEADLINE - 3:
        best_candidate = None
        best_score = -1
        best_new_sums = None
        
        candidates = list(remaining)
        if len(candidates) > 35:
            candidates = rng.sample(candidates, 35)
        
        for x in candidates:
            if not _can_add(x, selected, sums):
                continue
            
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
            
            # Score: immediate blocks + potential future blocks
            immediate_blocks = 0
            remaining_after = set()
            for y in remaining:
                if y == x:
                    continue
                if _can_add(y, selected + [x], new_sums):
                    remaining_after.add(y)
                else:
                    immediate_blocks += 1
            
            # Estimate future blocking power: count how many remaining
            # elements have high "blocking potential"
            score = immediate_blocks * 10 - len(remaining_after) * 0.1
            
            if score > best_score:
                best_score = score
                best_candidate = x
                best_new_sums = new_sums
        
        if best_candidate is None:
            break
        
        x = best_candidate
        selected.append(x)
        sums = best_new_sums
        remaining.discard(x)
        
        # Remove blocked
        still_remaining = set()
        for y in remaining:
            if _can_add(y, selected, sums):
                still_remaining.add(y)
        remaining = still_remaining
    
    return selected, sums

def _local_search(selected, sums, rng, time_limit):
    """
    Aggressive local search: try single, double, and triple removals,
    plus swaps. Uses fast maximality check.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    # Cache for fast maximality checks
    max_cache = {}
    
    def _check_maximal(sel, sums_set):
        key = tuple(sorted(sel))
        if key in max_cache:
            return max_cache[key]
        result = _is_maximal_fast(sel, sums_set)
        if len(max_cache) < 5000:
            max_cache[key] = result
        return result
    
    while time.time() - start < time_limit and time.time() < DEADLINE - 2:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        n = len(cur_sel)
        
        # --- Single Removal Pass ---
        for i in range(n):
            if time.time() > DEADLINE - 1:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            removed_sums = {2 * e}
            for a in new_sel:
                removed_sums.add(e + a)
            
            new_sums = cur_sums - removed_sums
            
            if _check_maximal(new_sel, new_sums):
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
            for i in range(n):
                if time.time() > DEADLINE - 1:
                    return best_size, best_sel, best_sums
                for j in range(i + 1, n):
                    if time.time() > DEADLINE - 1:
                        return best_size, best_sel, best_sums
                    
                    new_sel = [x for k, x in enumerate(cur_sel) if k != i and k != j]
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    if _check_maximal(new_sel, new_sums):
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
            # --- Triple Removal Pass (only if set is small enough) ---
            if n <= 30:
                for i in range(n):
                    if time.time() > DEADLINE - 1:
                        return best_size, best_sel, best_sums
                    for j in range(i + 1, n):
                        if time.time() > DEADLINE - 1:
                            return best_size, best_sel, best_sums
                        for k in range(j + 1, n):
                            if time.time() > DEADLINE - 1:
                                return best_size, best_sel, best_sums
                            
                            new_sel = [x for idx, x in enumerate(cur_sel) 
                                      if idx != i and idx != j and idx != k]
                            
                            new_sums = set()
                            for a in range(len(new_sel)):
                                for b in range(a, len(new_sel)):
                                    new_sums.add(new_sel[a] + new_sel[b])
                            
                            if _check_maximal(new_sel, new_sums):
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
                    if improved:
                        break
        
        if not improved:
            # --- Swap Pass: try replacing one element with another ---
            sset = set(cur_sel)
            for i in range(n):
                if time.time() > DEADLINE - 1:
                    return best_size, best_sel, best_sums
                e = cur_sel[i]
                
                # Try several random replacements
                for _ in range(8):
                    if time.time() > DEADLINE - 1:
                        return best_size, best_sel, best_sums
                    
                    # Pick a random element not in cur_sel
                    candidates = [x for x in range(1, N + 1) if x not in sset]
                    if not candidates:
                        break
                    new_e = rng.choice(candidates)
                    
                    new_sel = [x for x in cur_sel if x != e] + [new_e]
                    
                    new_sums = set()
                    valid = True
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            s = new_sel[a] + new_sel[b]
                            if s in new_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        if not valid:
                            break
                    
                    if valid and _check_maximal(new_sel, new_sums):
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
            # Try a few random restarts of local search
            for _ in range(3):
                if time.time() > DEADLINE - 2:
                    break
                # Perturb: remove 2 random elements, then try to re-add
                if len(cur_sel) > 4:
                    remove_indices = rng.sample(range(len(cur_sel)), 2)
                    remove_indices.sort(reverse=True)
                    perturbed = list(cur_sel)
                    for idx in remove_indices:
                        perturbed.pop(idx)
                    
                    p_sums = set()
                    for a in range(len(perturbed)):
                        for b in range(a, len(perturbed)):
                            p_sums.add(perturbed[a] + perturbed[b])
                    
                    if _check_maximal(perturbed, p_sums):
                        selected = perturbed
                        sums = p_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
            
            if not improved:
                break

    return best_size, best_sel, best_sums

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    
    # Phase 1: Multiple greedy constructions with different strategies
    iterations = 0
    while time.time() < DEADLINE - 20:
        iterations += 1
        
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        # Cycle through strategies
        strategy = iterations % 3
        if strategy == 0:
            selected, sums = _greedy_dominating(perm, rng, max_candidates=50)
        elif strategy == 1:
            selected, sums = _greedy_dominating(perm, rng, max_candidates=30)
        else:
            selected, sums = _greedy_lookahead(perm, rng)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)
            best_sums = set(sums)
            
        if best_size <= 18:
            break

    # Phase 2: Local search on the best found
    if best_result is not None:
        remaining = DEADLINE - time.time()
        if remaining > 5:
            t_limit = remaining - 5
            new_size, new_sel, new_sums = _local_search(best_result, best_sums, rng, t_limit)
            if new_size < best_size:
                best_size = new_size
                best_result = new_sel
                best_sums = new_sums

    # Phase 3: More restarts with local search if time remains
    if time.time() < DEADLINE - 10 and best_size > 16:
        while time.time() < DEADLINE - 5:
            perm = list(range(1, N + 1))
            rng.shuffle(perm)
            
            r = rng.random()
            if r < 0.4:
                selected, sums = _greedy_dominating(perm, rng, max_candidates=40)
            elif r < 0.7:
                selected, sums = _greedy_dominating(perm, rng, max_candidates=25)
            else:
                selected, sums = _greedy_lookahead(perm, rng)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
                
            # Run short local search on promising solutions
            if len(selected) <= best_size + 2:
                t_rem = DEADLINE - time.time()
                if t_rem > 4:
                    new_size, new_sel, new_sums = _local_search(
                        selected, sums, rng, min(3.0, t_rem - 4))
                    if new_size < best_size:
                        best_size = new_size
                        best_result = new_sel
                        best_sums = new_sums

    # Final polish: one more local search pass on the best
    if best_result is not None and time.time() < DEADLINE - 3:
        t_rem = DEADLINE - time.time()
        if t_rem > 2:
            new_size, new_sel, new_sums = _local_search(
                best_result, best_sums, rng, t_rem - 2)
            if new_size < best_size:
                best_size = new_size
                best_result = new_sel
                best_sums = new_sums

    if best_result is None:
        # Fallback: simple greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            if _can_add(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
