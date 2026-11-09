"""Improved program: graph-based structural optimization for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 27  # 27s safety margin (evaluator kills at 30s)

# EVOLVE-BLOCK-START

def _build_sum_set(selected):
    """Build the set of all unordered pair sums (including diagonal) for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set (all unordered pair sums distinct)."""
    sums = _build_sum_set(selected)
    return len(sums) == len(selected) * (len(selected) + 1) // 2

def _is_maximal(selected, sums=None):
    """Check if selected is inclusion-maximal: no missing element can be added."""
    if sums is None:
        sums = _build_sum_set(selected)
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        # x can be added iff 2*x not in sums AND for all a in selected, x+a not in sums
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _domination_score(selected, sums=None):
    """Count how many missing elements are blocked by selected.
    Higher is better (closer to N - len(selected))."""
    if sums is None:
        sums = _build_sum_set(selected)
    sset = set(selected)
    blocked = 0
    for x in range(1, N + 1):
        if x in sset:
            continue
        if 2 * x in sums:
            blocked += 1
            continue
        for a in selected:
            if (x + a) in sums:
                blocked += 1
                break
    return blocked

def _greedy_construct(order):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums

def _greedy_randomized(rng):
    """Randomized greedy with diverse ordering strategies."""
    base = list(range(1, N + 1))
    strategy = rng.random()
    if strategy < 0.2:
        order = sorted(base)
    elif strategy < 0.4:
        order = sorted(base, reverse=True)
    elif strategy < 0.6:
        order = base[:]
        rng.shuffle(order)
    elif strategy < 0.8:
        mod = rng.choice([2, 3, 4, 5, 7, 8, 16])
        order = sorted(base, key=lambda x: x % mod)
    else:
        # Random with bias toward extremes
        order = base[:]
        rng.shuffle(order)
        # Move some small and large elements to front
        smalls = [x for x in order if x <= N // 4]
        larges = [x for x in order if x > 3 * N // 4]
        rest = [x for x in order if N // 4 < x <= 3 * N // 4]
        rng.shuffle(smalls)
        rng.shuffle(larges)
        order = smalls[:rng.randint(2, 6)] + larges[:rng.randint(2, 6)] + rest
    return _greedy_construct(order)

def _try_remove_element(selected, sums, idx):
    """Try removing element at index idx. Returns (new_selected, new_sums, is_maximal)."""
    new_sel = selected[:idx] + selected[idx+1:]
    new_sums = set()
    for i in range(len(new_sel)):
        for j in range(i, len(new_sel)):
            new_sums.add(new_sel[i] + new_sel[j])
    maximal = _is_maximal(new_sel, new_sums)
    return new_sel, new_sums, maximal

def _local_search_remove(selected, sums, rng, deadline):
    """Iteratively try to remove elements while maintaining maximality."""
    cur_sel = list(selected)
    cur_sums = set(sums)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle removal order
        indices = list(range(len(cur_sel)))
        rng.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            new_sel, new_sums, maximal = _try_remove_element(cur_sel, cur_sums, i)
            if maximal:
                cur_sel = new_sel
                cur_sums = new_sums
                improved = True
                break  # Restart after a removal
    return cur_sel, cur_sums

def _swap_search(selected, sums, rng, deadline):
    """Try swaps: remove one element, add another, then try removals."""
    cur_sel = list(selected)
    cur_sums = set(sums)
    best_sel = list(cur_sel)
    best_sums = set(cur_sums)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        i = rng.randrange(len(cur_sel))
        # Remove element i
        new_sel = cur_sel[:i] + cur_sel[i+1:]
        new_sums = set()
        for a in range(len(new_sel)):
            for b in range(a, len(new_sel)):
                new_sums.add(new_sel[a] + new_sel[b])
        
        # Try adding a random element
        candidates = [x for x in range(1, N + 1) if x not in new_sel]
        if not candidates:
            continue
        # Bias toward elements that might block more
        cand = rng.choice(candidates)
        added = [2 * cand] + [cand + a for a in new_sel]
        if len(set(added)) != len(added) or new_sums.intersection(added):
            continue
        
        new_sel2 = new_sel + [cand]
        new_sums2 = new_sums.union(added)
        
        if _is_maximal(new_sel2, new_sums2):
            # Try removals from this new state
            ls_sel, ls_sums = _local_search_remove(new_sel2, new_sums2, rng, min(deadline, time.time() + 1.0))
            if len(ls_sel) < len(best_sel):
                best_sel = ls_sel
                best_sums = ls_sums
                cur_sel = ls_sel
                cur_sums = ls_sums
    
    return best_sel, best_sums

def _structured_construction(rng):
    """Try structured constructions that might yield smaller maximal sets."""
    results = []
    
    # Strategy 1: Arithmetic progressions with perturbations
    for start in [1, 2, 3]:
        for step in [4, 5, 6, 7, 8]:
            base = [start + k * step for k in range(N // step + 1) if start + k * step <= N]
            if len(base) >= 3:
                # Add some random elements
                extra = [x for x in range(1, N + 1) if x not in base]
                rng.shuffle(extra)
                order = base + extra[:rng.randint(0, 3)]
                sel, sums = _greedy_construct(order)
                if _is_maximal(sel, sums):
                    results.append((sel, sums))
    
    # Strategy 2: Cluster-based - pick a few well-spread elements
    for _ in range(5):
        # Pick 3-5 anchor points spread across the range
        anchors = sorted(rng.sample(range(1, N + 1), rng.randint(3, 5)))
        # Fill in with elements that interact well with anchors
        order = anchors[:]
        remaining = [x for x in range(1, N + 1) if x not in set(anchors)]
        rng.shuffle(remaining)
        order += remaining
        sel, sums = _greedy_construct(order)
        if _is_maximal(sel, sums):
            results.append((sel, sums))
    
    # Strategy 3: Power-of-2 based with modifications
    powers = [2**k for k in range(1, 9) if 2**k <= N]
    for _ in range(3):
        modified = powers[:]
        # Replace or add some elements
        for _ in range(rng.randint(0, 3)):
            idx = rng.randrange(len(modified))
            modified[idx] = rng.randint(1, N)
        # Add a few random elements
        for _ in range(rng.randint(0, 2)):
            modified.append(rng.randint(1, N))
        # Deduplicate
        modified = list(set(modified))
        rng.shuffle(modified)
        sel, sums = _greedy_construct(modified)
        if _is_maximal(sel, sums):
            results.append((sel, sums))
    
    return results

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    deadline = time.time() + 25  # 25s to leave margin
    
    # Phase 1: Structured constructions
    if time.time() < deadline - 15:
        structured = _structured_construction(rng)
        for sel, sums in structured:
            if len(sel) < best_size:
                best_size = len(sel)
                best_result = list(sel)
                best_sums = set(sums)
    
    # Phase 2: Randomized greedy with local search
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        selected, sums = _greedy_randomized(rng)
        
        if len(selected) >= best_size:
            continue
        
        if _is_maximal(selected, sums):
            best_size = len(selected)
            best_result = list(selected)
            best_sums = set(sums)
            
            # Apply local search
            if time.time() < deadline - 3:
                ls_sel, ls_sums = _local_search_remove(selected, sums, rng, min(deadline - 2, time.time() + 2.0))
                if len(ls_sel) < best_size:
                    best_size = len(ls_sel)
                    best_result = ls_sel
                    best_sums = ls_sums
                
                # Try swap search
                if time.time() < deadline - 2:
                    sw_sel, sw_sums = _swap_search(selected, sums, rng, min(deadline - 1, time.time() + 1.5))
                    if len(sw_sel) < best_size:
                        best_size = len(sw_sel)
                        best_result = sw_sel
                        best_sums = sw_sums
            
            if best_size <= 12:
                break
    
    # Phase 3: Final local search on best result
    if best_result is not None and time.time() < deadline - 1:
        final_sel, final_sums = _local_search_remove(best_result, best_sums, rng, deadline - 0.5)
        if len(final_sel) < best_size:
            best_size = len(final_sel)
            best_result = final_sel
            best_sums = final_sums
    
    # Phase 4: Additional swap search on best
    if best_result is not None and time.time() < deadline - 0.5:
        sw_sel, sw_sums = _swap_search(best_result, best_sums, rng, deadline - 0.2)
        if len(sw_sel) < best_size:
            best_size = len(sw_sel)
            best_result = sw_sel
            best_sums = sw_sums
    
    if best_result is None:
        # Fallback: deterministic ascending greedy
        selected, sums = _greedy_construct(list(range(1, N + 1)))
        best_result = selected
    
    return best_result

# EVOLVE-BLOCK-END
