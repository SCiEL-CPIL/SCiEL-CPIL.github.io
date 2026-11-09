"""Improved program: search-based approach for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 25  # 25s safety margin (evaluator kills at 30s)

# EVOLVE-BLOCK-START
def _is_maximal_fast(selected, sums):
    """Fast inclusion maximality check using precomputed sum membership.
    
    For each missing x, x is blocked iff:
      (a) 2*x is in sums, OR
      (b) there exists a in selected such that x+a is in sums, OR
      (c) there exist a,b in selected with a < b such that x+a == x+b (impossible since a!=b),
          or x+a == 2*x (i.e., a == x, but x not in selected), 
          or x+a == x+b for a!=b (impossible).
    Actually internal collision: x+a_i == x+a_j implies a_i == a_j, impossible.
    x+a == 2*x implies a == x, but x not in selected. So no internal collisions.
    Thus x is blocked iff 2*x in sums OR exists a in selected with (x+a) in sums.
    """
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if (x + a) in sums:
                blocked = True
                break
        if blocked:
            continue
        # x can be added -> not maximal
        return False
    return True

def _greedy_randomize(perm):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in perm:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums

def _greedy_with_bias(perm, bias_small=True):
    """Greedy with bias: prefer smaller or larger elements first to get smaller maximal sets."""
    selected = []
    sums = set()
    # Reorder perm: if bias_small, sort ascending; else descending
    if bias_small:
        perm = sorted(perm)
    else:
        perm = sorted(perm, reverse=True)
    for x in perm:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums

def _greedy_randomized_order(rng):
    """Generate a randomized ordering with some structure to encourage smaller maximal sets."""
    perm = list(range(1, N + 1))
    rng.shuffle(perm)
    # With some probability, sort by some criterion
    r = rng.random()
    if r < 0.25:
        perm.sort()  # ascending
    elif r < 0.5:
        perm.sort(reverse=True)  # descending
    elif r < 0.75:
        # Sort by x mod some number to create structure
        mod = rng.choice([2, 3, 4, 5, 7])
        perm.sort(key=lambda x: x % mod)
    return perm

def _local_search(selected, sums, rng, time_limit):
    """Local search: try removals to minimize size while maintaining maximality."""
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    cur_sel = list(selected)
    cur_sums = set(sums)
    start = time.time()
    
    # First, try all single removals
    for i in range(len(cur_sel)):
        if time.time() - start > time_limit or time.time() > DEADLINE:
            break
        new_sel = cur_sel[:i] + cur_sel[i+1:]
        new_sums = set()
        for a in range(len(new_sel)):
            for b in range(a, len(new_sel)):
                new_sums.add(new_sel[a] + new_sel[b])
        if _is_maximal_fast(new_sel, new_sums):
            cur_sel = new_sel
            cur_sums = new_sums
            if len(cur_sel) < best_size:
                best_size = len(cur_sel)
                best_sel = list(cur_sel)
                best_sums = set(cur_sums)
            # After a successful removal, restart the scan
            i = -1  # Reset loop index
    
    # Now try swaps: remove one, add another
    for _ in range(50):  # Limited number of swap attempts
        if time.time() - start > time_limit or time.time() > DEADLINE:
            break
        i = rng.randrange(len(cur_sel))
        new_sel = cur_sel[:i] + cur_sel[i+1:]
        new_sums = set()
        for a in range(len(new_sel)):
            for b in range(a, len(new_sel)):
                new_sums.add(new_sel[a] + new_sel[b])
        # Try to find an element we can add back that maintains maximality and doesn't increase size
        # Actually for a swap, we remove one and add one, so size stays same.
        # We want to find a swap that allows further removals later.
        # For now, just try random additions and see if maximality is maintained
        candidates = [x for x in range(1, N+1) if x not in new_sel]
        if not candidates:
            continue
        cand = rng.choice(candidates)
        added = [2*cand] + [cand + a for a in new_sel]
        if len(set(added)) != len(added) or new_sums.intersection(added):
            continue
        new_sel2 = new_sel + [cand]
        new_sums2 = new_sums.union(added)
        if _is_maximal_fast(new_sel2, new_sums2):
            cur_sel = new_sel2
            cur_sums = new_sums2
            # After swap, try removals again
            for j in range(len(cur_sel)):
                if time.time() - start > time_limit or time.time() > DEADLINE:
                    break
                test_sel = cur_sel[:j] + cur_sel[j+1:]
                test_sums = set()
                for a in range(len(test_sel)):
                    for b in range(a, len(test_sel)):
                        test_sums.add(test_sel[a] + test_sel[b])
                if _is_maximal_fast(test_sel, test_sums):
                    cur_sel = test_sel
                    cur_sums = test_sums
                    if len(cur_sel) < best_size:
                        best_size = len(cur_sel)
                        best_sel = list(cur_sel)
                        best_sums = set(cur_sums)
                    j = -1  # Reset
    
    if best_size < len(selected):
        return best_size, best_sel, best_sums
    return len(selected), list(selected), set(sums)

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    deadline = time.time() + 27  # 27s to leave margin
    
    # Phase 1: Try various greedy strategies
    strategies = [
        lambda: _greedy_randomize(list(range(1, N+1))),
        lambda: _greedy_with_bias(list(range(1, N+1)), bias_small=True),
        lambda: _greedy_with_bias(list(range(1, N+1)), bias_small=False),
    ]
    
    # Try structured greedy first
    for strat in strategies:
        if time.time() > deadline:
            break
        selected, sums = strat()
        if len(selected) < best_size and _is_maximal_fast(selected, sums):
            best_size = len(selected)
            best_result = list(selected)
    
    # Phase 2: Randomized greedy with local search
    while time.time() < deadline - 2:
        perm = _greedy_randomized_order(rng)
        selected, sums = _greedy_randomize(perm)
        
        if len(selected) >= best_size:
            continue
        
        if _is_maximal_fast(selected, sums):
            best_size = len(selected)
            best_result = list(selected)
            
            # Apply local search to try to reduce further
            if time.time() < deadline - 3:
                ls_size, ls_sel, ls_sums = _local_search(selected, sums, rng, time_limit=2.0)
                if ls_size < best_size:
                    best_size = ls_size
                    best_result = ls_sel
            
            if best_size <= 16:
                break
    
    # Phase 3: If we have a result, try more local search with remaining time
    if best_result is not None and time.time() < deadline - 1:
        # Rebuild sums for best_result
        best_sums = set()
        for i in range(len(best_result)):
            for j in range(i, len(best_result)):
                best_sums.add(best_result[i] + best_result[j])
        ls_size, ls_sel, ls_sums = _local_search(best_result, best_sums, rng, time_limit=min(5.0, deadline - time.time() - 0.5))
        if ls_size < best_size:
            best_size = ls_size
            best_result = ls_sel
    
    if best_result is None:
        # Fallback: deterministic greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_result = selected
    
    return best_result
# EVOLVE-BLOCK-END
