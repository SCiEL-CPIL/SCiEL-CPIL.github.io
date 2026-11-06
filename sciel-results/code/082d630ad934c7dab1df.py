"""
Combined approach: Fast bitmask Sidon checks + randomized greedy + local search
with a vertex-cover-inspired blocking heuristic.

The key structural idea from the current program is the bitmask-accelerated
Sidon/maximality checks and the randomized greedy with local search reduction.
The refinement from the inspirations is to explicitly model the "blocking power"
of each candidate as a degree in a conflict graph: when we add an element x,
it blocks other elements y if adding y would create a duplicate sum. We use a
greedy set-cover style heuristic: at each step, pick the element that blocks
the most remaining candidates (highest degree in the induced conflict subgraph),
which tends to force saturation quickly and minimize the final set size.

We also add a swap-based local search (remove one, add another) to escape local
minima in the reduction phase, and a final polish pass with tighter time budget.
"""
import random
import time

N = 256
MAX_SUM = 2 * N  # 512

def _build_sum_mask(sel):
    """Build a bitmask of all unordered sums a+b with a<=b from sel."""
    mask = 0
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            mask |= (1 << s)
    return mask

def _is_maximal(sel):
    """Check if sel is inclusion-maximal: no missing element can be added."""
    if not sel:
        return False
    sel_set = set(sel)
    sum_mask = _build_sum_mask(sel)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        
        # Check if x can be added
        # 1. 2*x must not be in sum_mask
        if (sum_mask >> (2 * x)) & 1:
            continue
        
        # 2. a + x must not be in sum_mask for any a in sel
        can_add = True
        for a in sel:
            if (sum_mask >> (a + x)) & 1:
                can_add = False
                break
        
        if can_add:
            return False  # Can add x, so not maximal
            
    return True

def _can_add_x(sel, sum_mask, x):
    """Check if x can be added to sel (given its sum_mask)."""
    if (sum_mask >> (2 * x)) & 1:
        return False
    for a in sel:
        if (sum_mask >> (a + x)) & 1:
            return False
    return True

def _greedy_maximal(random_seed=42, time_budget=3.0):
    """
    Randomized greedy to build a maximal Sidon set using a vertex-cover-inspired
    blocking heuristic: at each step, pick the element that blocks the most
    remaining candidates (highest degree in the conflict subgraph).
    """
    rng = random.Random(random_seed)
    start = time.time()
    
    selected = []
    sum_mask = 0
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() - start > time_budget:
            break
        
        # Find all addable candidates
        addable = []
        for x in remaining:
            if _can_add_x(selected, sum_mask, x):
                addable.append(x)
        
        if not addable:
            break
        
        # Heuristic: Pick the element that blocks the most others.
        # This is analogous to picking the highest-degree node in the
        # induced conflict subgraph of remaining candidates.
        best_x = None
        best_score = -1
        
        # If many addable, sample to save time
        if len(addable) > 30:
            candidates = rng.sample(addable, 30)
        else:
            candidates = addable
            
        for x in candidates:
            # Simulate adding x
            new_sum_mask = sum_mask
            for a in selected:
                new_sum_mask |= (1 << (a + x))
            new_sum_mask |= (1 << (2 * x))
            
            # Count how many remaining elements become non-addable
            # after adding x. This is the "degree" in the conflict graph.
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y is still addable with new_sum_mask
                if _can_add_x(selected, new_sum_mask, y):
                    pass  # y is still addable
                else:
                    blocked += 1
                    
            # We want to maximize blocked to force saturation quickly
            if blocked > best_score:
                best_score = blocked
                best_x = x
            elif blocked == best_score and best_x is not None:
                # Tie-break: prefer smaller x for determinism in some cases
                if x < best_x:
                    best_x = x
                
        if best_x is None:
            break
            
        selected.append(best_x)
        for a in selected[:-1]:
            sum_mask |= (1 << (a + best_x))
        sum_mask |= (1 << (2 * best_x))
        remaining.discard(best_x)
        
        # Check if maximal
        if _is_maximal(selected):
            break
            
    return selected

def _local_search_reduce(sel, time_budget=5.0):
    """
    Try to reduce the size of a maximal Sidon set by:
    1. Removing elements if the set remains maximal.
    2. Swapping elements (remove one, add another) to find better configurations.
    """
    start = time.time()
    rng = random.Random(12345)
    best = sorted(sel)
    
    while True:
        if time.time() - start > time_budget:
            break
            
        improved = False
        cur = sorted(best)
        
        # Phase 1: Try removing each element and check if the result is still maximal
        for i in range(len(cur)):
            if time.time() - start > time_budget:
                break
                
            x_remove = cur[i]
            new_sel = [x for x in cur if x != x_remove]
            
            # Check if new_sel is still maximal
            if _is_maximal(new_sel):
                best = new_sel
                improved = True
                break  # Restart loop with smaller set
                
        if improved:
            continue
        
        # Phase 2: Try swaps (remove one, add another) to find better configurations
        cur = sorted(best)
        swap_improved = False
        for i in range(len(cur)):
            if time.time() - start > time_budget:
                break
                
            x_remove = cur[i]
            new_sel = [x for x in cur if x != x_remove]
            new_sel_set = set(new_sel)
            new_sum_mask = _build_sum_mask(new_sel)
            
            # Try adding random candidates to see if we get a better configuration
            missing = [x for x in range(1, N + 1) if x not in new_sel_set]
            rng.shuffle(missing)
            
            for x_add in missing[:20]:
                if time.time() - start > time_budget:
                    break
                    
                if not _can_add_x(new_sel, new_sum_mask, x_add):
                    continue
                
                new_final = sorted(new_sel + [x_add])
                # Check if this new set is maximal
                if _is_maximal(new_final):
                    # Same size, but might enable future reductions
                    # Try to reduce from here
                    for j in range(len(new_final)):
                        x_try_remove = new_final[j]
                        candidate = [x for x in new_final if x != x_try_remove]
                        if _is_maximal(candidate):
                            best = candidate
                            swap_improved = True
                            break
                    if swap_improved:
                        break
            
            if swap_improved:
                break
        
        if not swap_improved:
            break
            
    return best

def construct():
    deadline = time.time() + 25  # Safety margin for 30s limit
    
    best_set = None
    best_len = N + 1
    
    # Phase 1: Multiple randomized greedy runs
    for seed in range(500):
        if time.time() > deadline - 10:
            break
            
        sel = _greedy_maximal(random_seed=seed, time_budget=3.0)
        
        if sel and _is_maximal(sel):
            if len(sel) < best_len:
                best_len = len(sel)
                best_set = sel
                
                # If we find a very small one, try to reduce further
                if best_len <= 12:
                    reduced = _local_search_reduce(best_set, time_budget=2.0)
                    if len(reduced) < best_len:
                        best_len = len(reduced)
                        best_set = reduced
                        
                    if best_len <= 9:
                        break
    
    # Fallback: Ascending greedy
    if best_set is None:
        selected = []
        sum_mask = 0
        for x in range(1, N + 1):
            if _can_add_x(selected, sum_mask, x):
                selected.append(x)
                for a in selected[:-1]:
                    sum_mask |= (1 << (a + x))
                sum_mask |= (1 << (2 * x))
        
        if _is_maximal(selected):
            best_set = selected
        else:
            # Force maximality
            cur = selected
            cur_mask = _build_sum_mask(cur)
            while True:
                addable = [x for x in range(1, N+1) if x not in set(cur) and _can_add_x(cur, cur_mask, x)]
                if not addable:
                    break
                x = addable[0]
                cur.append(x)
                for a in cur[:-1]:
                    cur_mask |= (1 << (a + x))
                cur_mask |= (1 << (2 * x))
            best_set = cur

    # Phase 2: Local search on best with remaining time
    remaining_time = deadline - time.time() - 2
    if remaining_time > 1:
        reduced = _local_search_reduce(best_set, time_budget=remaining_time)
        if len(reduced) < best_len:
            best_set = reduced
            best_len = len(reduced)

    return best_set
