"""Combined approach: Fast bitmask Sidon + blocking greedy + local search reduction."""
import random
import time

N = 256

def _build_sum_mask(sel):
    """Build a bitmask of all unordered sums a+b with a<=b from sel."""
    mask = 0
    n = len(sel)
    for i in range(n):
        a = sel[i]
        mask |= (1 << (2 * a))
        for j in range(i + 1, n):
            mask |= (1 << (a + sel[j]))
    return mask

def _is_maximal(sel, sum_mask=None):
    """Check if sel is inclusion-maximal: no missing element can be added."""
    if not sel:
        return False
    if sum_mask is None:
        sum_mask = _build_sum_mask(sel)
    sel_set = set(sel)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        if (sum_mask >> (2 * x)) & 1:
            continue
        can_add = True
        for a in sel:
            if (sum_mask >> (a + x)) & 1:
                can_add = False
                break
        if can_add:
            return False
    return True

def _can_add_x(sel, sum_mask, x):
    """Check if x can be added to sel (given its sum_mask)."""
    if (sum_mask >> (2 * x)) & 1:
        return False
    for a in sel:
        if (sum_mask >> (a + x)) & 1:
            return False
    return True

def _greedy_with_blocking(random_seed=42, time_budget=2.0):
    """Randomized greedy that picks elements with high blocking power."""
    start = time.time()
    rng = random.Random(random_seed)
    selected = []
    sum_mask = 0
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() - start > time_budget:
            break
        
        # Find all addable candidates
        addable = [x for x in remaining if _can_add_x(selected, sum_mask, x)]
        if not addable:
            break
        
        # Pick the element that blocks the most others
        # For efficiency, evaluate a sample if too many candidates
        if len(addable) > 15:
            candidates = rng.sample(addable, 15)
        else:
            candidates = addable
        
        best_x = None
        best_blocked = -1
        
        for x in candidates:
            # Simulate adding x and count how many remaining become non-addable
            new_mask = sum_mask
            new_mask |= (1 << (2 * x))
            for a in selected:
                new_mask |= (1 << (a + x))
            
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                if not _can_add_x(selected, new_mask, y):
                    blocked += 1
            
            if blocked > best_blocked:
                best_blocked = blocked
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        sum_mask |= (1 << (2 * best_x))
        for a in selected[:-1]:
            sum_mask |= (1 << (a + best_x))
        remaining.discard(best_x)
        
        # Early exit if maximal
        if _is_maximal(selected, sum_mask):
            break
    
    return selected, sum_mask

def _local_search_reduce(sel, time_budget=5.0):
    """Try to reduce the size by removing elements that don't break maximality."""
    start = time.time()
    cur = sorted(sel)
    cur_mask = _build_sum_mask(cur)
    
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for i in range(len(cur)):
            if time.time() - start > time_budget:
                break
            x_remove = cur[i]
            new_sel = cur[:i] + cur[i+1:]
            if not new_sel:
                continue
            new_mask = _build_sum_mask(new_sel)
            if _is_maximal(new_sel, new_mask):
                cur = new_sel
                cur_mask = new_mask
                improved = True
                break
    return cur

def construct():
    deadline = time.time() + 25  # Safety margin for 30s limit
    
    best_set = None
    best_len = N + 1
    
    # Phase 1: Multiple randomized greedy runs
    for seed in range(500):
        if time.time() > deadline - 10:
            break
        
        sel, mask = _greedy_with_blocking(random_seed=seed, time_budget=0.5)
        
        if sel and _is_maximal(sel, mask):
            if len(sel) < best_len:
                best_len = len(sel)
                best_set = sel
                if best_len <= 12:
                    reduced = _local_search_reduce(best_set, time_budget=1.0)
                    if len(reduced) < best_len:
                        best_len = len(reduced)
                        best_set = reduced
                    if best_len <= 10:
                        break
    
    # Fallback: Ascending greedy
    if best_set is None:
        selected = []
        sum_mask = 0
        for x in range(1, N + 1):
            if _can_add_x(selected, sum_mask, x):
                selected.append(x)
                sum_mask |= (1 << (2 * x))
                for a in selected[:-1]:
                    sum_mask |= (1 << (a + x))
        if _is_maximal(selected, sum_mask):
            best_set = selected
        else:
            cur = selected[:]
            cur_mask = sum_mask
            while True:
                addable = [x for x in range(1, N+1) if x not in set(cur) and _can_add_x(cur, cur_mask, x)]
                if not addable:
                    break
                x = addable[0]
                cur.append(x)
                cur_mask |= (1 << (2 * x))
                for a in cur[:-1]:
                    cur_mask |= (1 << (a + x))
            best_set = cur
            best_len = len(best_set)

    # Phase 2: Local search on best
    if time.time() < deadline - 3:
        reduced = _local_search_reduce(best_set, time_budget=deadline - time.time() - 3)
        if len(reduced) < best_len:
            best_set = reduced
            best_len = len(reduced)

    return best_set
