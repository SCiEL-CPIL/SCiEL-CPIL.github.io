"""Optimized program: Fast bitmask Sidon checks + randomized greedy + simple local search."""
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

def _greedy_maximal(random_seed=42):
    """Randomized greedy to build a maximal Sidon set."""
    rng = random.Random(random_seed)
    selected = []
    sum_mask = 0
    remaining = list(range(1, N + 1))
    rng.shuffle(remaining)
    
    # We want to build a set that is maximal.
    # Strategy: Pick elements that "block" many others.
    # Simple heuristic: Just pick randomly from addable candidates?
    # Or pick the one that minimizes the number of remaining addable candidates?
    
    while True:
        # Find all addable candidates
        addable = []
        for x in remaining:
            if _can_add_x(selected, sum_mask, x):
                addable.append(x)
        
        if not addable:
            break
        
        # Heuristic: Pick the element that blocks the most others
        # For speed, we might just pick randomly if the set is large,
        # but for small N, we can evaluate a few.
        
        # Let's evaluate the "blocking power" of a few candidates
        best_x = None
        best_score = -1
        
        # If many addable, sample a few to save time
        if len(addable) > 20:
            candidates = rng.sample(addable, 20)
        else:
            candidates = addable
            
        for x in candidates:
            # Simulate adding x
            new_sum_mask = sum_mask
            for a in selected:
                new_sum_mask |= (1 << (a + x))
            new_sum_mask |= (1 << (2 * x))
            
            # Count how many remaining elements become non-addable
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y was addable before? 
                # Actually, we just check if y is addable AFTER adding x.
                # If it was NOT addable before, and is NOT addable after, it doesn't count as "newly blocked"
                # But for simplicity, let's just count how many are NOT addable after.
                # Ideally we want to maximize the number of "dead" elements.
                
                # Quick check: is y addable with new_sum_mask?
                if _can_add_x(selected + [x], new_sum_mask, y):
                    # y is still addable
                    pass
                else:
                    blocked += 1
                    
            # We want to maximize blocked? Or minimize remaining addable?
            # Maximizing blocked is good for reaching maximality quickly with small size.
            if blocked > best_score:
                best_score = blocked
                best_x = x
                
        if best_x is None:
            break
            
        selected.append(best_x)
        for a in selected[:-1]:
            sum_mask |= (1 << (a + best_x))
        sum_mask |= (1 << (2 * best_x))
        remaining.remove(best_x)
        
        # Check if maximal
        if _is_maximal(selected):
            break
            
    return selected

def _local_search_reduce(sel, time_budget=5.0):
    """Try to reduce the size of a maximal Sidon set."""
    start = time.time()
    best = sorted(sel)
    
    while True:
        if time.time() - start > time_budget:
            break
            
        improved = False
        cur = sorted(best)
        
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
                
        if not improved:
            break
            
    return best

def construct():
    deadline = time.time() + 25  # Safety margin for 30s limit
    
    best_set = None
    best_len = N + 1
    
    # Phase 1: Multiple randomized greedy runs
    for seed in range(1000):
        if time.time() > deadline - 10:
            break
            
        sel = _greedy_maximal(random_seed=seed)
        
        if sel and _is_maximal(sel):
            # Verify Sidon property (should be true by construction, but just in case)
            # We can skip full Sidon check if we trust the bitmask logic, but it's cheap.
            # Actually, _is_maximal assumes Sidon. Let's ensure it is Sidon.
            # The greedy construction ensures Sidon property.
            
            if len(sel) < best_len:
                best_len = len(sel)
                best_set = sel
                
                # If we find a very small one, we can stop early
                if best_len <= 10:
                    # Try to reduce further
                    reduced = _local_search_reduce(best_set, time_budget=2.0)
                    if len(reduced) < best_len:
                        best_len = len(reduced)
                        best_set = reduced
                        
                    if best_len <= 8:
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
            # This shouldn't happen, but if it does, we need a maximal set.
            # Just extend it? No, we need to return a valid maximal set.
            # The ascending greedy usually produces a maximal set.
            # If not, we force maximality by adding any addable elements.
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

    # Phase 2: Local search on best
    if time.time() < deadline - 2:
        reduced = _local_search_reduce(best_set, time_budget=deadline - time.time() - 2)
        if len(reduced) < best_len:
            best_set = reduced
            best_len = len(reduced)

    return best_set
