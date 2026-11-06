"""Backtracking/greedy blocking-priority search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check all unordered pair sums a+b (a<=b) are distinct."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in s:
                return False
            s.add(t)
    return True

def _compute_sums(selected):
    """Compute set of all pair sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_maximal_fast(selected, sums):
    """
    Fast check: for every missing x in [1,N], verify that adding x would
    create a duplicate sum. Returns True if maximal.
    """
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added without conflict
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok and 2 * x in sums:
            ok = False
        if ok:
            return False  # x could be added, so not maximal
    return True

def _greedy_construct(rng, time_limit, strategy="block"):
    """
    Greedy construction: start empty, repeatedly add the element that
    blocks the most other elements (or random, or first-fit).
    Returns (selected, sums) or None if time exceeded.
    """
    selected = []
    sums = set()
    sel_set = set()
    
    while True:
        if time.time() > time_limit:
            return None
        
        # Find all addable elements
        addable = []
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # Check if x can be added
            ok = True
            for a in selected:
                if x + a in sums:
                    ok = False
                    break
            if ok and 2 * x in sums:
                ok = False
            if ok:
                addable.append(x)
        
        if not addable:
            break
        
        # Score each candidate
        if strategy == "block":
            best_x = None
            best_score = -1
            for x in addable:
                # Count how many addable elements would become blocked
                new_sums = set()
                for a in selected:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                
                blocked_count = 0
                for y in addable:
                    if y == x:
                        continue
                    # Check if y would be blocked after adding x
                    y_blocked = False
                    for a in selected:
                        if y + a in sums:
                            y_blocked = True
                            break
                    if not y_blocked and 2 * y in sums:
                        y_blocked = True
                    if not y_blocked:
                        for a in selected:
                            if y + a in new_sums:
                                y_blocked = True
                                break
                        if not y_blocked and y + x in sums:
                            y_blocked = True
                        if not y_blocked and 2 * y in new_sums:
                            y_blocked = True
                    if y_blocked:
                        blocked_count += 1
                
                if blocked_count > best_score:
                    best_score = blocked_count
                    best_x = x
            
            if best_x is None:
                best_x = addable[0]
        elif strategy == "random":
            best_x = rng.choice(addable)
        elif strategy == "first":
            best_x = addable[0]
        elif strategy == "last":
            best_x = addable[-1]
        elif strategy == "min_block":
            # Pick element that blocks the fewest
            best_x = None
            best_score = float('inf')
            for x in addable:
                new_sums = set()
                for a in selected:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                
                blocked_count = 0
                for y in addable:
                    if y == x:
                        continue
                    y_blocked = False
                    for a in selected:
                        if y + a in sums:
                            y_blocked = True
                            break
                    if not y_blocked and 2 * y in sums:
                        y_blocked = True
                    if not y_blocked:
                        for a in selected:
                            if y + a in new_sums:
                                y_blocked = True
                                break
                        if not y_blocked and y + x in sums:
                            y_blocked = True
                        if not y_blocked and 2 * y in new_sums:
                            y_blocked = True
                    if y_blocked:
                        blocked_count += 1
                
                if blocked_count < best_score:
                    best_score = blocked_count
                    best_x = x
            
            if best_x is None:
                best_x = addable[0]
        else:
            best_x = rng.choice(addable)
        
        # Add best_x
        selected.append(best_x)
        sel_set.add(best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
        sums.add(2 * best_x)
    
    return selected, sums

def _local_search(selected, sums, rng, time_limit):
    """
    Local search: try removing one element and re-adding to see if we can
    get a smaller maximal set. Also try swapping elements.
    Returns (selected, sums) or None if time exceeded.
    """
    selected = list(selected)
    sums = set(sums)
    sel_set = set(selected)
    best_size = len(selected)
    
    # Try removing elements one at a time
    for _ in range(50):
        if time.time() > time_limit:
            break
        
        improved = False
        for i in range(len(selected)):
            if time.time() > time_limit:
                break
            
            # Try removing selected[i]
            removed = selected[i]
            new_selected = selected[:i] + selected[i+1:]
            new_sel_set = set(new_selected)
            
            # Recompute sums
            new_sums = set()
            for j, a in enumerate(new_selected):
                for b in new_selected[j:]:
                    new_sums.add(a + b)
            
            # Check if still Sidon (should be, since subset)
            # Check if maximal
            if not _is_maximal_fast(new_selected, new_sums):
                # Try to make it maximal by adding elements
                can_extend = True
                while can_extend:
                    can_extend = False
                    for x in range(1, N + 1):
                        if x in new_sel_set:
                            continue
                        ok = True
                        for a in new_selected:
                            if x + a in new_sums:
                                ok = False
                                break
                        if ok and 2 * x in new_sums:
                            ok = False
                        if ok:
                            new_selected.append(x)
                            new_sel_set.add(x)
                            for a in new_selected[:-1]:
                                new_sums.add(x + a)
                            new_sums.add(2 * x)
                            can_extend = True
                            break
                
                if len(new_selected) < best_size and _is_maximal_fast(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    sel_set = new_sel_set
                    best_size = len(selected)
                    improved = True
                    break
        
        if not improved:
            # Try random swaps
            if len(selected) < 2:
                break
            i = rng.randint(0, len(selected) - 1)
            removed = selected[i]
            new_selected = selected[:i] + selected[i+1:]
            new_sel_set = set(new_selected)
            
            new_sums = set()
            for j, a in enumerate(new_selected):
                for b in new_selected[j:]:
                    new_sums.add(a + b)
            
            # Try adding a random element
            candidates = [x for x in range(1, N + 1) if x not in new_sel_set]
            if not candidates:
                continue
            
            rng.shuffle(candidates)
            for x in candidates[:20]:
                ok = True
                for a in new_selected:
                    if x + a in new_sums:
                        ok = False
                        break
                if ok and 2 * x in new_sums:
                    ok = False
                if ok:
                    test_selected = new_selected + [x]
                    test_sums = set(new_sums)
                    for a in new_selected:
                        test_sums.add(x + a)
                    test_sums.add(2 * x)
                    
                    if _is_maximal_fast(test_selected, test_sums) and len(test_selected) < best_size:
                        selected = test_selected
                        sums = test_sums
                        sel_set = set(selected)
                        best_size = len(selected)
                        improved = True
                        break
            
            if not improved:
                break
    
    return selected, sums

def _verify_maximal(selected):
    """Verify selected is Sidon and inclusion-maximal."""
    if not _is_sidon(selected):
        return False
    sel_set = set(selected)
    sums = _compute_sums(selected)
    return _is_maximal_fast(selected, sums)

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin before 30s limit
    
    # Get initial upper bound from simple greedy
    simple_selected, simple_sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)
    
    best_set = simple_selected
    best_size = len(best_set)
    
    # Verify it's maximal
    if not _verify_maximal(best_set):
        # Force maximality by adding all possible
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in best_set:
                    added = [x + a for a in best_set] + [2 * x]
                    if len(set(added)) == len(added) and not any(s in simple_sums for s in added):
                        best_set.append(x)
                        simple_sums.update(added)
                        changed = True
    
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Try multiple strategies
    strategies = ["block", "random", "first", "last", "min_block"]
    
    for strategy in strategies:
        if time.time() > deadline - 2.0:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 2.0:
                break
            
            # Start with a random seed
            seed_size = rng.randint(0, 3)
            seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
            
            # Check seed is Sidon
            seed_ok = True
            seed_sums = set()
            for i, a in enumerate(seed_elements):
                for b in seed_elements[i:]:
                    s = a + b
                    if s in seed_sums:
                        seed_ok = False
                        break
                    seed_sums.add(s)
                if not seed_ok:
                    break
            
            if not seed_ok:
                continue
            
            # Greedy construct from seed
            selected = list(seed_elements)
            sums = set(seed_sums)
            sel_set = set(selected)
            
            # Continue greedy
            while True:
                if time.time() > deadline - 1.0:
                    break
                
                addable = []
                for x in range(1, N + 1):
                    if x in sel_set:
                        continue
                    ok = True
                    for a in selected:
                        if x + a in sums:
                            ok = False
                            break
                    if ok and 2 * x in sums:
                        ok = False
                    if ok:
                        addable.append(x)
                
                if not addable:
                    break
                
                if strategy == "block":
                    best_x = None
                    best_score = -1
                    for x in addable:
                        new_sums = set()
                        for a in selected:
                            new_sums.add(x + a)
                        new_sums.add(2 * x)
                        
                        blocked_count = 0
                        for y in addable:
                            if y == x:
                                continue
                            y_blocked = False
                            for a in selected:
                                if y + a in sums:
                                    y_blocked = True
                                    break
                            if not y_blocked and 2 * y in sums:
                                y_blocked = True
                            if not y_blocked:
                                for a in selected:
                                    if y + a in new_sums:
                                        y_blocked = True
                                        break
                                if not y_blocked and y + x in sums:
                                    y_blocked = True
                                if not y_blocked and 2 * y in new_sums:
                                    y_blocked = True
                            if y_blocked:
                                blocked_count += 1
                        
                        if blocked_count > best_score:
                            best_score = blocked_count
                            best_x = x
                    
                    if best_x is None:
                        best_x = addable[0]
                elif strategy == "random":
                    best_x = rng.choice(addable)
                elif strategy == "first":
                    best_x = addable[0]
                elif strategy == "last":
                    best_x = addable[-1]
                elif strategy == "min_block":
                    best_x = None
                    best_score = float('inf')
                    for x in addable:
                        new_sums = set()
                        for a in selected:
                            new_sums.add(x + a)
                        new_sums.add(2 * x)
                        
                        blocked_count = 0
                        for y in addable:
                            if y == x:
                                continue
                            y_blocked = False
                            for a in selected:
                                if y + a in sums:
                                    y_blocked = True
                                    break
                            if not y_blocked and 2 * y in sums:
                                y_blocked = True
                            if not y_blocked:
                                for a in selected:
                                    if y + a in new_sums:
                                        y_blocked = True
                                        break
                                if not y_blocked and y + x in sums:
                                    y_blocked = True
                                if not y_blocked and 2 * y in new_sums:
                                    y_blocked = True
                            if y_blocked:
                                blocked_count += 1
                        
                        if blocked_count < best_score:
                            best_score = blocked_count
                            best_x = x
                    
                    if best_x is None:
                        best_x = addable[0]
                else:
                    best_x = rng.choice(addable)
                
                selected.append(best_x)
                sel_set.add(best_x)
                for a in selected[:-1]:
                    sums.add(best_x + a)
                sums.add(2 * best_x)
            
            # Verify and potentially do local search
            if _verify_maximal(selected):
                if len(selected) < best_size:
                    # Try local search
                    result = _local_search(selected, sums, rng, deadline - 1.0)
                    if result is not None:
                        sel, sm = result
                        if _verify_maximal(sel) and len(sel) < best_size:
                            best_size = len(sel)
                            best_set = sel
                    else:
                        best_size = len(selected)
                        best_set = selected
                elif len(selected) == best_size:
                    # Same size, but try local search anyway
                    result = _local_search(selected, sums, rng, deadline - 1.0)
                    if result is not None:
                        sel, sm = result
                        if _verify_maximal(sel) and len(sel) < best_size:
                            best_size = len(sel)
                            best_set = sel
    
    return best_set
# EVOLVE-BLOCK-END
