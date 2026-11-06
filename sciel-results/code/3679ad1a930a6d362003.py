import random
import time

N = 256


def _compute_sums(sel):
    """Compute set of all pairwise sums a+b with a<=b."""
    sums = set()
    n = len(sel)
    for i in range(n):
        a = sel[i]
        for j in range(i, n):
            sums.add(a + sel[j])
    return sums


def _is_maximal_fast(sel, sums):
    """Check if sel is inclusion-maximal in [1, N]."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        blocked = False
        for a in sel:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True


def _can_add(x, sel, sums):
    """Check if x can be added to sel maintaining Sidon property."""
    if 2 * x in sums:
        return False
    for a in sel:
        if x + a in sums:
            return False
    return True


def _build_conflict_graph():
    """
    Build a conflict graph where edge (i,j) exists if i and j cannot coexist
    in a Sidon set. Two elements i,j conflict if 2*i == 2*j (impossible for distinct)
    or if there exist a,b in some set such that i+a = j+b. 
    
    For the initial graph (empty set), two elements i,j conflict if 2*i == 2*j 
    (never for distinct) or if i+j creates a duplicate with some other pair.
    
    Actually, for an empty set, no two elements conflict initially. Conflicts
    arise as we build the set. So we need a dynamic approach.
    
    Instead, let's define: for a given current set S with sums, an element x 
    "blocks" element y if adding x to S would make y non-addable.
    """
    pass


def _greedy_max_degree(order, sel_init=None, sums_init=None):
    """
    Greedy construction: at each step, pick the candidate with the highest
    'blocking degree' - i.e., the element that would block the most other
    candidates if added.
    """
    if sel_init is None:
        sel = []
        sums = set()
    else:
        sel = sel_init[:]
        sums = sums_init.copy()
    
    sel_set = set(sel)
    candidates = [x for x in order if x not in sel_set]
    
    while candidates:
        # Filter candidates to only those that can be added
        valid_candidates = []
        for x in candidates:
            if _can_add(x, sel, sums):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # For each valid candidate, compute how many other valid candidates it would block
        best_x = None
        best_block_count = -1
        for x in valid_candidates:
            block_count = 0
            for y in valid_candidates:
                if y == x:
                    continue
                # Check if y can be added to sel ∪ {x}
                # y can be added if: 2*y not in sums∪{2*x} AND for all a in sel∪{x}: y+a not in sums∪{2*x, x+a}
                can_add_y = True
                if 2 * y == 2 * x:  # y == x, already skipped
                    can_add_y = False
                elif 2 * y in sums:
                    can_add_y = False
                else:
                    # Check against new sums from x
                    if x + y in sums or x + y == 2 * x:
                        can_add_y = False
                    else:
                        for a in sel:
                            if y + a in sums:
                                can_add_y = False
                                break
                            if y + a == 2 * x:
                                can_add_y = False
                                break
                            if y + a == x + a:  # y == x, handled
                                can_add_y = False
                                break
                if not can_add_y:
                    block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        sel.append(best_x)
        sums.add(2 * best_x)
        for a in sel:
            sums.add(best_x + a)
        candidates.remove(best_x)
    
    return sel, sums


def _try_remove(sel, sums):
    """Try to remove an element while maintaining maximality."""
    n = len(sel)
    for idx in range(n):
        x = sel[idx]
        new_sel = sel[:idx] + sel[idx+1:]
        new_sums = _compute_sums(new_sel)
        if _is_maximal_fast(new_sel, new_sums):
            return new_sel, new_sums, x
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 28.0  # Safety margin before 30s kill

    best_sel = None
    best_size = N + 1

    # Phase 1: Ascending and descending greedy with max-degree selection
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        if time.time() > deadline:
            break
        sel, sums = _greedy_max_degree(order)
        if _is_maximal_fast(sel, sums) and len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            
            # Local search: try to remove elements
            if time.time() < deadline - 2:
                improved = True
                while improved:
                    if time.time() > deadline - 2:
                        break
                    improved = False
                    result = _try_remove(sel, sums)
                    if result is not None:
                        sel, sums, _ = result
                        improved = True
                        if len(sel) < best_size:
                            best_size = len(sel)
                            best_sel = sel[:]

    # Phase 2: Randomized greedy with max-degree priority
    rng = random.Random(42)
    trials = 0
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        sel, sums = _greedy_max_degree(order)
        
        if _is_maximal_fast(sel, sums) and len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            
            # Local search: try to remove elements
            if time.time() < deadline - 3:
                improved = True
                while improved:
                    if time.time() > deadline - 3:
                        break
                    improved = False
                    result = _try_remove(sel, sums)
                    if result is not None:
                        sel, sums, _ = result
                        improved = True
                        if len(sel) < best_size:
                            best_size = len(sel)
                            best_sel = sel[:]
        
        trials += 1
    
    # Phase 3: Aggressive local search on best candidate
    if best_sel is not None and time.time() < deadline - 2:
        sel = best_sel[:]
        sums = _compute_sums(sel)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for attempt in range(200):
                if time.time() > deadline - 2:
                    break
                result = _try_remove(sel, sums)
                if result is not None:
                    sel, sums, _ = result
                    improved = True
                    if len(sel) < best_size:
                        best_size = len(sel)
                        best_sel = sel[:]
    
    if best_sel is None:
        # Fallback: simple ascending greedy
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(x, sel, sums):
                sel.append(x)
                sums.add(2 * x)
                for a in sel:
                    sums.add(x + a)
        best_sel = sel

    return best_sel
