"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_sums_set(selected):
    """Compute all pairwise sums a+b for a<=b in selected as a set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon_fast(selected):
    """Fast Sidon check."""
    if len(selected) == 0:
        return True
    sums = _compute_sums_set(selected)
    total = len(selected) * (len(selected) + 1) // 2
    return len(sums) == total

def _is_maximal_fast(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    selected_set = set(selected)
    sums = _compute_sums_set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for s in selected:
            if x + s in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _can_add(selected, sums, selected_set, x):
    """Check if x can be added to selected (Sidon-compatible)."""
    if x in selected_set:
        return False
    if 2 * x in sums:
        return False
    for s in selected:
        if x + s in sums:
            return False
    return True

def _build_conflict_graph():
    """Build conflict graph: edge (a,b) if a and b cannot coexist in Sidon set.
    a,b conflict iff 2a == 2b (impossible for distinct), a+b appears as 2c or d+e for c,d,e in {a,b}...
    Actually: a and b are incompatible if adding both to empty set violates Sidon.
    For a pair {a,b}: sums are 2a, 2b, a+b. They conflict iff 2a==2b (no), 2a==a+b (b=a, no), 2b==a+b (a=b, no).
    So NO pair of distinct elements conflicts in isolation!
    
    The real conflict is: given a partial set S, x conflicts with S if x can't be added.
    For MIS interpretation: we want a small set S such that every x not in S conflicts with S.
    This is a dominating set problem on a hypergraph, not standard MIS.
    
    Alternative: Think of it as - we want to MINIMIZE |S| subject to S being Sidon and maximal.
    Maximal means: for every x not in S, x conflicts with S (can't be added).
    
    Let's use a different approach: greedy elimination.
    Start with all 256 elements. We want to find a small subset that is Sidon and maximal.
    """
    pass

def _greedy_construct(order, initial=None):
    """Greedy Sidon set construction following the given order."""
    if initial is not None:
        selected = list(initial)
        sums = _compute_sums_set(selected)
    else:
        selected = []
        sums = set()
    selected_set = set(selected)
    for x in order:
        if x in selected_set:
            continue
        if _can_add(selected, sums, selected_set, x):
            selected.append(x)
            new_sums = [x + a for a in selected[:-1]] + [2 * x]
            sums.update(new_sums)
            selected_set.add(x)
    return selected

def _forced_additions_v2(core, rng=None):
    """
    Given a Sidon core, greedily add elements to make it maximal.
    Pick the element that blocks the most 'addable' elements.
    """
    if rng is None:
        rng = random.Random()
    
    selected = list(core)
    sums = _compute_sums_set(selected)
    selected_set = set(selected)
    
    def _get_addable():
        addable = []
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if _can_add(selected, sums, selected_set, x):
                addable.append(x)
        return addable
    
    max_iter = 500
    for _ in range(max_iter):
        addable = _get_addable()
        if not addable:
            break
        
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        best_candidate = None
        best_blocked = -1
        best_candidates = []
        
        for cand in candidates:
            if not _can_add(selected, sums, selected_set, cand):
                continue
            
            new_sums_list = [cand + a for a in selected] + [2 * cand]
            new_sums_set = sums | set(new_sums_list)
            new_selected = selected + [cand]
            new_selected_set = selected_set | {cand}
            
            blocked = 0
            for y in addable:
                if y == cand:
                    continue
                if not _can_add(new_selected, new_sums_set, new_selected_set, y):
                    blocked += 1
            
            if blocked > best_blocked:
                best_blocked = blocked
                best_candidates = [cand]
            elif blocked == best_blocked:
                best_candidates.append(cand)
        
        if not best_candidates:
            break
        
        best_candidate = rng.choice(best_candidates)
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        sums = _compute_sums_set(selected)
    
    return selected

def _local_search_reduce(selected, deadline):
    """
    Try to remove elements while maintaining maximality.
    Uses swap moves to escape local minima.
    """
    current = list(selected)
    improved = True
    pass_count = 0
    max_passes = 30
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Phase 1: Try removals
        for x in list(current):
            if time.time() > deadline:
                return current
            new_list = [y for y in current if y != x]
            if len(new_list) == 0:
                continue
            if _is_maximal_fast(new_list):
                current = new_list
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps
        current_set = set(current)
        for x in list(current):
            if time.time() > deadline:
                return current
            without_x = [y for y in current if y != x]
            if len(without_x) == 0:
                continue
            without_x_set = set(without_x)
            sums_x = _compute_sums_set(without_x)
            
            for y in range(1, N + 1):
                if y in without_x_set:
                    continue
                if not _can_add(without_x, sums_x, without_x_set, y):
                    continue
                
                candidate = without_x + [y]
                if _is_maximal_fast(candidate):
                    current = candidate
                    improved = True
                    break
            
            if improved:
                break
    
    return current

def _construct_spread_core(seed_val, target_size=8):
    """Construct a well-spread initial core."""
    rng = random.Random(seed_val)
    core = []
    
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(core) >= target_size:
            break
        if _can_add(core, _compute_sums_set(core), set(core), x):
            core.append(x)
    
    return core

def _mis_greedy_elimination(seed_val, deadline):
    """
    MIS-inspired approach: 
    1. Start with a large Sidon set (greedy ascending).
    2. Iteratively try to remove elements, preferring those whose removal
       creates the fewest new 'addable' elements (i.e., those that are most
       'central' in the conflict structure).
    3. If removal breaks maximality, try swaps.
    """
    rng = random.Random(seed_val)
    
    # Start with a maximal Sidon set
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = _greedy_construct(order)
    
    if not _is_maximal_fast(selected):
        # Make it maximal
        selected = _forced_additions_v2(selected, rng)
    
    current = list(selected)
    improved = True
    max_passes = 50
    
    while improved and time.time() < deadline and max_passes > 0:
        improved = False
        max_passes -= 1
        
        # Try to remove each element, score by how many elements become addable
        best_removal = None
        best_score = float('inf')
        
        for x in list(current):
            if time.time() > deadline:
                break
            new_list = [y for y in current if y != x]
            if len(new_list) == 0:
                continue
            
            # Check if still Sidon (it should be, since subset of Sidon is Sidon)
            if not _is_sidon_fast(new_list):
                continue
            
            # Count how many elements become addable
            new_sums = _compute_sums_set(new_list)
            new_set = set(new_list)
            addable_count = 0
            for y in range(1, N + 1):
                if y in new_set:
                    continue
                if _can_add(new_list, new_sums, new_set, y):
                    addable_count += 1
            
            # We want to minimize addable_count (keep it maximal or close)
            # But we also need the result to actually be maximal
            if _is_maximal_fast(new_list):
                # Perfect removal
                if addable_count < best_score:
                    best_score = addable_count
                    best_removal = x
        
        if best_removal is not None:
            current = [y for y in current if y != best_removal]
            improved = True
            continue
        
        # Try swaps: remove x, add y
        for x in list(current):
            if time.time() > deadline:
                break
            without_x = [y for y in current if y != x]
            if len(without_x) == 0:
                continue
            without_x_set = set(without_x)
            sums_x = _compute_sums_set(without_x)
            
            for y in range(1, N + 1):
                if y in without_x_set:
                    continue
                if not _can_add(without_x, sums_x, without_x_set, y):
                    continue
                
                candidate = without_x + [y]
                if _is_maximal_fast(candidate):
                    current = candidate
                    improved = True
                    break
            
            if improved:
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best = None
    best_size = N + 1
    
    def _update_best(candidate):
        nonlocal best, best_size
        if candidate is None:
            return
        if not _is_sidon_fast(candidate):
            return
        if not _is_maximal_fast(candidate):
            return
        if len(candidate) < best_size:
            best = list(candidate)
            best_size = len(candidate)
    
    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    selected = _greedy_construct(order)
    _update_best(selected)
    
    if time.time() < deadline:
        reduced = _local_search_reduce(best, deadline)
        _update_best(reduced)
    
    # Strategy 2: MIS-style elimination from various starts
    if time.time() < deadline - 5.0:
        for trial in range(200):
            if time.time() > deadline - 2.0:
                break
            result = _mis_greedy_elimination(trial * 7919, deadline)
            _update_best(result)
    
    # Strategy 3: Small cores with forced additions
    if time.time() < deadline - 5.0:
        for trial in range(300):
            if time.time() > deadline - 2.0:
                break
            target_size = 4 + (trial % 7)  # Try cores of size 4-10
            core = _construct_spread_core(trial * 104729, target_size)
            if len(core) == 0:
                continue
            if not _is_sidon_fast(core):
                continue
            
            rng_comp = random.Random(trial * 31337)
            result = _forced_additions_v2(core, rng=rng_comp)
            _update_best(result)
            
            if time.time() < deadline - 2.0 and len(result) <= best_size:
                reduced = _local_search_reduce(result, deadline)
                _update_best(reduced)
    
    # Strategy 4: Random orders with greedy
    if time.time() < deadline - 3.0:
        for trial in range(150):
            if time.time() > deadline - 2.0:
                break
            rng = random.Random(trial * 7919 + 12345)
            order = list(range(1, N + 1))
            rng.shuffle(order)
            selected = _greedy_construct(order)
            _update_best(selected)
    
    # Final safety check
    if best is None:
        best = _greedy_construct(list(range(1, N + 1)))
    
    return best
# EVOLVE-BLOCK-END
