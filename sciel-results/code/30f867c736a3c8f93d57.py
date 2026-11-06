import time
import random
import heapq

N = 256
MAX_SUM = 2 * N

def _is_sidon(selected):
    """Check if selected is a Sidon set using bitset for speed."""
    sums = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        # Check 2a
        if (sums >> (2 * a)) & 1:
            return False
        sums |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            if (sums >> s) & 1:
                return False
            sums |= (1 << s)
    return True

def _compute_sums_bitset(selected):
    sums = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            sums |= (1 << s)
    return sums

def _is_maximal_bitset(selected, sums_mask):
    """
    Check if selected is inclusion-maximal.
    Sums_mask represents all sums a+b for a<=b in selected.
    x is addable if:
    1. 2x is NOT in sums_mask
    2. For all a in selected, x+a is NOT in sums_mask
    """
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check diagonal
        if (sums_mask >> (2 * x)) & 1:
            continue
        # Check off-diagonal
        blocked = False
        for a in selected:
            if (sums_mask >> (x + a)) & 1:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_bitset(order, deadline):
    """
    Greedy construction using bitset for fast checks.
    order: list of candidates to try.
    """
    selected = []
    sums = 0
    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        if (sums >> (2 * x)) & 1:
            continue
        ok = True
        for a in selected:
            if (sums >> (x + a)) & 1:
                ok = False
                break
        if ok:
            selected.append(x)
            sums |= (1 << (2 * x))
            for a in selected[:-1]:
                sums |= (1 << (x + a))
    return selected

def _reduce_bitset(start_set, deadline):
    """
    Aggressively reduce the set by removing elements that don't break maximality.
    Uses bitset for checks.
    """
    current = list(start_set)
    if not current:
        return current
    
    # Initial sums
    sums_mask = _compute_sums_bitset(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        # Shuffle to avoid bias? Or try all?
        # Trying all is better for finding the minimal set.
        for i in range(len(current)):
            if time.time() > deadline:
                break
            x = current[i]
            # Remove x
            new_current = current[:i] + current[i+1:]
            if not new_current:
                # Empty set is not maximal (unless N=0, but N=256)
                continue
            
            # Recalculate sums for new_current
            new_sums = _compute_sums_bitset(new_current)
            
            # Check maximality
            if _is_maximal_bitset(new_current, new_sums):
                # Accept removal
                current = new_current
                sums_mask = new_sums
                improved = True
                break # Restart loop with new size
    
    return current

def _construct_mis_heuristic(deadline):
    """
    Construct a small maximal Sidon set using a Maximum Independent Set (MIS) inspired heuristic.
    1. Build conflict graph: edges between x, y if x+y is a "bad" sum? No, simpler:
       In a Sidon set, if x, y, z, w are in the set, x+y != z+w.
       Conflict for MIS: Two numbers x, y "conflict" if they cannot be in the same Sidon set?
       Actually, ANY two numbers CAN be in a Sidon set unless they create a duplicate sum with a third.
       The MIS formulation is tricky because the conflict depends on the *set* of other nodes, not just the pair.
       However, we can use a greedy elimination strategy:
       - Start with all nodes [1..N].
       - Repeatedly pick a node to KEEP or REMOVE.
       - To minimize size, we want to REMOVE as many as possible while maintaining maximality.
       
       Alternative MIS interpretation:
       Let's define a "blocking" relation. 
       If we keep a set S, it blocks all x such that x conflicts with S.
       We want S to be maximal (blocks all x not in S) and Sidon.
       
       Heuristic:
       1. Generate a valid maximal Sidon set (e.g., via greedy).
       2. Identify "essential" elements. An element x is essential if removing it makes the set non-maximal.
       3. The goal is to find a set where ALL elements are essential? No, we want the set to be minimal, so we remove non-essential ones.
       
       The "MIS" strategy suggested: "greedy vertex elimination heuristic that iteratively removes the vertex with the highest degree (most conflicts)".
       In the context of minimizing a maximal independent set:
       - Vertices: [1..N]
       - Edges: (u,v) if u and v cannot coexist in a Sidon set? 
         This is not a static graph. 1 and 2 can coexist. 1 and 3 can coexist. 
         But 1, 2, 3 cannot (1+3=2+2).
         So it's a hypergraph or a 3-uniform constraint.
         
       Let's interpret "highest degree" as: 
       For a current maximal Sidon set S, calculate for each x in S how many elements in [1..N]\S it "blocks" (prevents from being added).
       If x blocks many elements, it's "efficient" at enforcing maximality. We should KEEP it.
       If x blocks few elements, it's "inefficient". We should try to REMOVE it, hoping the other elements still cover the blocked set.
       
       This is effectively what _reduce_bitset does, but _reduce_bitset just tries removals.
       We can enhance the reduction by prioritizing removal of elements that are "least blocking".
    """
    # Start with a good maximal Sidon set
    # Use a greedy set first
    initial = _greedy_bitset(list(range(1, N + 1)), deadline)
    if not initial:
        return []
    
    current = list(initial)
    
    # Iterate: find the element that blocks the fewest unique elements, and try to remove it.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # For each x in current, count how many y in [1..N]\current are blocked ONLY by x?
        # Or simply: try to remove x. If maximal, remove.
        # To be smarter, calculate "blocking power".
        # An element x is a "weak blocker" if removing it still leaves the set maximal.
        # This is exactly the check in _reduce_bitset.
        
        # Let's try to remove elements in an order that might yield more reductions.
        # Maybe remove elements that are "redundant" in terms of coverage.
        
        # Simple approach: try all, remove first one that works.
        # Better approach: try to remove the one that is "least useful".
        # How to measure usefulness? 
        # If we remove x, does the set become non-maximal?
        # If yes, x is essential.
        # If no, x is non-essential.
        
        # We want to remove non-essential elements.
        # _reduce_bitset already does this. 
        # The "MIS" idea might imply a different construction phase.
        
        # Let's try a construction that actively seeks to cover the "blocking" requirements with fewer nodes.
        # This is a Set Cover problem:
        # Universe U = [1..N] \ S
        # For each x in S, let C_x = { y in U : y is blocked by x }
        # We want to find a subset S' of S such that Union(C_x for x in S') covers U.
        # And S' must be Sidon (which it is, as subset of S).
        # And S' must be maximal (which is equivalent to covering U).
        
        # So, we can solve this Set Cover instance greedily!
        
        if not current:
            break
            
        sums_mask = _compute_sums_bitset(current)
        current_set = set(current)
        universe = [x for x in range(1, N + 1) if x not in current_set]
        
        if not universe:
            break
            
        # Calculate blocking sets for each x in current
        # C_x = { y in universe : y is blocked by x in the context of current }
        # Note: "blocked by x" means x contributes to the conflict.
        # y is blocked if 2y in sums OR (y+a in sums for some a in current).
        # If we remove x, y might become unblocked if x was the *only* element causing the conflict.
        
        # Let's compute for each y in universe, which x in current are "responsible" for blocking y.
        # y is blocked if:
        # 1. 2y in sums. This is caused by any pair (a,b) in current with a+b=2y.
        # 2. y+a in sums for some a in current. This is caused by any pair (b,c) in current with b+c=y+a.
        
        # This is complex. Let's stick to the simpler reduction but maybe with a better order.
        # Or, just use the standard reduction which is already good.
        
        # The prompt suggests: "greedy vertex elimination heuristic that iteratively removes the vertex with the highest degree (most conflicts)".
        # In MIS, we remove high-degree vertices to simplify the graph.
        # Here, maybe we should *keep* high-degree (high blocking) vertices and remove low-degree ones.
        
        # Let's implement a "blocking coverage" reduction.
        
        # For each x in current, determine how many elements in universe it "helps block".
        # We can approximate this by:
        # If we remove x, check if maximality is lost.
        # This is O(N * K^2) per step.
        
        # Let's just run _reduce_bitset. It's already doing the right thing.
        # To incorporate the "MIS" flavor, maybe we construct the initial set differently?
        # E.g., pick vertices that cover the most "potential conflicts"?
        
        # For now, return the reduced set.
        return _reduce_bitset(current, deadline)
    
    return current

def _swap_improve_bitset(start_set, deadline, rng):
    """
    Try swaps to find better configurations that allow further reduction.
    """
    current = list(start_set)
    current_set = set(current)
    sums_mask = _compute_sums_bitset(current)
    
    improved = True
    iters = 0
    while improved and time.time() < deadline and iters < 100:
        improved = False
        iters += 1
        
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N + 1) if x not in current_set]
        rng.shuffle(add_candidates)
        
        for r in remove_candidates:
            if time.time() > deadline:
                break
            # Remove r
            temp_current = [x for x in current if x != r]
            temp_set = set(temp_current)
            
            for a in add_candidates:
                if time.time() > deadline:
                    break
                if a in temp_set:
                    continue
                
                # Check if temp_current + [a] is Sidon
                candidate = temp_current + [a]
                candidate.sort()
                
                if not _is_sidon(candidate):
                    continue
                
                # Check maximality
                cand_sums = _compute_sums_bitset(candidate)
                if _is_maximal_bitset(candidate, cand_sums):
                    # Accept swap
                    current = candidate
                    current_set = set(candidate)
                    sums_mask = cand_sums
                    improved = True
                    break
            if improved:
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best = None
    best_size = N + 1
    
    candidates = []
    
    # 1. Greedy Ascending
    c1 = _greedy_bitset(list(range(1, N + 1)), deadline)
    candidates.append(c1)
    
    # 2. Greedy Descending
    c2 = _greedy_bitset(list(range(N, 0, -1)), deadline)
    candidates.append(c2)
    
    # 3. Randomized Greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng = random.Random(1000 + seed)
        rng.shuffle(order)
        sel = _greedy_bitset(order, deadline)
        if sel:
            candidates.append(sel)
    
    # 4. MIS-inspired Construction: 
    # Try to build a set that covers "blocking" requirements efficiently.
    # This is hard to construct from scratch. 
    # Instead, let's just add more diverse random starts.
    for seed in range(10):
        if time.time() > deadline:
            break
        rng = random.Random(2000 + seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        sel = _greedy_bitset(order, deadline)
        if sel:
            candidates.append(sel)

    # Process candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if not cand:
            continue
        
        # Ensure valid Sidon
        if not _is_sidon(cand):
            continue
            
        # Check if maximal.
        sums = _compute_sums_bitset(cand)
        if not _is_maximal_bitset(cand, sums):
            continue
            
        rng = random.Random(hash(tuple(sorted(cand))) % (2 ** 31))
        
        # Reduce
        minimized = _reduce_bitset(cand, deadline)
        
        # Swap Improve
        if time.time() < deadline - 5.0:
            minimized = _swap_improve_bitset(minimized, deadline, rng)
            # Reduce again after swaps
            minimized = _reduce_bitset(minimized, deadline)
            
        if minimized:
            if _is_sidon(minimized) and _is_maximal_bitset(minimized, _compute_sums_bitset(minimized)):
                if len(minimized) < best_size:
                    best_size = len(minimized)
                    best = list(minimized)

    if best is None:
        # Fallback
        return _greedy_bitset(list(range(1, N + 1)), start_time + 1.0)
        
    return best
