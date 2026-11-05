import time
import itertools
import random

def _can_add(x, selected):
    """Check if x can be added to selected without violating the constraint."""
    n = len(selected)
    if n >= 3:
        # x must not divide sum of any 3 distinct elements in selected
        # Optimize: Check if any 3 distinct elements sum to a multiple of x.
        # s_i + s_j + s_k == 0 (mod x)
        # Iterate pairs (i, j), check if there exists k != i, j such that s_k == - (s_i + s_j) (mod x)
        
        # Precompute residues and counts
        residues = [s % x for s in selected]
        count_res = {}
        for r in residues:
            count_res[r] = count_res.get(r, 0) + 1
            
        for i in range(n):
            for j in range(i + 1, n):
                target_res = (-selected[i] - selected[j]) % x
                if target_res in count_res:
                    cnt = count_res[target_res]
                    # Check if we can pick a k distinct from i and j
                    # If s_i % x == target_res, then i is a candidate for k, but k must be distinct from i.
                    # If s_j % x == target_res, then j is a candidate for k, but k must be distinct from j.
                    
                    # Let's count how many of i, j have residue target_res
                    bad_indices = 0
                    if residues[i] == target_res:
                        bad_indices += 1
                    if residues[j] == target_res:
                        bad_indices += 1
                        
                    # We need at least one k from the 'cnt' elements with that residue,
                    # excluding i and j if they are among them.
                    available = cnt - bad_indices
                    if available > 0:
                        return False
    
    # For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    # x + u + v == 0 (mod a) => u + v == -x (mod a)
    for a_idx, a in enumerate(selected):
        if a == 1:
            # 1 divides everything. So if we can find ANY u, v in selected \ {1} such that 1 + u + v is a multiple of 1.
            # Since 1 divides any integer, this is ALWAYS true if n >= 2 (we need distinct u,v).
            # Wait, if a=1, then 1 | (x+u+v) is always true.
            # So x cannot be added if there are at least 2 other elements.
            if n >= 2:
                return False
            # If n < 2, we can't pick u,v distinct from a. So it's safe?
            # If n=0, no u,v. If n=1, no u,v distinct from a.
            continue
            
        # Check if there exist u, v in selected \ {a} such that u + v == -x (mod a)
        target_sum = (-x) % a
        # We can use a similar residue trick, but modulo a.
        # However, a changes for each iteration.
        # Since n is small, O(n^2) is fine here.
        others = [selected[i] for i in range(n) if i != a_idx]
        m = len(others)
        if m < 2:
            continue
            
        # Use set for faster lookup? Or just double loop.
        # Double loop is O(m^2). m <= 47. 47^2 = 2209.
        # We can optimize by storing residues of others modulo a.
        # But we need distinct u, v.
        # Let's just do the double loop for simplicity, it's fast enough.
        for i in range(m):
            for j in range(i + 1, m):
                if (others[i] + others[j]) % a == target_sum:
                    return False
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping or adding elements."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(selected)
        
        # 1. Try adding any element not in selected
        for x in range(192, 0, -1):
            if time.time() > deadline:
                return selected
            if x not in in_set and _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                break
        
        # 2. If no addition, try 1-swap
        if not improved:
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected
                candidate = selected[i]
                selected_without = selected[:i] + selected[i+1:]
                for x in range(192, 0, -1):
                    if x in in_set:
                        continue
                    if _can_add(x, selected_without):
                        # Check if this swap is beneficial.
                        # Ideally, we want to increase size, but here size is constant.
                        # We accept the swap to potentially enable future additions.
                        # To avoid cycles, we could require some metric to improve,
                        # but simple randomization or "first found" works in practice if we randomize order.
                        selected = selected_without + [x]
                        in_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
            
        # 3. If no 1-swap, try 2-swap (expensive, use sparingly)
        if not improved and len(selected) >= 2:
            # Try to remove two elements and add two new ones
            # This is O(n^2 * 192^2 * Cost(can_add)) which is too high.
            # We can restrict to a few random pairs.
            if random.random() < 0.1: # Only do this occasionally
                for _ in range(10):
                    if time.time() > deadline:
                        return selected
                    i, j = random.sample(range(len(selected)), 2)
                    to_remove = [selected[i], selected[j]]
                    selected_without = [s for s in selected if s not in to_remove]
                    
                    # Try to add two elements
                    # We can do this sequentially: add one, then add another
                    # Or try pairs.
                    # Let's try adding one element that fits in selected_without, 
                    # then another.
                    
                    # Actually, a simpler 2-swap is: remove u, add v, then remove w, add z.
                    # This is just two 1-swaps. If 1-swaps failed, this might fail too unless the first swap enables the second.
                    # So let's just stick to 1-swaps and maybe random restarts.
                    pass
                    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best = []
    
    # Strategy 1: Greedy descending
    t1 = time.time()
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 5:
        result = _greedy(range(1, 193), deadline)
        if len(result) > len(best):
            best = result
            
    # Strategy 3: Random shuffles
    random.seed(42)
    # Since _can_add is faster, we can do more restarts
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Local search on best
    if time.time() < deadline - 10:
        best = _local_search(best, deadline)
        
    # Run local search on a few other good candidates too
    # Keep a list of top candidates from greedy
    # For simplicity, just re-run local search on best
    
    return best
