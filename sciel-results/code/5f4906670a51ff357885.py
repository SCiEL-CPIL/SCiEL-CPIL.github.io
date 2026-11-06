import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Precompute differences of fixed elements
    delta_f = set()
    for i in range(len(fixed_elements)):
        for j in range(i+1, len(fixed_elements)):
            delta_f.add(abs(fixed_elements[i] - fixed_elements[j]))
            
    # Filter candidates
    candidates = []
    for x in range(1, 285):
        if x in fixed_set:
            continue
        # Check if x is individually valid
        valid = True
        diffs_with_f = set()
        for f in fixed_elements:
            d = abs(x - f)
            if d in delta_f:
                valid = False
                break
            if d in diffs_with_f:
                # This implies |x-f1| == |x-f2| -> f1+f2 == 2x
                # We checked this earlier, but just in case
                valid = False
                break
            diffs_with_f.add(d)
        if valid:
            candidates.append((x, diffs_with_f))
            
    # Build static conflict graph
    # u, v are statically incompatible if:
    # 1. |u-v| in delta_f
    # 2. |u-v| in diffs_u (implies v = 2u - f)
    # 3. |u-v| in diffs_v (implies u = 2v - f)
    
    # Map value to index in candidates list for faster lookup?
    # Or just use sets.
    
    # To speed up backtracking, we can represent the state as:
    # - current_set (list of values)
    # - current_deltas (set of differences)
    # - available_indices (set of indices into candidates list)
    
    # Precompute static conflicts as a list of sets of indices
    n_cands = len(candidates)
    static_conflicts = [set() for _ in range(n_cands)]
    
    for i in range(n_cands):
        u, diffs_u = candidates[i]
        for j in range(i+1, n_cands):
            v, diffs_v = candidates[j]
            d_uv = abs(u - v)
            if d_uv in delta_f:
                static_conflicts[i].add(j)
                static_conflicts[j].add(i)
            elif d_uv in diffs_u:
                static_conflicts[i].add(j)
                static_conflicts[j].add(i)
            elif d_uv in diffs_v:
                static_conflicts[i].add(j)
                static_conflicts[j].add(i)
                
    best_size = len(fixed_elements)
    best_set = list(fixed_elements)
    
    # Greedy initialization
    def greedy_init():
        a = list(fixed_elements)
        deltas = set(delta_f)
        a_set = set(a)
        available = set(range(n_cands))
        
        # Sort candidates by value? Or random?
        # Let's try a few random orders
        for _ in range(10):
            a = list(fixed_elements)
            deltas = set(delta_f)
            a_set = set(a)
            available = set(range(n_cands))
            
            order = list(range(n_cands))
            random.shuffle(order)
            
            for idx in order:
                if idx not in available:
                    continue
                x, diffs_x = candidates[idx]
                # Check dynamic compatibility
                ok = True
                for s in a:
                    d = abs(x - s)
                    if d in deltas:
                        ok = False
                        break
                if ok:
                    a.append(x)
                    a_set.add(x)
                    # Update deltas
                    for s in a[:-1]: # s is the new element x, so we check against old a
                        deltas.add(abs(x - s))
                    deltas |= diffs_x
                    # Remove static conflicts from available
                    for conf in static_conflicts[idx]:
                        available.discard(conf)
                    available.discard(idx)
                    
        return a

    best_set = greedy_init()
    best_size = len(best_set)
    
    # Backtracking search
    # We will try to improve upon best_size
    # To avoid deep recursion, we limit depth or use iterative deepening?
    # Given max size 20, depth is small.
    
    def search(current_set, current_deltas, available_indices, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_size + len(available_indices) <= best_size:
            return
            
        # Try to find a better solution
        # Heuristic: pick candidate with fewest static neighbors in available?
        # Or just pick the first one?
        
        # Let's try all candidates in available, ordered by some heuristic
        # To speed up, we can just iterate
        
        # To avoid duplicate work, we can fix an ordering?
        # But we want to find *any* large set.
        
        # Let's pick a candidate to include
        # If we include x, we must exclude its static neighbors.
        
        # To make it faster, we can try to add candidates that are "safe"
        
        # Let's iterate through available candidates
        # To prevent exponential explosion, we can limit the branching factor?
        # Or just rely on pruning.
        
        # Sort available by index?
        avail_list = list(available_indices)
        
        # Heuristic: try candidates that have fewer conflicts first?
        # This might lead to a more "open" set.
        avail_list.sort(key=lambda idx: len(static_conflicts[idx] & available_indices))
        
        for idx in avail_list:
            if time.time() > deadline:
                return
                
            x, diffs_x = candidates[idx]
            
            # Check dynamic compatibility
            ok = True
            new_deltas = set(current_deltas)
            for s in current_set:
                d = abs(x - s)
                if d in new_deltas:
                    ok = False
                    break
                new_deltas.add(d)
            if not ok:
                continue
                
            new_deltas |= diffs_x
            
            new_set = current_set + [x]
            new_size = current_size + 1
            
            if new_size > best_size:
                best_size = new_size
                best_set = list(new_set)
                
            # New available: remove idx and its static neighbors
            new_available = set(available_indices)
            new_available.discard(idx)
            for conf in static_conflicts[idx]:
                new_available.discard(conf)
                
            # Recurse
            search(new_set, new_deltas, new_available, new_size)
            
            # Also, we should try NOT including idx?
            # In a standard MIS, we branch on include/exclude.
            # Here, if we don't include idx, we just remove it from available.
            # But we are iterating, so the next iteration effectively "excludes" idx
            # by moving to the next candidate.
            # However, the "exclude" branch should still allow other candidates
            # that might conflict with idx? No, if we don't pick idx,
            # its neighbors are still available.
            # So the loop structure handles this:
            # If we pick idx, we recurse with reduced available.
            # If we don't pick idx, we just continue the loop,
            # and the next candidate is considered with the SAME available set
            # (minus idx? No, the loop iterates over the original avail_list).
            # Wait, if I don't pick idx, I should still be able to pick its neighbors.
            # My loop iterates over `avail_list`.
            # If I pick idx, I recurse.
            # If I don't pick idx, I just move to the next item in `avail_list`.
            # This implicitly excludes idx from the current path,
            # but keeps its neighbors available for subsequent choices in this level?
            # No, if I pick a neighbor later in the loop, it's fine.
            # But what if I should have picked idx to enable something?
            # The recursion handles the "pick idx" case.
            # The loop continues for the "don't pick idx" case.
            # This is correct for finding *some* independent set.
            
    # Start search from the greedy result
    # To make it more effective, we can start from scratch or from the greedy set.
    # Starting from the greedy set might be faster if it's good.
    # But we need to construct the `available_indices` correctly.
    
    # Let's restart the search from the initial state (only F) to ensure we don't miss better structures
    # but use the best_size found by greedy for pruning.
    
    initial_available = set(range(n_cands))
    initial_set = list(fixed_elements)
    initial_deltas = set(delta_f)
    
    # To speed up, we can shuffle the initial available set?
    # The search function sorts available by conflict count, which is deterministic.
    # To introduce randomness, we can shuffle the candidates list before building conflicts?
    # Or just run the search multiple times with different random seeds for the greedy init?
    
    # Let's run the backtracking search once, but with a randomized order of candidates in the loop?
    # The sort by conflict count is a good heuristic.
    
    search(initial_set, initial_deltas, initial_available, len(initial_set))
    
    return best_set
