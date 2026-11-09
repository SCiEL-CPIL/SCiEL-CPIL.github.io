"""Improved program: bitmask-based exact MIS with pruning + greedy initialization for reciprocal-Schur-free subset of [1,180]."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    N = 180
    # Precompute all forbidden triples (a,b,c) with distinct elements where 1/a = 1/b + 1/c
    forbidden = set()
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    forbidden.add(triple)
    
    # Identify the "core" vertices involved in any forbidden triple
    core_vertices = set()
    for t in forbidden:
        core_vertices.update(t)
    
    # Non-core vertices are always safe to include
    non_core = set(range(1, N+1)) - core_vertices
    
    # --- Part 1: Greedy initialization ---
    S_greedy = set(non_core)
    active_triples = list(forbidden)
    
    while True:
        if time.time() - start_time > DEADLINE - 40:
            break
        found = None
        for t in active_triples:
            if all(x in S_greedy for x in t):
                found = t
                break
        if found is None:
            break
        
        best_elem = None
        best_count = -1
        for elem in found:
            count = 0
            for t2 in active_triples:
                if elem in t2 and all(x in S_greedy for x in t2):
                    count += 1
            if count > best_count:
                best_count = count
                best_elem = elem
        S_greedy.discard(best_elem)
    
    # Try to add back removed elements
    removed_greedy = set(core_vertices) - S_greedy
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 40:
        improved = False
        for elem in list(removed_greedy):
            if time.time() - start_time > DEADLINE - 40:
                break
            S_test = S_greedy | {elem}
            valid = True
            for t in forbidden:
                if elem in t and all(x in S_test for x in t):
                    valid = False
                    break
            if valid:
                S_greedy = S_test
                removed_greedy.discard(elem)
                improved = True
    
    # --- Part 2: Exact MIS using bitmask-based branch and bound ---
    # Map core vertices to indices
    core_list = sorted(core_vertices)
    n_core = len(core_list)
    vert_to_idx = {v: i for i, v in enumerate(core_list)}
    idx_to_vert = {i: v for i, v in enumerate(core_list)}
    
    # Build conflict graph as bitmasks
    # conflict_mask[i] = bitmask of vertices that conflict with vertex i
    conflict_mask = [0] * n_core
    for t in forbidden:
        verts = list(t)
        idxs = [vert_to_idx[v] for v in verts]
        for i in range(3):
            for j in range(3):
                if i != j:
                    conflict_mask[idxs[i]] |= (1 << idxs[j])
    
    # Greedy upper bound: color the graph, use max color class size
    def greedy_color_bound(available_mask):
        """Return an upper bound on MIS size using greedy coloring."""
        if available_mask == 0:
            return 0
        remaining = available_mask
        colors = 0
        while remaining:
            colors += 1
            # Greedily color: pick uncolored vertex, assign color, remove it and its neighbors from current color class
            color_class = 0
            while remaining:
                # Pick vertex with lowest index
                v = (remaining & -remaining).bit_length() - 1
                color_class |= (1 << v)
                remaining &= ~(1 << v)
                # Remove neighbors from remaining for this color
                remaining &= ~conflict_mask[v]
                remaining &= ~color_class
        return colors
    
    # Exact MIS solver with branch and bound
    best_mis_size = len(S_greedy & core_vertices)
    best_mis_mask = 0
    for v in S_greedy:
        if v in vert_to_idx:
            best_mis_mask |= (1 << vert_to_idx[v])
    
    # Count bits helper
    def popcount(x):
        return bin(x).count('1')
    
    def solve_mis_bb(available_mask, current_mask, current_size):
        """Branch and bound for maximum independent set."""
        nonlocal best_mis_size, best_mis_mask
        
        if available_mask == 0:
            if current_size > best_mis_size:
                best_mis_size = current_size
                best_mis_mask = current_mask
            return
        
        # Upper bound: current_size + popcount(available_mask)
        if current_size + popcount(available_mask) <= best_mis_size:
            return
        
        # Upper bound using greedy coloring
        if current_size + greedy_color_bound(available_mask) <= best_mis_size:
            return
        
        # Pick a vertex to branch on: choose one with highest degree in available subgraph
        # For efficiency, pick the first available vertex
        v = (available_mask & -available_mask).bit_length() - 1
        
        # Branch 1: Include v
        # Remove v and its neighbors from available
        new_available = available_mask & ~(1 << v) & ~conflict_mask[v]
        solve_mis_bb(new_available, current_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        solve_mis_bb(available_mask & ~(1 << v), current_mask, current_size)
    
    if n_core <= 60:  # Only attempt exact solve if core is small enough
        all_core_mask = (1 << n_core) - 1
        solve_mis_bb(all_core_mask, 0, 0)
        
        # Convert best_mis_mask back to vertex set
        S_mis = set(non_core)
        for i in range(n_core):
            if best_mis_mask & (1 << i):
                S_mis.add(idx_to_vert[i])
    else:
        S_mis = S_greedy
    
    # Choose the larger set
    if len(S_mis) > len(S_greedy):
        S_final = S_mis
    else:
        S_final = S_greedy

    # Final validation
    result_set = set(S_final)
    for t in forbidden:
        if all(x in result_set for x in t):
            for x in t:
                if x in result_set:
                    result_set.discard(x)
                    break
            break
    
    result = sorted(result_set)
    if not result:
        result = [91]
        
    return result
