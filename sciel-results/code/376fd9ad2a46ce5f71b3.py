"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    import random

    deadline = time.time() + 80  # safety margin before 90s kill
    N = 512

    # Build compatibility using bitmasks for fast intersection checks
    # compat_mask[v] is a bitmask of all u != v such that lcm(v,u) <= N
    compat_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= N:
                compat_mask[a] |= (1 << b)
                compat_mask[b] |= (1 << a)

    # Precompute neighbor lists for each vertex
    neighbors = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        mask = compat_mask[v]
        nbrs = []
        for u in range(1, N + 1):
            if u != v and (mask >> u) & 1:
                nbrs.append(u)
        neighbors[v] = nbrs
        # Also store as bitmask for fast intersection
    # Store neighbor bitmasks for fast operations
    nbr_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        m = 0
        for u in neighbors[v]:
            m |= (1 << u)
        nbr_mask[v] = m

    degrees = [len(neighbors[v]) for v in range(N + 1)]

    # Greedy heuristic to get initial solution
    def greedy_clique(order):
        selected = []
        sel_mask = 0
        for v in order:
            if (nbr_mask[v] & sel_mask) == sel_mask:
                selected.append(v)
                sel_mask |= (1 << v)
        return selected

    # Sort by degree descending
    vertices_sorted = sorted(range(1, N + 1), key=lambda v: -degrees[v])
    best_clique = greedy_clique(vertices_sorted)
    best_size = len(best_clique)

    # Local search: add-only
    def local_search_add(clique):
        in_set = set(clique)
        sel_mask = 0
        for v in in_set:
            sel_mask |= (1 << v)
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for v in range(1, N + 1):
                if v not in in_set:
                    if (nbr_mask[v] & sel_mask) == sel_mask:
                        in_set.add(v)
                        sel_mask |= (1 << v)
                        improved = True
        return list(in_set)

    best_clique = local_search_add(best_clique)
    best_size = len(best_clique)

    # Branch and bound max clique (Tomita-style with coloring)
    # We work on the full graph. Use a recursive approach with pruning.
    
    # For efficiency, we'll use a greedy coloring bound
    def greedy_color_bound(vertices, nbr_mask_arr):
        """Color vertices greedily to get an upper bound on clique size in this subgraph."""
        if not vertices:
            return 0, []
        # Simple greedy coloring
        color = {}
        colors_used = []
        for v in vertices:
            # Find smallest color not used by neighbors of v that are already colored
            used_colors = set()
            for u in colors_used:
                if (nbr_mask_arr[v] >> u) & 1:
                    used_colors.add(color.get(u, -1))
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            colors_used.append(v)
        max_color = max(color.values()) if color else 0
        return max_color + 1, list(color.values())

    # Branch and bound
    # We represent the candidate set as a list
    # Use the ordering: sort candidates by degree within the candidate set
    
    best_found = best_size
    best_clique_found = list(best_clique)
    time_limit = deadline - 10

    def branch_and_bound(candidates, current_clique):
        nonlocal best_found, best_clique_found
        if time.time() > time_limit:
            return
        if not candidates:
            if len(current_clique) > best_found:
                best_found = len(current_clique)
                best_clique_found = list(current_clique)
            return
        if len(current_clique) + len(candidates) <= best_found:
            return
        
        # Greedy coloring for upper bound
        # Order candidates by degree in the candidate subgraph
        cand_set = set(candidates)
        cand_degrees = {}
        for v in candidates:
            deg = 0
            vmask = nbr_mask[v]
            for u in candidates:
                if u != v and (vmask >> u) & 1:
                    deg += 1
            cand_degrees[v] = deg
        
        # Sort by degree descending for better pruning
        candidates_sorted = sorted(candidates, key=lambda v: -cand_degrees[v])
        
        # Greedy coloring: assign colors greedily
        color_assign = {}
        max_color = 0
        for v in candidates_sorted:
            # Colors used by already-colored neighbors
            used = set()
            for u in candidates_sorted:
                if u not in color_assign:
                    break
                if (nbr_mask[v] >> u) & 1:
                    used.add(color_assign[u])
            c = 0
            while c in used:
                c += 1
            color_assign[v] = c
            if c + 1 > max_color:
                max_color = c + 1
        
        # Upper bound
        if len(current_clique) + max_color <= best_found:
            return
        
        # Process vertices in reverse color order (highest color first)
        for i in range(len(candidates_sorted) - 1, -1, -1):
            if time.time() > time_limit:
                return
            v = candidates_sorted[i]
            # Upper bound for this branch: current + (max_color for remaining)
            # The remaining candidates are candidates_sorted[0:i] (those with index < i)
            # Their max color is at most max_color, but we can tighten
            # Simple bound: len(current_clique) + (i + 1) <= best_found -> prune
            if len(current_clique) + i + 1 <= best_found:
                break
            
            # Check if v is compatible with all in current_clique
            compatible = True
            for cv in current_clique:
                if not ((nbr_mask[v] >> cv) & 1):
                    compatible = False
                    break
            if not compatible:
                continue
            
            # Recurse
            new_candidates = [u for u in candidates_sorted[:i] if (nbr_mask[v] >> u) & 1]
            branch_and_bound(new_candidates, current_clique + [v])
            
            # Remove v from candidates for subsequent iterations
            # (already handled by the loop structure)

    # Start B&B from the full vertex set, but use the greedy solution as lower bound
    # To make it tractable, start with vertices sorted by degree
    # Actually, let's try B&B on the full graph but it might be too slow
    # Instead, do multiple local search restarts with swap operations
    
    # Swap-based local search
    def local_search_swap(clique, max_iters=1000):
        in_set = set(clique)
        sel_mask = 0
        for v in in_set:
            sel_mask |= (1 << v)
        best_local = list(in_set)
        best_local_size = len(in_set)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding a vertex
            for v in range(1, N + 1):
                if v not in in_set:
                    if (nbr_mask[v] & sel_mask) == sel_mask:
                        in_set.add(v)
                        sel_mask |= (1 << v)
                        best_local = list(in_set)
                        best_local_size = len(in_set)
                        improved = True
                        break
            if improved:
                continue
            # Try swap: remove one, add another
            for v_in in list(in_set):
                for v_out in range(1, N + 1):
                    if v_out in in_set:
                        continue
                    # Check if v_out is compatible with (in_set - {v_in})
                    test_mask = sel_mask ^ (1 << v_in)
                    if (nbr_mask[v_out] & test_mask) == test_mask:
                        in_set.remove(v_in)
                        in_set.add(v_out)
                        sel_mask = test_mask | (1 << v_out)
                        if len(in_set) > best_local_size:
                            best_local = list(in_set)
                            best_local_size = len(in_set)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_local

    # Run multiple restarts
    rng = random.Random(42)
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        # Create a perturbed ordering
        order = list(range(1, N + 1))
        # Bias by degree with random noise
        order.sort(key=lambda v: -degrees[v] + rng.random() * 20)
        clique = greedy_clique(order)
        clique = local_search_add(clique)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
            # Try swap local search on the best
            improved_clique = local_search_swap(clique, max_iters=500)
            if len(improved_clique) > best_size:
                best_size = len(improved_clique)
                best_clique = improved_clique
    
    # Final polish: try to add any remaining compatible vertices
    final_set = set(best_clique)
    sel_mask = 0
    for v in final_set:
        sel_mask |= (1 << v)
    for v in range(1, N + 1):
        if v not in final_set:
            if (nbr_mask[v] & sel_mask) == sel_mask:
                final_set.add(v)
                sel_mask |= (1 << v)
    best_clique = list(final_set)
    best_size = len(best_clique)

    return best_clique
# EVOLVE-BLOCK-END
