import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # --- C6 Detection using Bitsets ---
    def _has_c6_bitset(adj, n):
        """
        Check if the graph defined by adjacency masks 'adj' contains a 6-cycle.
        adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists.
        n is the number of vertices.
        """
        for v0 in range(n):
            # v1 in N(v0)
            mask_v1 = adj[v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                
                # v2 in N(v1) \ {v0}
                mask_v2 = adj[v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    
                    # v3 in N(v2) \ {v0, v1}
                    mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        
                        # v4 in N(v3) \ {v0, v1, v2}
                        mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            
                            # Check if there exists v5 such that:
                            # v5 in N(v4) AND v5 in N(v0)
                            # v5 not in {v0, v1, v2, v3, v4}
                            
                            # Candidates for v5
                            mask_v5 = adj[v4] & adj[v0]
                            # Exclude used vertices
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            
                            if mask_v5:
                                return True
        return False

    def _is_valid(colors, n):
        """Check if the coloring is valid (no mono C6)."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
            
        for c in range(4):
            if _has_c6_bitset(adj[c], n):
                return False
        return True

    def _backtrack_csp(n, time_limit=100.0):
        """
        Backtracking search with Forward Checking and MRV heuristic.
        Returns colors list if found, else None.
        """
        deadline = time.time() + time_limit
        E = n * (n - 1) // 2
        # Edges in lex order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # State:
        # adj[c][v] is bitmask of neighbors
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * E
        
        # To speed up backtracking, we can pre-calculate which edges are "critical"
        # or simply use a dynamic variable ordering.
        # For n=17, E=136. Dynamic ordering is expensive.
        # Let's try a static ordering first: Edges involving vertex 0 first? 
        # Or just lex order? Lex order is often decent for graphs.
        # However, MRV (Most Constrained Variable) is powerful.
        # Implementing full MRV with dynamic re-ordering of 136 edges is complex.
        
        # Alternative: Use a fixed "good" ordering.
        # Heuristic: Color edges in a way that spreads colors?
        # Let's stick to a slightly randomized lex order or just lex.
        # Actually, let's try a simple optimization:
        # If we add an edge (u,v) with color c, it creates a C6 if there is a path of length 5 between u and v in G_c.
        # We can pre-compute "potential" conflicts? No, it's dynamic.
        
        # Let's implement a simple backtracker with Forward Checking (FC).
        # FC: When assigning color c to edge (u,v), check if any other unassigned edge (x,y)
        # is forced to be invalid? 
        # Actually, the constraint is global (no C6). It's not a binary constraint on edges.
        # So standard FC doesn't apply directly to "edge-edge" constraints.
        # The constraint is on the *set* of edges of color c.
        
        # So, we must check: Does adding (u,v) to color c create a C6 in the *current* graph of color c?
        # This is exactly what the current seed does in _backtrack_search (which was a placeholder).
        # The key to making this fast is the ORDER of edges and early pruning.
        
        # Let's use a heuristic order:
        # 1. Edges incident to vertex 0
        # 2. Edges incident to vertex 1 (not 0)
        # ...
        # This tends to build up the graph structure quickly, allowing early detection of cycles.
        
        ordered_edges = sorted(edges, key=lambda e: (e[0], e[1]))
        
        # We can also shuffle the color order for each edge to break symmetry?
        # Symmetry breaking: Fix color of edge (0,1) to 0?
        # Yes, due to color symmetry, we can fix the color of the first edge.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(ordered_edges):
                return True
                
            u, v = ordered_edges[idx]
            
            # Try colors 0-3
            # Optimization: If idx == 0, only try color 0 (symmetry breaking)
            if idx == 0:
                color_range = [0]
            else:
                color_range = [0, 1, 2, 3]
                
            for c in color_range:
                # Add edge (u,v) to color c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                # Check if this creates a C6 in color c
                if _has_c6_bitset(adj[c], n):
                    # Undo
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    continue
                else:
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    # Undo
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    
            return False
        
        # To improve success rate, we can try multiple random restarts with different edge orderings
        # or color permutations if the first attempt fails quickly.
        # But for n=17, the search space is huge.
        # Let's try a "greedy" initialization first to get close, then backtrack?
        # No, backtracking from scratch is cleaner if optimized.
        
        # Let's add a small randomization to the edge order to escape bad local structures
        # in the search tree? No, backtracking is exhaustive.
        # The only randomization is in the *choice* of which n to try or restarts.
        
        # Let's try the search.
        if backtrack(0):
            # Reconstruct colors in lex order
            res_colors = [0] * E
            # Map back from ordered_edges to original indices
            # Since we sorted, we need to know where each ordered edge came from.
            # edges list is already in lex order. 
            # If we didn't change the order, idx in ordered_edges corresponds to idx in edges?
            # Yes, if we just use the same list.
            # In my code above, ordered_edges = sorted(edges...). 
            # edges = [(0,1), (0,2)...]
            # So ordered_edges[k] is the k-th edge in lex order.
            # So res_colors[idx] = colors[idx] is correct if we used lex order.
            # Wait, I used `ordered_edges = sorted(edges, key=lambda e: (e[0], e[1]))`.
            # `edges` was generated as `[(i, j) for i in range(n) for j in range(i + 1, n)]`.
            # This is already sorted by i, then j.
            # So `ordered_edges` is identical to `edges`.
            # So `colors` array is already in the correct lex order!
            return colors
        
        return None

    # Strategy:
    # 1. Try to find a solution for n=17 using the CSP backtracker.
    # 2. If successful, return it.
    # 3. If it fails (or times out), try n=16, etc.
    # 4. If n=17 is too slow for pure backtracking, we might need a hybrid.
    #    But let's try pure backtracking with symmetry breaking first.
    
    best_n = 0
    best_colors = None
    
    # We prioritize n=17.
    # If we find n=17, we stop.
    # If we don't find it in ~90s, we drop to n=16.
    
    # Time allocation:
    # n=17: 90s
    # n=16: 15s
    # n=15: 5s
    # ...
    
    time_allocs = {17: 90, 16: 15, 15: 5, 14: 2, 13: 1, 12: 0.5, 11: 0.5, 10: 0.5, 9: 0.5, 8: 0.5, 7: 0.5, 6: 0.5, 5: 0.5, 4: 0.5, 3: 0.5, 2: 0.5, 1: 0.5}
    
    for n in range(17, 0, -1):
        if time.time() > deadline - 2:
            break
            
        t_alloc = time_allocs.get(n, 1.0)
        
        # For n=17, backtracking might be too slow if the first path is bad.
        # We can try a few random restarts with different random seeds for color ordering?
        # Actually, the edge order is fixed. The only randomness is if we shuffle colors?
        # No, we try 0,1,2,3 in order.
        # To improve, we can shuffle the color order for each edge?
        # Let's modify the inner loop to use a shuffled color list?
        # That would make it a randomized backtracker.
        
        # Let's implement a randomized backtracker for better exploration.
        
        def _randomized_backtrack(n, time_limit):
            deadline_local = time.time() + time_limit
            E = n * (n - 1) // 2
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            # Try a few different edge orderings?
            # Ordering 1: Lex
            # Ordering 2: Random
            # Let's stick to Lex for determinism in the "structure" but randomize color choice.
            
            adj = [[0] * n for _ in range(4)]
            colors = [-1] * E
            
            def backtrack(idx):
                if time.time() > deadline_local:
                    return False
                
                if idx == len(edges):
                    return True
                    
                u, v = edges[idx]
                
                # Randomize color order to escape bad local minima in the search tree
                # (Though backtracking is exhaustive, the *order* affects which solution is found first)
                color_order = [0, 1, 2, 3]
                if idx == 0:
                    color_order = [0] # Symmetry breaking
                else:
                    random.shuffle(color_order)
                    
                for c in color_order:
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if _has_c6_bitset(adj[c], n):
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                        continue
                    else:
                        colors[idx] = c
                        if backtrack(idx + 1):
                            return True
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                        
                return False
            
            # Try multiple restarts with different random seeds for color shuffling
            for attempt in range(5):
                if time.time() > deadline_local:
                    break
                random.seed(attempt * 12345 + n)
                if backtrack(0):
                    return colors
            return None

        sol = _randomized_backtrack(n, t_alloc)
        if sol is not None:
            # Verify
            if _is_valid(sol, n):
                best_n = n
                best_colors = sol
                break # Found the max n we tried
        
    # Fallback
    if best_n == 0:
        n = 4
        best_colors = [0, 1, 2, 3, 0, 1] 
        best_n = n

    return {'n': best_n, 'colors': best_colors}

# Ensure the function is named construct
# The code above defines construct.

if __name__ == "__main__":
    res = construct()
    print(res)
