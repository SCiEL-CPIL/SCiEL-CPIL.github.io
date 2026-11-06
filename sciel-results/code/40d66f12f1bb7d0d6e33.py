import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but we want to be safe)
    
    # Fallback solution: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Strategy: Greedy Constructive with Backtracking
    # We build the graph vertex by vertex.
    # State:
    #   colors: list of length n*(n-1)/2, -1 if unassigned
    #   adj_mask[c][u]: bitmask of neighbors of u in color c
    
    # To handle recursion depth and speed, we use an iterative DFS or a recursive one with sys.setrecursionlimit.
    # Given n is small (max ~20), recursion is fine.
    
    def try_solve(n, time_limit):
        """
        Attempts to find a valid 5-coloring of K_n using backtracking.
        Returns colors list or None.
        """
        if time.time() > time_limit:
            return None
            
        num_edges = n * (n - 1) // 2
        # Precompute edge list in lexicographic order (i, j) with i < j
        # But for vertex-by-vertex construction, we process edges (i, j) where j is the new vertex.
        # Order of edges: 
        # Vertex 0: (0,1), (0,2), ..., (0,n-1)
        # Vertex 1: (1,2), (1,3), ..., (1,n-1)
        # ...
        # Vertex n-2: (n-2, n-1)
        
        # This order is NOT the same as the required output order (lexicographic i<j over all pairs).
        # We will maintain a temporary assignment array `tmp_colors` indexed by the construction order,
        # or better, map construction steps to the final indices.
        
        # Let's define the construction order explicitly.
        # Step k corresponds to adding edge (u, v) where v is the current vertex being connected.
        # Actually, simpler: iterate v from 1 to n-1. For each v, iterate u from 0 to v-1.
        # The edge is (u, v).
        
        # We need to map (u, v) to the index in the final output array.
        # Final index for (i, j) with i < j:
        # idx = i*n - i*(i+1)//2 + (j - i - 1)
        # Let's verify:
        # i=0: j=1 -> 0 + 0 = 0. Correct.
        # i=0: j=2 -> 0 + 1 = 1. Correct.
        # i=1: j=2 -> 1*n - 1 + 0 = n-1. 
        # Let's check formula: sum_{k=0}^{i-1} (n-k-1) + (j-i-1)
        # Sum is i*n - i(i+1)/2.
        # So idx = i*n - i*(i+1)//2 + j - i - 1.
        
        def get_idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + j - i - 1

        colors = [-1] * num_edges
        # adj_mask[c][u] is a bitmask of neighbors of u in color c
        adj_mask = [[0] * n for _ in range(5)]
        
        # We will use a recursive backtracking function.
        # To allow restarts with different random seeds/orders, we can randomize the color order.
        # But the vertex order is fixed (0, 1, ..., n-1) for determinism in structure, 
        # though we can randomize color choices.
        
        # Optimization: Precompute the list of edges to assign in construction order.
        # construction_edges: list of (u, v, final_idx)
        construction_edges = []
        for v in range(1, n):
            for u in range(v):
                construction_edges.append((u, v, get_idx(u, v)))
        
        # We also need to know, for a given edge (u, v) and color c, 
        # how many common neighbors they already have in color c.
        # This is popcount(adj_mask[c][u] & adj_mask[c][v]).
        # If this count is >= 2, assigning color c creates a C4.
        
        # We can also use a heuristic: if a color is "tight" (count == 1), it's risky.
        # If count == 2, it's forbidden.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
                
            if idx == len(construction_edges):
                return True
                
            u, v, final_idx = construction_edges[idx]
            
            # Try colors 0..4
            # Heuristic: Try colors with fewer common neighbors first? 
            # Or randomize to escape local traps?
            # Let's randomize color order slightly to allow different paths in restarts.
            # But for speed, deterministic order might be faster if we use good heuristics.
            # Let's try colors 0,1,2,3,4 but shuffle them for each restart?
            # The function `try_solve` is called multiple times. We can pass a random seed or shuffle inside.
            
            # For now, fixed order 0-4. If we want diversity, we shuffle `color_order` before calling.
            
            for c in range(5):
                # Check constraint
                common = adj_mask[c][u] & adj_mask[c][v]
                if common.bit_count() >= 2:
                    continue
                    
                # Assign
                colors[final_idx] = c
                adj_mask[c][u] |= (1 << v)
                adj_mask[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                    
                # Unassign
                colors[final_idx] = -1
                adj_mask[c][u] &= ~(1 << v)
                adj_mask[c][v] &= ~(1 << u)
                
            return False

        # To introduce diversity, we can shuffle the color order for each restart.
        # But `backtrack` uses `range(5)`. 
        # Let's modify `backtrack` to accept a color order.
        
        def backtrack_with_order(idx, color_order):
            if time.time() > time_limit:
                return False
                
            if idx == len(construction_edges):
                return True
                
            u, v, final_idx = construction_edges[idx]
            
            for c in color_order:
                common = adj_mask[c][u] & adj_mask[c][v]
                if common.bit_count() >= 2:
                    continue
                    
                colors[final_idx] = c
                adj_mask[c][u] |= (1 << v)
                adj_mask[c][v] |= (1 << u)
                
                if backtrack_with_order(idx + 1, color_order):
                    return True
                    
                colors[final_idx] = -1
                adj_mask[c][u] &= ~(1 << v)
                adj_mask[c][v] &= ~(1 << u)
                
            return False

        # Try with a few different color orders
        for _ in range(10):
            if time.time() > time_limit:
                return None
            color_order = list(range(5))
            random.shuffle(color_order)
            # Reset state
            colors = [-1] * num_edges
            adj_mask = [[0] * n for _ in range(5)]
            if backtrack_with_order(0, color_order):
                return colors
        return None

    # Try to solve for increasing n
    # Start from 12, go up to 20 or 21.
    # If we fail for n, we break, assuming n+1 is harder.
    # However, backtracking is exponential. We need to be careful with time limits per n.
    
    # Time allocation:
    # Give each n a fixed time budget, e.g., 5 seconds? 
    # Or a shrinking budget?
    # Let's try 12 with 10s, 13 with 10s, etc.
    # If we find a solution for n, we keep it and try n+1.
    
    max_n_to_try = 20
    for n in range(12, max_n_to_try + 1):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time based on difficulty? 
        # Let's just give 5 seconds per attempt initially.
        time_limit_for_n = time.time() + 5.0
        if time_limit_for_n > deadline - 5.0:
            time_limit_for_n = deadline - 5.0
            
        # Increase time limit for smaller n? No, smaller n is easier.
        # Actually, backtracking might get stuck in bad branches.
        # Let's use a slightly larger budget for the first few.
        if n <= 14:
            time_limit_for_n = time.time() + 10.0
        else:
            time_limit_for_n = time.time() + 5.0
            
        if time_limit_for_n > deadline - 5.0:
            time_limit_for_n = deadline - 5.0

        res = try_solve(n, time_limit_for_n)
        if res is not None:
            best_n = n
            best_colors = res
            # If we found a solution, try the next n.
        else:
            # If we failed, break. 
            # Is it possible that n failed but n+1 succeeds? Unlikely.
            # Also, if n is hard, n+1 is harder.
            break

    return {'n': best_n, 'colors': best_colors}

