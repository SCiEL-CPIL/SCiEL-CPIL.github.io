"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

def construct():
    # Deadline guard: finish well before the 150s wall-clock kill.
    _t0 = time.time()
    _DEADLINE = 135.0  # leave ~15s margin

    def _instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _build_conflict_masks(edges, adj):
        """
        Precompute conflict masks for all edges.
        conflict_mask[i] is an integer bitmask where bit j is 1 if edge i and edge j conflict.
        """
        n_edges = len(edges)
        conflict_masks = [0] * n_edges
        
        # Precompute endpoint sets for quick lookup? 
        # Actually, just iterate pairs. N is small (< 240).
        for i in range(n_edges):
            u1, v1 = edges[i]
            mask_i = 0
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                # Check conflict: share vertex or joined by edge
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    mask_i |= (1 << j)
                    conflict_masks[j] |= (1 << i)
                else:
                    # Check adjacency
                    if (u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]):
                        mask_i |= (1 << j)
                        conflict_masks[j] |= (1 << i)
            conflict_masks[i] = mask_i
        return conflict_masks

    def _greedy_color_bitmask(edges, conflict_masks, order):
        """
        Greedy coloring using bitmask conflict checks.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        colour = [-1] * n_edges
        # color_masks[c] is an integer bitmask of edges in color c
        color_masks = []
        max_c = 0
        
        for i in order:
            # Find smallest c such that color_masks[c] & conflict_masks[i] == 0
            placed = False
            for c in range(max_c):
                if (color_masks[c] & conflict_masks[i]) == 0:
                    colour[i] = c
                    color_masks[c] |= (1 << i)
                    placed = True
                    break
            
            if not placed:
                colour[i] = max_c
                color_masks.append(1 << i)
                max_c += 1
                
        return colour, max_c

    def _local_search_bitmask(colour, num_colors, edges, conflict_masks, time_limit=10.0):
        """
        Improve coloring by eliminating the highest color class.
        Uses 1-moves and 2-swaps.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colour, num_colors
        
        col = colour[:]
        
        # Reconstruct color_masks from col
        color_masks = [0] * num_colors
        for i, c in enumerate(col):
            color_masks[c] |= (1 << i)
            
        current_max_c = num_colors - 1
        
        # We want to reduce current_max_c
        while current_max_c >= 0:
            if time.time() - t_start > time_limit:
                break
            
            # Try to eliminate color current_max_c
            # Get list of edges in this color
            # Iterate through bits set in color_masks[current_max_c]
            mask_c = color_masks[current_max_c]
            if mask_c == 0:
                current_max_c -= 1
                continue
                
            # Order edges in this color by "flexibility" (fewest conflicts with lower colors?)
            # Or just iterate. Let's try iterating and see if we can empty it.
            # To make it deterministic and potentially better, sort by degree or something?
            # For now, just iterate.
            
            # We need to move all edges out of current_max_c.
            # Let's try to move them one by one.
            # If we get stuck, we might need to swap.
            
            # Strategy: 
            # 1. Try to move any edge from C to any c < C.
            # 2. If no simple move works, try 2-swaps.
            
            progress = True
            while progress and current_max_c >= 0:
                if time.time() - t_start > time_limit:
                    break
                progress = False
                
                mask_c = color_masks[current_max_c]
                if mask_c == 0:
                    current_max_c -= 1
                    continue
                    
                # Get edges in current_max_c
                # Iterate through set bits
                # We need to pick an edge to move.
                # Let's try all edges in current_max_c and see if any can be moved.
                
                # To avoid infinite loops, we track if we made any move in this pass.
                
                # List of edges in current_max_c
                edges_in_c = []
                m = mask_c
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    edges_in_c.append(idx)
                    m ^= lsb
                    
                # Try to move each edge
                moved_any = False
                for e in edges_in_c:
                    if time.time() - t_start > time_limit:
                        break
                        
                    # e is in current_max_c
                    # Try to move e to c < current_max_c
                    moved = False
                    for c in range(current_max_c):
                        # Check if e can go to c
                        if (color_masks[c] & conflict_masks[e]) == 0:
                            # Move e to c
                            color_masks[current_max_c] &= ~(1 << e)
                            color_masks[c] |= (1 << e)
                            col[e] = c
                            moved = True
                            moved_any = True
                            break
                        
                    if moved:
                        continue
                        
                    # Try 2-swap:
                    # Find a color c < current_max_c where e conflicts with some edges.
                    # Let S = conflict_masks[e] & color_masks[c]
                    # If S is not empty, pick f in S.
                    # Check if f can move to current_max_c (given e is removed from current_max_c).
                    # i.e., is (color_masks[current_max_c] & ~(1<<e) & conflict_masks[f]) == 0?
                    
                    for c in range(current_max_c):
                        S = conflict_masks[e] & color_masks[c]
                        if S == 0:
                            # This case is handled above (simple move)
                            # But wait, if S==0, we would have moved it.
                            # So here S != 0.
                            pass
                            
                        if S != 0:
                            # Try to swap with an edge f in S
                            # Iterate through bits in S
                            m_s = S
                            while m_s:
                                lsb = m_s & -m_s
                                f = lsb.bit_length() - 1
                                m_s ^= lsb
                                
                                # Check if f can move to current_max_c
                                # Remove e from current_max_c temporarily for check
                                temp_mask_c_max = color_masks[current_max_c] & ~(1 << e)
                                if (temp_mask_c_max & conflict_masks[f]) == 0:
                                    # Swap e and f
                                    color_masks[current_max_c] = (color_masks[current_max_c] & ~(1 << e)) | (1 << f)
                                    color_masks[c] = (color_masks[c] & ~(1 << f)) | (1 << e)
                                    col[e] = c
                                    col[f] = current_max_c
                                    moved = True
                                    moved_any = True
                                    break
                                if moved:
                                    break
                            
                            if moved:
                                break
                                
                if moved_any:
                    progress = True
                    # If we emptied current_max_c, the while loop condition `mask_c == 0` will handle decrement
                    # But we should check if it's empty now to break inner loop if needed?
                    # The outer while `while progress` will loop again.
                    # If current_max_c becomes empty, `mask_c` will be 0, and we decrement.
                    
            if not progress:
                # Stuck at this color. 
                # We could try a random restart or perturbation, but for now, let's just break
                # and accept this coloring. 
                # To improve, we could try swapping colors of two non-conflicting edges to change the landscape.
                # Given time constraints, simple break is safer.
                break
                
        # Final count
        # Find highest non-zero mask
        final_num_colors = 0
        for c in range(len(color_masks)):
            if color_masks[c] != 0:
                final_num_colors = c + 1
                
        return col, final_num_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict_masks = _build_conflict_masks(edges, adj)
        
        best_colours = None
        best_num_colors = float('inf')
        
        # Try multiple orderings
        n_edges = len(edges)
        
        # Order 1: Natural
        order1 = list(range(n_edges))
        # Order 2: Reverse
        order2 = list(range(n_edges - 1, -1, -1))
        # Order 3: Degree-based (approximate: sort by sum of endpoint degrees)
        # Precompute degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        order3 = sorted(range(n_edges), key=lambda i: deg[edges[i][0]] + deg[edges[i][1]], reverse=True)
        
        orders = [order1, order2, order3]
        
        for order in orders:
            col, num_c = _greedy_color_bitmask(edges, conflict_masks, order)
            if num_c < best_num_colors:
                best_colours = col
                best_num_colors = num_c
        
        # Local search on the best initial coloring
        if time.time() < _DEADLINE - 10:
            # Allocate remaining time roughly equally or based on graph size
            # Graph 3 is largest, give it more time? 
            # Or just use a fixed chunk for each.
            # Total time ~ 135s. 3 graphs. ~ 40s each? 
            # Let's use 30s per graph to be safe.
            remaining = _DEADLINE - time.time()
            time_limit_ls = max(5.0, remaining / 3.0 - 5.0)
            
            best_colours, best_num_colors = _local_search_bitmask(
                best_colours, best_num_colors, edges, conflict_masks, time_limit=time_limit_ls
            )
            
        out.append(best_colours)

    return out
# EVOLVE-BLOCK-END
