"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # 1. Precompute all 4-cycles and their edge indices
    all_edges = []
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            all_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
    m = len(all_edges) # 190 edges
    
    # Map edge index to its position in the list
    # Precompute cycles as lists of edge indices
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # 3 distinct 4-cycles on {a,b,c,d}
        # 1: a-b-c-d-a
        e1 = [edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]]
        cycles.append(e1)
        # 2: a-b-d-c-a
        e2 = [edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]]
        cycles.append(e2)
        # 3: a-c-b-d-a
        e3 = [edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]]
        cycles.append(e3)
        
    num_cycles = len(cycles)
    
    # For each edge, store the list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(m)]
    for i, cyc in enumerate(cycles):
        for e_idx in cyc:
            edge_to_cycles[e_idx].append(i)

    # CSP Solver: Backtracking with MRV
    # State: colors array (size m), -1 for uncolored
    # We will try to color a specific set of active edges.
    
    def solve_coloring(active_edges_mask):
        """
        Tries to assign colors to active edges such that all 4-cycles formed by active edges are rainbow.
        Returns a list of colors for active edges, or None if impossible.
        """
        # Get indices of active edges
        active_indices = [i for i in range(m) if active_edges_mask[i]]
        if not active_indices:
            return []
            
        # Map active index to position in active_indices for faster lookup?
        # Actually, we just need to assign colors to these edges.
        
        # Heuristic: Order edges by degree in the constraint graph (number of cycles they touch)
        # Pre-compute this order once? No, it depends on which cycles are "active" (all 4 edges active).
        # But for simplicity, let's use the static count of cycles per edge.
        # Static count is fine for MRV approximation.
        
        # Sort active edges by static constraint count descending (most constrained first)
        sorted_active = sorted(active_indices, key=lambda x: len(edge_to_cycles[x]), reverse=True)
        
        colors = [-1] * m
        # We only care about colors of active edges
        
        # Forward checking:
        # When we assign a color to edge e, we check all cycles containing e.
        # If a cycle has 3 edges colored and the 4th is uncolored, we can restrict the 4th.
        # If a cycle has 4 edges colored, we check if they are distinct.
        
        # To make this efficient, we track for each uncolored edge, which colors are forbidden.
        # But with 20 colors and 190 edges, a simple recursive check might be slow in Python.
        
        # Optimized Backtracking:
        # We maintain a stack of assignments.
        # We use a recursive function.
        
        def backtrack(pos):
            if time.time() - t0 > DEADLINE:
                return False
            
            if pos == len(sorted_active):
                return True
                
            e_idx = sorted_active[pos]
            
            # Get current forbidden colors for e_idx based on colored neighbors in cycles
            # A color c is forbidden if there exists a cycle C containing e_idx such that
            # another edge e' in C is already colored c.
            # Since we assign one by one, we can just check the colored edges in each cycle containing e_idx.
            
            forbidden = [False] * num_colors
            
            for c_idx in edge_to_cycles[e_idx]:
                cycle = cycles[c_idx]
                # Check other 3 edges in this cycle
                for other_e in cycle:
                    if other_e == e_idx:
                        continue
                    if active_edges_mask[other_e] and colors[other_e] != -1:
                        forbidden[colors[other_e]] = True
            
            # Try each color not forbidden
            for c in range(num_colors):
                if not forbidden[c]:
                    colors[e_idx] = c
                    
                    # Check if this assignment creates an immediate conflict in any cycle
                    # where all 4 edges are now colored.
                    valid = True
                    for c_idx in edge_to_cycles[e_idx]:
                        cycle = cycles[c_idx]
                        # If all edges in this cycle are active and colored
                        if all(active_edges_mask[oe] and colors[oe] != -1 for oe in cycle):
                            cyc_colors = [colors[oe] for oe in cycle]
                            if len(set(cyc_colors)) != 4:
                                valid = False
                                break
                    
                    if valid:
                        if backtrack(pos + 1):
                            return True
                            
                    colors[e_idx] = -1
                    
            return False

        # Run backtracking
        # To avoid deep recursion limits, we might need to increase recursion limit
        # or implement iteratively. 190 depth is fine for default 1000.
        
        # Randomize the order of colors to help escape local minima if we were doing local search,
        # but for backtracking, deterministic order is fine unless we add randomness to edge order.
        # Let's add a bit of randomness to the edge order to avoid getting stuck in a specific bad path?
        # No, MRV is strong. Let's stick to deterministic first.
        
        if backtrack(0):
            # Extract colors for active edges
            result = [colors[i] for i in active_indices]
            return result
        else:
            return None

    # Main Logic:
    # Start with all edges active.
    active_mask = np.ones(m, dtype=bool)
    
    best_edges = []
    best_score = 0
    
    # Try to solve K20
    # If it fails, remove edges greedily based on "conflict centrality" or just random/heuristic removal.
    
    # Since backtracking can be exponential, we need a timeout guard inside the solver.
    # The `backtrack` function checks time.
    
    # Strategy:
    # 1. Try to color K20.
    # 2. If fail, identify edges to remove.
    #    We can use the "failed search" to identify which edges were assigned last or caused backtracking.
    #    Simpler: Randomly remove an edge, retry.
    #    Even better: Remove the edge with the highest "constraint degree" among those involved in the most backtracks?
    #    We don't track backtracks easily.
    
    # Let's try a different heuristic for removal:
    # If K20 fails, remove an edge that is part of the most 4-cycles (highest degree).
    # This reduces the constraint density.
    
    # Precompute edge degrees (number of 4-cycles containing the edge)
    edge_degrees = np.array([len(edge_to_cycles[i]) for i in range(m)])
    
    # We will iteratively remove edges if the solver fails.
    # To be efficient, we will keep the "best" valid coloring found so far.
    
    # Initial attempt
    colors_found = solve_coloring(active_mask)
    
    if colors_found is not None:
        # Success! K20 is colorable.
        for i, c in enumerate(colors_found):
            u, v = all_edges[i]
            best_edges.append([u, v, c])
        # Return immediately as this is the maximum possible
        return best_edges

    # If K20 failed, we need to remove edges.
    # We will remove edges one by one, prioritizing those with high degree, 
    # and re-run the solver.
    # To speed up, we can cache the "failure point" or just restart.
    
    # Let's try removing edges in order of high degree.
    # This might be slow if we have to restart the solver from scratch each time.
    # Optimization: Keep the partial assignment from the failed run?
    # The `colors` array inside `solve_coloring` is lost.
    
    # Let's modify `solve_coloring` to return the colors assigned so far on failure,
    # or just restart. Given 150s, and backtracking is fast if it fails early, restarting is okay.
    
    # Heuristic for removal:
    # Remove the edge with the highest degree among active edges.
    
    current_active_count = m
    while current_active_count > 0 and time.time() - t0 < DEADLINE:
        # Find the active edge with the highest degree
        # We need to consider only active edges
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            break
            
        # Get degrees of active edges
        active_degrees = edge_degrees[active_indices]
        # Find the index with max degree
        max_deg_idx_in_active = np.argmax(active_degrees)
        edge_to_remove = active_indices[max_deg_idx_in_active]
        
        # Remove it
        active_mask[edge_to_remove] = False
        current_active_count -= 1
        
        # Try to solve
        colors_found = solve_coloring(active_mask)
        
        if colors_found is not None:
            # Success
            for i, c in enumerate(colors_found):
                u, v = all_edges[i]
                best_edges.append([u, v, c])
            break # Found a valid coloring, stop removing
            
    # If we exited the loop without finding a solution (e.g., time out or no edges left),
    # we return the best we have. If best_edges is empty, we return a valid sparse fallback.
    
    if not best_edges:
        # Fallback: 1-factorization (known valid)
        edges = []
        for r in range(19):
            colour = r
            edges.append([19, r, colour])
            for i in range(1, 10):
                u = (r + i) % 19
                v = (r - i) % 19
                edges.append([u, v, colour])
        # Recolor one edge to use color 19
        for i, e in enumerate(edges):
            if e[0] == 19 and e[1] == 0:
                edges[i][2] = 19
                break
        return edges

    return best_edges

# EVOLVE-BLOCK-END
