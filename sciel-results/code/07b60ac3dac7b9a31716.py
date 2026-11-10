"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy:
 1. Start from the complete 5-uniform hypergraph K^5_9 (126 edges).
 2. Use greedy edge deletion with randomized orderings to find a minimal
    non-2-colorable subgraph.
 3. Use a fast backtracking solver with unit propagation for verification.
 4. Try multiple vertex counts and random seeds to find the smallest possible.
"""

import time
import random
from itertools import combinations

_DEADLINE_MARGIN = 10.0  # seconds to leave before the 120s limit


def _is_non_2colorable(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable.
    Returns True if no proper 2-coloring exists.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
        
    m = len(edges)
    if m == 0:
        return False
        
    # Map vertices to list of edge indices containing them
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Colors: -1 unassigned, 0, 1
    colors = [-1] * n
    # Count of 0s and 1s in each edge
    count_0 = [0] * m
    count_1 = [0] * m
    
    # Precompute edge vertices for faster access
    edge_verts = edges

    def propagate():
        """
        Apply unit propagation.
        Returns False if a conflict is found (edge becomes monochromatic).
        Returns True if no immediate conflict and propagation is stable.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0 = count_0[i]
                c1 = count_1[i]
                e = edge_verts[i]
                
                # Conflict: all 5 are same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit Propagation: 4 are same color, 1 unassigned
                # The unassigned one MUST be the other color
                if c0 == 4 and c1 == 0:
                    # Find the unassigned vertex
                    for v in e:
                        if colors[v] == -1:
                            # It must be 1
                            colors[v] = 1
                            count_1[i] += 1
                            # Update other edges containing v
                            for j in vert_to_edges[v]:
                                if j != i:
                                    count_1[j] += 1
                            changed = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in e:
                        if colors[v] == -1:
                            # It must be 0
                            colors[v] = 0
                            count_0[i] += 1
                            for j in vert_to_edges[v]:
                                if j != i:
                                    count_0[j] += 1
                            changed = True
                            break
        return True

    def backtrack():
        # Check for conflicts and propagate
        if not propagate():
            return False
            
        # Find unassigned vertex
        # Heuristic: Pick vertex with most unassigned neighbors in "tight" edges?
        # Simple heuristic: Pick vertex with highest degree (number of edges) among unassigned.
        # Or: Pick vertex involved in an edge with 3 or 4 assigned?
        
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True # Found a valid coloring
            
        # Score vertices to pick the most constrained one
        best_v = -1
        max_score = -1
        for v in unassigned:
            score = 0
            for j in vert_to_edges[v]:
                c0 = count_0[j]
                c1 = count_1[j]
                # Prefer edges that are close to being unit (4 assigned)
                if c0 + c1 >= 3:
                    score += 1
            if score > max_score:
                max_score = score
                best_v = v
                
        if best_v == -1:
            best_v = unassigned[0]
            
        v = best_v
        
        # Save state
        old_colors = colors[:]
        old_c0 = count_0[:]
        old_c1 = count_1[:]
        
        # Try color 0
        colors[v] = 0
        for j in vert_to_edges[v]:
            count_0[j] += 1
        if backtrack():
            return True
            
        # Restore state
        colors[:] = old_colors
        count_0[:] = old_c0
        count_1[:] = old_c1
        
        # Try color 1
        colors[v] = 1
        for j in vert_to_edges[v]:
            count_1[j] += 1
        if backtrack():
            return True
            
        return False
        
    return not backtrack()


def construct():
    """
    Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    Strategy: 
    1. Iterate over small vertex counts n (9, 10, 11, 12).
    2. Start with the complete 5-uniform hypergraph K^5_n.
    3. Perform iterative greedy edge deletion with randomized orderings.
    4. Keep the best (smallest) non-2-colorable hypergraph found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with a known valid graph (Complete K^5_9)
    # C(9,5) = 126
    n_init = 9
    best_m = 126
    best_graph = {'n': n_init, 'edges': [list(e) for e in combinations(range(n_init), 5)]}
    
    # Try different vertex counts
    # Smaller n means fewer total edges, so higher density required for non-2-colorability
    # K^5_9 is 126 edges. K^5_10 is 252 edges.
    # We expect the minimal core to be smaller than 126.
    
    for n in [9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 15:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        total_possible = len(all_edges)
        
        if total_possible < best_m:
            # If the complete graph is already smaller than our best, update best
            best_m = total_possible
            best_graph = {'n': n, 'edges': all_edges[:]}
        
        # If the complete graph is not smaller, we try to reduce it
        # We run multiple trials with different random shuffles of the edge list
        # to escape local minima of the greedy deletion process.
        
        num_trials = 100
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            # Create a copy of edges and shuffle
            current_edges = all_edges[:]
            random.shuffle(current_edges)
            
            # Greedy Deletion Pass
            # We iterate through the shuffled list.
            # If removing an edge keeps the graph non-2-colorable, we remove it.
            
            i = 0
            while i < len(current_edges):
                if time.time() > deadline - 2:
                    break
                    
                # Try removing edges[i]
                candidate_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if still non-2-colorable
                if _is_non_2colorable(n, candidate_edges):
                    # Keep it removed
                    current_edges = candidate_edges
                    # Do not increment i
                else:
                    # Put it back (it's already there)
                    i += 1
            
            # After one pass, we might be able to do another pass
            # But for now, let's just check if we improved
            
            if len(current_edges) < best_m:
                best_m = len(current_edges)
                best_graph = {'n': n, 'edges': current_edges[:]}
                
                # If we've reached a very low number, we can stop early
                if best_m <= 50:
                    return best_graph

    return best_graph
