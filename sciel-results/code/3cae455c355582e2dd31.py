import time
import random
from itertools import combinations

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
    edge_verts = edges # list of lists

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
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True # Found a valid coloring
            
        # Score vertices to pick the most constrained one
        # Prefer vertices in edges with 3+ assigned, then by degree
        best_v = -1
        max_score = -1
        for v in unassigned:
            score = 0
            for j in vert_to_edges[v]:
                c0 = count_0[j]
                c1 = count_1[j]
                assigned = c0 + c1
                # Strong preference for edges close to unit (4 assigned)
                if assigned == 4:
                    score += 100
                elif assigned == 3:
                    score += 10
                elif assigned == 2:
                    score += 1
            # Tie-break by degree
            score += len(vert_to_edges[v]) * 0.1
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

def _greedy_delete(n, edges, deadline):
    """
    Greedily delete edges while maintaining non-2-colorability.
    Returns the reduced edge list.
    """
    current = list(edges)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try each edge in random order
        random.shuffle(current)
        i = 0
        while i < len(current) and time.time() < deadline:
            # Try removing current[i]
            candidate = current[:i] + current[i+1:]
            if _is_non_2colorable(n, candidate):
                # Keep it removed
                current = candidate
                changed = True
                # Don't increment i since indices shifted
            else:
                i += 1
    return current

def construct():
    """
    Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    Strategy: 
    1. Start with complete hypergraphs on small n.
    2. Use greedy deletion with multiple restarts.
    3. Track the best (smallest) non-2-colorable hypergraph found.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initialize with a known valid graph (Complete K^5_9)
    n_init = 9
    all_edges_9 = [list(e) for e in combinations(range(n_init), 5)]
    best_m = len(all_edges_9)
    best_graph = {'n': n_init, 'edges': all_edges_9[:]}
    
    # Try different vertex counts, starting with smaller n for denser graphs
    # which are more likely to remain non-2-colorable after deletions
    for n in [9, 10, 11]:
        if time.time() > deadline - 15:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        total_possible = len(all_edges)
        
        # Only proceed if this could potentially beat our best
        if total_possible >= best_m:
            # Still try, because we might find a sparse subset
            pass
        
        # Multiple trials with different random orders
        num_trials = 30
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                    break
                    
            # Create a shuffled copy
            current_edges = all_edges[:]
            random.shuffle(current_edges)
            
            # Run greedy deletion
            current_edges = _greedy_delete(n, current_edges, deadline - 5)
            
            # Update best if improved
            if len(current_edges) < best_m:
                best_m = len(current_edges)
                best_graph = {'n': n, 'edges': current_edges}
                
                # If we've found something very good, we can reduce effort
                if best_m <= 51:
                    # Continue but with fewer trials
                    num_trials = min(num_trials, 10)
                    
    # Final verification
    if not _is_non_2colorable(best_graph['n'], best_graph['edges']):
        # Fallback to complete K^5_9
        best_graph = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return best_graph

# Test harness for local verification
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Edges: {len(res['edges'])}")
    print(f"Vertices: {res['n']}")
    is_nc = _is_non_2colorable(res['n'], res['edges'])
    print(f"Is Non-2-Colorable: {is_nc}")
