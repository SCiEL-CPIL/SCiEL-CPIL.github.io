import time
import random
from itertools import combinations

def check_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Build incidence structure
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    # For each edge, count how many vertices are color 0 and color 1
    # edge_counts[i] = [count0, count1]
    edge_counts = [[0, 0] for _ in range(len(edges))]
    
    # We'll use a stack for backtracking
    # State: (vertex, color)
    # But we need to handle unit propagation.
    
    # Let's implement a recursive backtracking that tries to force colors.
    # This is a standard SAT solver approach.
    
    def backtrack():
        # 1. Unit Propagation
        # Check if any edge is forced
        # If an edge has 4 vertices of color 0, the 5th MUST be 1 (to avoid monochromatic 0)
        # Wait, the condition is "no edge is monochromatic".
        # So if an edge has 4 vertices of color 0, the 5th CANNOT be 0. It MUST be 1.
        # If an edge has 5 vertices of color 0, it's invalid (monochromatic).
        
        forced = True
        while forced:
            forced = False
            for i in range(len(edges)):
                c0, c1 = edge_counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    # Must be color 1
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Assign v = 1
                            colors[v] = 1
                            for j in vertex_to_edges[v]:
                                edge_counts[j][1] += 1
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    # Must be color 0
                    for v in edges[i]:
                        if colors[v] == -1:
                            # Assign v = 0
                            colors[v] = 0
                            for j in vertex_to_edges[v]:
                                edge_counts[j][0] += 1
                            forced = True
                            break
            # Note: The above loop might assign multiple vertices in one pass, 
            # but we restart the while loop to catch cascades.
            # However, the inner 'for i' loop might assign a vertex, then 
            # subsequent checks in the same 'for i' iteration might be stale?
            # No, edge_counts is updated immediately.
            # But if we assigned a vertex in edge i, we should probably continue checking other edges.
            # The 'while forced' loop handles this.
        
        # 2. Find an unassigned vertex
        # Heuristic: Pick the vertex involved in the most constrained edges?
        # Or just the first unassigned one for simplicity.
        # Let's try to pick a vertex that appears in an edge with the highest "pressure".
        # Pressure = max(count0, count1) for edges containing v.
        
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True # All assigned, no contradiction -> 2-colorable
        
        # Pick a vertex
        # Simple: first unassigned
        v = unassigned[0]
        
        # Try color 0
        saved_colors = []
        saved_counts = []
        
        # Assign v = 0
        colors[v] = 0
        for j in vertex_to_edges[v]:
            saved_counts.append((edge_counts[j][0], edge_counts[j][1]))
            edge_counts[j][0] += 1
        
        if backtrack():
            return True
            
        # Undo
        colors[v] = -1
        for j, (c0, c1) in zip(vertex_to_edges[v], saved_counts):
            edge_counts[j][0] = c0
            edge_counts[j][1] = c1
            
        # Try color 1
        colors[v] = 1
        for j in vertex_to_edges[v]:
            saved_counts.append((edge_counts[j][0], edge_counts[j][1]))
            edge_counts[j][1] += 1
            
        if backtrack():
            return True
            
        # Undo
        colors[v] = -1
        for j, (c0, c1) in zip(vertex_to_edges[v], saved_counts):
            edge_counts[j][0] = c0
            edge_counts[j][1] = c1
            
        return False

    # We want to know if it is NOT 2-colorable.
    # So if backtrack() returns True (found a coloring), then it IS 2-colorable -> return False.
    # If backtrack() returns False (no coloring), then it is NOT 2-colorable -> return True.
    
    # Wait, the logic above:
    # backtrack() returns True if a valid coloring is found.
    # So check_non_2_colorable should return not backtrack().
    
    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # We will try to find a small non-2-colorable hypergraph.
    # Start with n=9, K_9^5 (126 edges).
    # Then use local search to remove edges.
    
    # To make this faster, we can pre-compute some structures or use a smarter heuristic.
    # But let's stick to the robust backtracking for now, and optimize the search.
    
    # One optimization: If we remove an edge, we only need to check if the graph is STILL non-2-colorable.
    # This is the expensive part.
    
    # Let's try a few different starting points and local searches.
    
    # Strategy 1: Start with n=9, remove edges.
    n_init = 9
    init_edges = [list(e) for e in combinations(range(n_init), 5)]
    
    if check_non_2_colorable(n_init, init_edges):
        best_edges = init_edges
        best_n = n_init
        best_m = len(init_edges)
    else:
        # Should not happen
        return {'n': 0, 'edges': []}
    
    current_n = best_n
    current_edges = best_edges[:]
    current_m = len(current_edges)
    
    # Local Search
    # We will use a simple hill climbing with random restarts if stuck.
    # Move: Remove a random edge. If still non-2-colorable, accept.
    # If stuck, try swapping an edge (remove one, add another).
    
    # Precompute all possible edges for n=9
    all_edges_9 = set(tuple(e) for e in combinations(range(9), 5))
    
    stuck_counter = 0
    max_stuck = 100
    
    while time.time() < deadline:
        if current_m == 0:
            break
            
        # Try to remove a random edge
        idx = random.randint(0, current_m - 1)
        e_to_remove = current_edges[idx]
        new_edges = current_edges[:idx] + current_edges[idx+1:]
        
        if check_non_2_colorable(current_n, new_edges):
            current_edges = new_edges
            current_m -= 1
            if current_m < best_m:
                best_m = current_m
                best_edges = current_edges[:]
                best_n = current_n
            stuck_counter = 0
        else:
            stuck_counter += 1
            
            # If stuck, try a swap
            if stuck_counter > max_stuck:
                # Find an edge to add
                current_set = set(tuple(e) for e in current_edges)
                available_adds = [e for e in all_edges_9 if e not in current_set]
                
                if available_adds:
                    e_to_add = random.choice(available_adds)
                    # Swap: remove e_to_remove, add e_to_add
                    new_edges_swap = [e for i, e in enumerate(current_edges) if i != idx]
                    new_edges_swap.append(list(e_to_add))
                    
                    if check_non_2_colorable(current_n, new_edges_swap):
                        current_edges = new_edges_swap
                        # m unchanged, but structure changed
                        stuck_counter = 0
                    else:
                        # Try just adding an edge? (Increases m, usually bad, but might escape local min)
                        # Or just restart?
                        pass
                else:
                    # No edges to add? We are at max density?
                    # Break and try different n?
                    break
            elif stuck_counter > 50:
                # Try a different edge to remove
                pass
        
        # If we've been stuck for a long time, maybe restart with a different initial graph?
        # Or try n=10?
        # For now, let's just keep trying.
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Final verification
    if best_edges is not None:
        if check_non_2_colorable(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    return {'n': best_n, 'edges': best_edges if best_edges else []}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    assert not check_non_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
