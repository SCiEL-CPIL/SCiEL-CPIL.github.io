import random
import itertools
import time

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Sort edges by length (all 5) and then by content for consistency
    # Unit propagation: if an edge has 4 vertices of the same color, the 5th must be the other.
    # We'll use a simple recursive backtracking with heuristics.
    
    # To make it fast, we can use a bitset or list representation.
    # For small n, a simple list of colors is fine.
    
    # Heuristic: pick the edge with the most constrained vertices?
    # For 5-uniform, all edges have 5 vertices.
    # We'll just pick the first edge that isn't fully colored yet?
    # Or better: pick the edge with the most vertices already assigned.
    
    def backtrack(colors, edge_idx):
        # Check if any edge is violated (monochromatic)
        # Actually, we check during assignment.
        # If we assign a color, we check all edges containing that vertex.
        
        # Find an edge that is not fully colored and has at least one uncolored vertex
        # To be efficient, we can maintain a list of "active" edges.
        # For simplicity in this short script, we'll just scan.
        # This might be slow for large n, but n <= 26 is small.
        
        # Check for violation
        for e in edges:
            colors_in_edge = [colors[v] for v in e]
            if -1 not in colors_in_edge:
                # Fully colored
                if all(c == colors_in_edge[0] for c in colors_in_edge):
                    return False # Monochromatic edge found
            else:
                # Check if it's possible to avoid monochromatic
                # If 4 are same, the 5th must be different.
                # If 3 are same, the other 2 must not be that color? No.
                # If 4 are 0, the 5th MUST be 1.
                zeros = colors_in_edge.count(0)
                ones = colors_in_edge.count(1)
                uncolored = colors_in_edge.count(-1)
                
                if zeros == 4 and uncolored == 1:
                    # The uncolored one must be 1
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 1
                            # Recursively check? Or just continue?
                            # This is unit propagation.
                            # We need to handle this carefully.
                            pass
                if ones == 4 and uncolored == 1:
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 0
                            pass
                            
        # If we didn't return False, we need to make a choice.
        # Find the first uncolored vertex?
        # Better: Find the most constrained edge.
        
        # For simplicity and robustness in the search loop, 
        # I'll use a simpler check: 
        # If we assign colors greedily, we might miss solutions.
        # But for non-2-colorable detection, we need to prove NO solution exists.
        # Backtracking is required.
        
        # Find an edge with the most assigned vertices
        best_edge = None
        max_assigned = -1
        for e in edges:
            assigned = sum(1 for v in e if colors[v] != -1)
            if assigned > max_assigned:
                max_assigned = assigned
                best_edge = e
        
        if max_assigned == 5:
            # All edges fully colored?
            # If we get here, no edge was monochromatic in the check above.
            # But we must ensure all edges are checked.
            # The check above returns False if any is mono.
            # If we finish the loop without returning False, and all edges are fully colored,
            # then it's a valid coloring?
            # Wait, the check above only returns False if mono.
            # If we go through all edges and none are mono, and all are fully colored,
            # then we have a valid 2-coloring.
            return True

        # Pick an uncolored vertex in best_edge
        # Try color 0, then color 1
        for v in best_edge:
            if colors[v] == -1:
                # Try 0
                colors[v] = 0
                if backtrack(colors, edge_idx):
                    return True
                colors[v] = -1
                
                # Try 1
                colors[v] = 1
                if backtrack(colors, edge_idx):
                    return True
                colors[v] = -1
                return False
        
        # Should not reach here if best_edge has uncolored vertices
        return False

    colors = [-1] * n
    return backtrack(colors, 0)

def construct():
    # We know K_9^(5) is non-2-colorable with 126 edges.
    # We try to find a smaller one.
    # Strategy: Randomized search for non-2-colorable hypergraphs.
    # We try n from 9 to 14.
    # For each n, we try edge counts e from 29 to 125.
    # We generate random hypergraphs and check if they are non-2-colorable.
    # If we find one, we return it.
    
    # To be safe, we start with the known 126-edge solution.
    default_edges = [list(e) for e in itertools.combinations(range(9), 5)]
    
    best_n = 9
    best_edges = default_edges
    best_e = 126
    
    start_time = time.time()
    timeout = 110 # Leave 10s for overhead
    
    # We want to minimize e.
    # We'll iterate e from 29 upwards.
    # For each e, we try a few random instances.
    # If we find a non-2-colorable one with e < best_e, we update best.
    
    # To make the search effective, we should bias towards dense graphs?
    # No, non-2-colorable graphs are usually dense.
    # But we want FEW edges.
    # This is counter-intuitive. Usually, sparse graphs are 2-colorable.
    # So we are looking for a "sparse" non-2-colorable graph.
    # These are rare.
    # The 51-edge construction is likely a specific design, not random.
    # Random search might fail to find it.
    
    # Alternative: Use the 9-vertex complete graph and try to remove edges?
    # If we remove an edge from K_9^(5), is it still non-2-colorable?
    # Probably yes, until we remove many.
    # Let's try to find a minimal subgraph of K_9^(5) that is non-2-colorable.
    # We can do this by iteratively removing edges and checking.
    # This is a "peeling" process.
    
    # Let's try this approach:
    # Start with K_9^(5).
    # While there is an edge that can be removed without making it 2-colorable, remove it.
    # This finds a MINIMAL non-2-colorable subgraph.
    # The size of this minimal subgraph might be smaller than 126.
    
    current_edges = set(map(tuple, default_edges))
    
    # To speed up, we'll only check a subset of edges for removal.
    # Or we can just try to remove all edges one by one in random order.
    
    # However, checking 2-colorability is expensive.
    # We have 120s.
    # Each check might take milliseconds if optimized, or seconds if not.
    # My simple backtracker is not optimized.
    
    # Let's use the simple backtracker but limit the depth or use a better heuristic.
    # Actually, for n=9, the search space is small.
    
    # Let's try to remove edges from K_9^(5).
    # We'll shuffle the edges and try to remove them.
    
    edge_list = list(current_edges)
    random.shuffle(edge_list)
    
    # We'll try to remove each edge if the remaining graph is still non-2-colorable.
    # This is O(E * T_check).
    # E=126. T_check might be 0.1s. Total 12s. Feasible.
    
    for e in edge_list:
        if time.time() - start_time > timeout:
            break
            
        # Try removing e
        new_edges_set = current_edges - {e}
        
        # Check if new_edges_set is non-2-colorable
        # We need to convert back to list for the checker
        new_edges_list = [list(edge) for edge in new_edges_set]
        
        # If it is 2-colorable, we must keep e.
        # If it is NOT 2-colorable, we can remove e.
        
        if is_2_colorable(9, new_edges_list):
            # It became 2-colorable, so we cannot remove e.
            pass
        else:
            # It is still non-2-colorable, so we remove e.
            current_edges = new_edges_set
            
    # After this process, we have a minimal non-2-colorable subgraph of K_9^(5).
    # Let's see how many edges are left.
    final_edges = [list(e) for e in current_edges]
    
    if len(final_edges) < best_e:
        best_e = len(final_edges)
        best_edges = final_edges
        best_n = 9
        
    # We can also try other n values if time permits, but K_9 is the smallest n.
    # Smaller n means fewer possible edges, so it's a good place to start.
    
    return {'n': best_n, 'edges': best_edges}
