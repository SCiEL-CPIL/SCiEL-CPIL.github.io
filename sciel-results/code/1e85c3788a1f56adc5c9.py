import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Precompute edge indices for each vertex
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # State of each edge: [count_0, count_1, count_unassigned]
    # Also need to know which vertex is unassigned if count_unassigned == 1
    # For simplicity, we can just recalculate or store the set of unassigned vertices per edge.
    # Recalculation is O(m*5) which is small.
    
    colors = [-1] * n
    
    # We'll use a stack for backtracking
    # Each step: (vertex, color)
    # We need to restore state.
    
    # A simpler recursive approach with a global state and undo log might be faster in Python for small n.
    
    # Let's use a recursive function with memoization on the set of unassigned vertices?
    # No, the coloring matters.
    
    # Let's stick to standard backtracking.
    
    def propagate(colors, edge_state):
        """
        edge_state: list of [c0, c1, c_unassigned] for each edge.
        Returns False if conflict, True if consistent.
        Modifies colors and edge_state.
        Returns a list of (vertex, color) assignments made for undo.
        """
        queue = []
        # Initially, we don't know which edges to check. 
        # In backtracking, we call propagate after assigning a vertex.
        # So we pass the affected edges.
        pass

    # This is getting complicated to implement efficiently in Python with undo.
    # Let's use a simpler, slightly slower but correct backtracking.
    # n <= 20, m <= 126.
    
    # Sort vertices by degree (number of edges they are in) to prune faster?
    # Or just pick the first unassigned vertex.
    
    def backtrack(idx, colors, edge_counts):
        # edge_counts[i] = [count_0, count_1]
        # We can derive unassigned = 5 - c0 - c1
        
        # Check for conflicts and unit propagation
        # We need to find the next vertex to assign.
        # For efficiency, we should pick a vertex involved in an edge with 4 assigned.
        
        # Find a vertex to assign
        target_vertex = -1
        target_color = -1
        forced = False
        
        # Check for conflicts
        for i in range(len(edges)):
            c0, c1 = edge_counts[i]
            if c0 == 5 or c1 == 5:
                return False
            if c0 == 4 and c1 == 0:
                # The remaining vertex must be 1
                for v in edges[i]:
                    if colors[v] == -1:
                        target_vertex = v
                        target_color = 1
                        forced = True
                        break
            elif c1 == 4 and c0 == 0:
                for v in edges[i]:
                    if colors[v] == -1:
                        target_vertex = v
                        target_color = 0
                        forced = True
                        break
            if forced:
                break
        
        if not forced:
            # Find any unassigned vertex
            for v in range(n):
                if colors[v] == -1:
                    target_vertex = v
                    break
            if target_vertex == -1:
                return True # All assigned, no conflict
            
            # Try color 0
            # We need to update edge_counts for edges containing target_vertex
            
        # Apply assignment
        colors[target_vertex] = target_color
        affected_edges = vertex_edges[target_vertex]
        
        # Save state for undo
        saved_counts = []
        for i in affected_edges:
            saved_counts.append(list(edge_counts[i]))
            if target_color == 0:
                edge_counts[i][0] += 1
            else:
                edge_counts[i][1] += 1
        
        # Recurse
        if backtrack(idx + 1, colors, edge_counts):
            return True
        
        # Undo
        colors[target_vertex] = -1
        for i, sc in zip(affected_edges, saved_counts):
            edge_counts[i] = sc
            
        # If it was forced, we don't try the other color
        if forced:
            return False
        
        # Try the other color
        other_color = 1 - target_color
        colors[target_vertex] = other_color
        saved_counts = []
        for i in affected_edges:
            saved_counts.append(list(edge_counts[i]))
            if other_color == 0:
                edge_counts[i][0] += 1
            else:
                edge_counts[i][1] += 1
        
        if backtrack(idx + 1, colors, edge_counts):
            return True
            
        # Undo
        colors[target_vertex] = -1
        for i, sc in zip(affected_edges, saved_counts):
            edge_counts[i] = sc
            
        return False

    # Initialize edge counts
    edge_counts = [[0, 0] for _ in edges]
    
    # To speed up, we can pick the first vertex with the highest degree?
    # The current logic picks the first unassigned vertex if no forced move.
    # This might be slow. Let's improve vertex selection.
    
    # Actually, the forced move check iterates through ALL edges.
    # We can maintain a list of "active" edges (those with 4 assigned).
    # But for n=9, m=126, the overhead is small.
    
    # Let's just run it.
    
    # One optimization: If n is small, we can just try all 2^n colorings?
    # 2^9 = 512. Very fast.
    # 2^10 = 1024.
    # 2^15 = 32768.
    # 2^20 = 1M.
    
    # If n <= 15, brute force is faster and simpler.
    if n <= 15:
        for mask in range(1 << n):
            # Check if this coloring is valid
            valid = True
            for e in edges:
                # Check if all vertices in e have the same bit in mask
                val = (mask >> e[0]) & 1
                for v in e[1:]:
                    if ((mask >> v) & 1) != val:
                        break
                else:
                    # All same color -> invalid
                    valid = False
                    break
            if valid:
                return True
        return False

    # For n > 15, use backtracking
    return backtrack(0, colors, edge_counts)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with K9^5 (126 edges)
    n_start = 9
    edges_start = [list(e) for e in combinations(range(n_start), 5)]
    best_edges = edges_start
    best_n = n_start
    
    # Try to shrink
    def shrink(n, edges):
        current_edges = edges[:]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle indices to pick random edge to remove
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                # Try removing edge i
                new_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable(n, new_edges):
                    current_edges = new_edges
                    changed = True
                    break # Restart with new list
        return current_edges
    
    # Run shrink multiple times
    for _ in range(5):
        if time.time() > deadline:
            break
        shrunk = shrink(best_n, best_edges)
        if len(shrunk) < len(best_edges):
            best_edges = shrunk
    
    # Also try random search on larger n
    # Try n=10, 11, 12
    for n_try in [10, 11, 12, 13]:
        if time.time() > deadline:
            break
        # Estimate required edges. 
        # Random 5-uniform hypergraph becomes non-2-colorable around m ~ n * log(n) ?
        # Let's try m from 50 to 100.
        for m_try in range(50, 100, 5):
            if time.time() > deadline:
                break
            for _ in range(10): # 10 random tries
                if time.time() > deadline:
                    break
                # Generate random hypergraph
                all_possible = list(combinations(range(n_try), 5))
                random.shuffle(all_possible)
                candidate_edges = all_possible[:m_try]
                if not is_2_colorable(n_try, candidate_edges):
                    # Found a non-2-colorable one. Shrink it.
                    shrunk = shrink(n_try, candidate_edges)
                    if len(shrunk) < len(best_edges):
                        best_edges = shrunk
                        best_n = n_try
                    break # Found one, try next m or n
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    # Verify
    print("Verifying...")
    assert not is_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
