import time
import random
from itertools import combinations

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 115.0 # Leave 5s margin for 120s limit

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable (i.e., there exists a coloring 
    such that no edge is monochromatic).
    Returns False if it is NOT 2-colorable.
    """
    if not edges:
        return True

    # Adjacency list for constraint propagation
    # For each vertex, which edges does it belong to?
    vertex_to_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # We need a fast way to check if an edge is "satisfied" or "violated" or "pending"
    # Color: -1 (unassigned), 0, 1
    colors = [-1] * n
    
    # Precompute edge vertex lists for quick access
    edge_vertices = [set(e) for e in edges]
    
    # Counters for each edge: how many 0s, how many 1s
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    
    def propagate(queue):
        """
        Apply unit propagation.
        If an edge has 4 vertices of color 0, the 5th must be 1.
        If an edge has 5 vertices of color 0, contradiction (return False).
        """
        while queue:
            v = queue.pop()
            c = colors[v]
            if c == -1:
                continue
            for e_idx in vertex_to_edges[v]:
                # Update counts
                if c == 0:
                    count_0[e_idx] += 1
                else:
                    count_1[e_idx] += 1
                
                # Check for contradiction
                if count_0[e_idx] == 5 or count_1[e_idx] == 5:
                    return False
                
                # Check for unit clause
                # If 4 are 0, the remaining must be 1
                # If 4 are 1, the remaining must be 0
                if count_0[e_idx] == 4:
                    # Find the unassigned vertex in this edge
                    for u in edge_vertices[e_idx]:
                        if colors[u] == -1:
                            if colors[u] != 1: # It is -1
                                colors[u] = 1
                                count_1[e_idx] += 1 # Pre-emptive? No, we should be careful.
                                # Actually, let's just set color and push to queue.
                                # But we need to update counts correctly.
                                # Let's refactor: 
                                # When we assign a color, we increment the count for that edge.
                                # Here we are determining a forced assignment.
                                pass
                    # This logic is getting messy with pre-incrementing.
                    # Let's use a cleaner backtracking approach with explicit state.
        
        return True

    # The above propagation is tricky to get right with counts.
    # Let's use a simpler, recursive backtracking with a good heuristic.
    
    # Heuristic: Choose the vertex with the most colored neighbors in its edges?
    # Or simpler: Just pick the first uncolored vertex?
    # With n<=26 and edges<=200, even simple backtracking might be slow if not careful.
    # But unit propagation is very powerful.
    
    # Let's try a different implementation:
    # State: colors array
    # At each step, pick an uncolored vertex v.
    # Try color 0. Propagate. If contradiction, try color 1.
    # Propagation: 
    #   If we set v to 0:
    #   For each edge e containing v:
    #       If e already has 4 zeros, the 5th must be 1.
    #       If e already has 5 zeros, fail.
    #       If setting 5th to 1 causes another edge to have 5 ones, fail, etc.
    
    def check_coloring(colors):
        # Verify if current partial coloring leads to contradiction
        # This is effectively the propagation step.
        pass

    # Let's implement a robust backtracking solver.
    # To maximize speed, we will use a list of edges as tuples of vertices.
    # We will maintain `colors` list.
    # We will maintain `edge_counts` [ [c0, c1] for each edge ].
    
    edge_counts = [[0, 0] for _ in edges]
    colors = [-1] * n
    
    # Precompute: for each vertex, list of edge indices
    v_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_edges[v].append(i)
            
    # Heuristic: find uncolored vertex with max "impact"
    # Impact = number of edges where this vertex is the last uncolored one?
    # Or just number of uncolored neighbors in edges?
    
    def get_unassigned():
        unassigned = []
        for v in range(n):
            if colors[v] == -1:
                unassigned.append(v)
        return unassigned

    def propagate(v, c):
        """
        Assign color c to vertex v.
        Returns False if contradiction found.
        Modifies colors and edge_counts.
        Uses a queue for unit propagation.
        """
        queue = [(v, c)]
        while queue:
            cur_v, cur_c = queue.pop()
            if colors[cur_v] != -1:
                if colors[cur_v] != cur_c:
                    return False
                else:
                    continue
            
            colors[cur_v] = cur_c
            
            # Update edges containing cur_v
            for e_idx in v_edges[cur_v]:
                edge_counts[e_idx][cur_c] += 1
                
                # Contradiction check
                if edge_counts[e_idx][cur_c] == 5:
                    return False
                
                # Unit propagation check
                # If 4 are cur_c, the 5th must be 1-cur_c
                if edge_counts[e_idx][cur_c] == 4:
                    # Find the unassigned vertex in this edge
                    for u in edges[e_idx]:
                        if colors[u] == -1:
                            # Must be 1 - cur_c
                            if cur_c == 0:
                                queue.append((u, 1))
                            else:
                                queue.append((u, 0))
                            break
                            
        return True

    def backtrack():
        # Find unassigned vertex
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True # All assigned, no contradiction found
        
        # Heuristic: pick vertex that appears in most "tight" edges?
        # Simple heuristic: pick first one. Or one with most colored neighbors.
        # Let's pick the one with the most colored neighbors in its edges.
        best_v = -1
        max_colored = -1
        for v in unassigned:
            colored_count = 0
            for e_idx in v_edges[v]:
                for u in edges[e_idx]:
                    if colors[u] != -1:
                        colored_count += 1
            if colored_count > max_colored:
                max_colored = colored_count
                best_v = v
                
        if best_v == -1:
            return True # Should not happen if unassigned not empty
            
        # Try color 0
        # Save state
        old_colors = colors[:]
        old_counts = [c[:] for c in edge_counts]
        
        if propagate(best_v, 0):
            if backtrack():
                return True
        
        # Restore state
        colors[:] = old_colors
        for i in range(len(edge_counts)):
            edge_counts[i] = old_counts[i][:]
            
        # Try color 1
        old_colors = colors[:]
        old_counts = [c[:] for c in edge_counts]
        
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    
    # Initial hypergraph: K_9^5
    n = 9
    initial_edges = [list(e) for e in combinations(range(n), 5)]
    
    best_edges = initial_edges
    best_score = len(initial_edges)
    
    # Try multiple randomized greedy deletions
    for i in range(50): # Max 50 restarts
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        current_edges = initial_edges[:]
        random.shuffle(current_edges)
        
        # Greedy deletion
        for e_idx in range(len(current_edges)):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            edge_to_test = current_edges[e_idx]
            # Remove it
            remaining = [e for e in current_edges if e != edge_to_test]
            
            # Check if remaining is 2-colorable
            # If it IS 2-colorable, we MUST keep the edge.
            # If it IS NOT 2-colorable, we can remove it.
            
            # Optimization: Only check if the edge is "critical"
            # We just check every time.
            
            is_col = is_2_colorable(n, remaining)
            
            if not is_col:
                # It is still non-2-colorable without this edge.
                # So we remove it permanently.
                current_edges = remaining
            else:
                # It became 2-colorable. We must keep it.
                pass # It's already in current_edges
                
        # Update best
        if len(current_edges) < best_score:
            best_score = len(current_edges)
            best_edges = current_edges[:]
            
            # If we hit the known lower bound or close to it, we can stop or try harder
            if best_score <= 51:
                # We found a very good solution. 
                # We can try to optimize further with local search if time permits, 
                # but 51 is the target.
                pass

    return {'n': n, 'edges': best_edges}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Found hypergraph with {len(res['edges'])} edges on {res['n']} vertices.")
    # Verify
    print("Verifying...")
    valid = not is_2_colorable(res['n'], res['edges'])
    print(f"Is non-2-colorable: {valid}")
