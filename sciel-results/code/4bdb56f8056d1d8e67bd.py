import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Map vertices to indices 0..n-1
    # Edges are lists of vertices
    
    # Pre-process: Convert edges to frozensets for quick lookup?
    # For backtracking, we need to know which edges are affected by a vertex assignment.
    
    # State:
    # color[v] = 0, 1, or None
    # For each edge, we track:
    # - count_0: number of vertices colored 0
    # - count_1: number of vertices colored 1
    # - uncolored: number of vertices uncolored
    
    # Unit propagation:
    # If an edge has count_1 == 5 (all 1), it's monochromatic -> Invalid coloring for that edge.
    # Wait, the condition for 2-colorability is that NO edge is monochromatic.
    # So if an edge becomes monochromatic (all 0 or all 1), the current branch is invalid.
    # Also, if an edge has 4 vertices of color 0 and 1 uncolored, the uncolored vertex MUST be 1.
    # If an edge has 4 vertices of color 1 and 1 uncolored, the uncolored vertex MUST be 0.
    
    def backtrack(colors, edge_states):
        # colors: list of 0, 1, or None
        # edge_states: list of [count_0, count_1, uncolored_vertices_list]
        
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            for i, (c0, c1, uncol) in enumerate(edge_states):
                # Check for contradiction: monochromatic
                if c0 == 5: return False
                if c1 == 5: return False
                
                # Check for forced assignments
                if c0 == 4 and len(uncol) == 1:
                    v = uncol[0]
                    if colors[v] is None:
                        colors[v] = 1
                        # Update edge states
                        for j, (c0j, c1j, uncolj) in enumerate(edge_states):
                            if v in uncolj:
                                edge_states[j][1] += 1
                                edge_states[j][2] = [u for u in uncolj if u != v]
                        changed = True
                    elif colors[v] == 0:
                        return False # Contradiction
                elif c1 == 4 and len(uncol) == 1:
                    v = uncol[0]
                    if colors[v] is None:
                        colors[v] = 0
                        for j, (c0j, c1j, uncolj) in enumerate(edge_states):
                            if v in uncolj:
                                edge_states[j][0] += 1
                                edge_states[j][2] = [u for u in uncolj if u != v]
                        changed = True
                    elif colors[v] == 1:
                        return False # Contradiction

        # 2. Check if all colored
        if all(c is not None for c in colors):
            return True # All edges are non-monochromatic
            
        # 3. Choose a vertex to branch on
        # Heuristic: pick a vertex in an edge with most uncolored?
        # Or just first uncolored.
        v = next((i for i, c in enumerate(colors) if c is None), None)
        if v is None:
            return True

        # Try 0
        colors[v] = 0
        new_edge_states = []
        valid = True
        for c0, c1, uncol in edge_states:
            if v in uncol:
                new_c0 = c0 + 1
                new_c1 = c1
                new_uncol = [u for u in uncol if u != v]
            else:
                new_c0 = c0
                new_c1 = c1
                new_uncol = uncol
            if new_c0 == 5:
                valid = False
                break
            new_edge_states.append([new_c0, new_c1, new_uncol])
        
        if valid and backtrack(colors, new_edge_states):
            return True

        # Try 1
        colors[v] = 1
        new_edge_states = []
        valid = True
        for c0, c1, uncol in edge_states:
            if v in uncol:
                new_c0 = c0
                new_c1 = c1 + 1
                new_uncol = [u for u in uncol if u != v]
            else:
                new_c0 = c0
                new_c1 = c1
                new_uncol = uncol
            if new_c1 == 5:
                valid = False
                break
            new_edge_states.append([new_c0, new_c1, new_uncol])
        
        if valid and backtrack(colors, new_edge_states):
            return True

        colors[v] = None
        return False

    # Initialize edge states
    edge_states = []
    for e in edges:
        c0 = 0
        c1 = 0
        uncol = list(e)
        edge_states.append([c0, c1, uncol])
        
    colors = [None] * n
    return backtrack(colors, edge_states)

def construct():
    # Target: Find a non-2-colorable 5-uniform hypergraph with minimal edges.
    # Known optimal is 51 edges.
    # We will try to find one with n=10, 11, 12 vertices.
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_edges = None
    best_n = None
    
    # Try small n first
    for n in range(10, 14):
        if time.time() > deadline:
            break
            
        # Randomized search
        # We want to find a set of edges that is NOT 2-colorable.
        # Start with a random set of edges and add more if it is 2-colorable.
        
        # Strategy: 
        # 1. Generate a random permutation of all possible edges.
        # 2. Add edges one by one.
        # 3. After each addition, check if it's 2-colorable.
        # 4. If it becomes non-2-colorable, stop.
        # 5. Record the number of edges.
        # 6. Try to remove edges to minimize.
        
        all_edges = list(combinations(range(n), 5))
        
        # To find a small one, we might need to be smart.
        # Random addition often leads to large counts.
        # Let's try a specific known structure or a more directed search.
        
        # Actually, for n=10, m(5) is likely close to 51?
        # Let's just run a few random restarts with edge addition.
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            random.shuffle(all_edges)
            current_edges = []
            
            # Add edges until non-2-colorable
            # We can skip some checks for speed? No, must be exact.
            # But is_2_colorable is fast for small n and few edges.
            
            # Optimization: Only check every k edges? 
            # If we add 1 edge and it's 2-colorable, adding another might make it non-2-colorable.
            # We need to find the MINIMAL set.
            # So we add one, check. If 2-colorable, add next.
            
            for e in all_edges:
                current_edges.append(e)
                
                # Check if non-2-colorable
                if not is_2_colorable(n, [list(x) for x in current_edges]):
                    # Found a non-2-colorable hypergraph
                    # Try to minimize
                    # Remove edges if possible
                    minimized = True
                    while minimized:
                        minimized = False
                        # Try removing each edge
                        for i in range(len(current_edges)):
                            if time.time() > deadline:
                                break
                            candidate = current_edges[:i] + current_edges[i+1:]
                            if not is_2_colorable(n, [list(x) for x in candidate]):
                                current_edges = candidate
                                minimized = True
                                break
                        if time.time() > deadline:
                            break
                            
                    if best_edges is None or len(current_edges) < len(best_edges):
                        best_edges = [list(x) for x in current_edges]
                        best_n = n
                        # If we found 51, we can stop? 
                        # The task says "Minimize". 51 is the known minimum.
                        if len(current_edges) <= 51:
                            return {'n': best_n, 'edges': best_edges}
                    break # Stop adding edges for this restart
                    
            if best_edges and len(best_edges) <= 60:
                # Good enough? Let's keep searching for a bit if time permits.
                pass
                
    if best_edges:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to the original 126 edges if search failed (should not happen)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
