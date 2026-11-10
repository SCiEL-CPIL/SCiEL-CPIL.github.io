import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph (n, edges) is 2-colorable.
    edges is a list of lists of 5 vertices.
    """
    m = len(edges)
    if m == 0:
        return True
    
    # Pre-compute vertex to edges mapping
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    color = [-1] * n
    count0 = [0] * m
    count1 = [0] * m
    
    # Propagation function
    # Returns False if conflict, True otherwise
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(m):
                e = edges[i]
                c0 = count0[i]
                c1 = count1[i]
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4:
                    # The unassigned vertex must be 1
                    for v in e:
                        if color[v] == -1:
                            color[v] = 1
                            count1[i] += 1
                            for j in vertex_edges[v]:
                                if j != i:
                                    count1[j] += 1
                            changed = True
                            break
                            
                if c1 == 4:
                    # The unassigned vertex must be 0
                    for v in e:
                        if color[v] == -1:
                            color[v] = 0
                            count0[i] += 1
                            for j in vertex_edges[v]:
                                if j != i:
                                    count0[j] += 1
                            changed = True
                            break
        return True

    def backtrack():
        if not propagate():
            return False
        
        # Find an unassigned vertex
        # Heuristic: pick vertex with most unassigned neighbors in tight edges?
        # For now, just pick the first one.
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned
            # Check if any edge is monochromatic (should be caught by propagate, but just in case)
            for i in range(m):
                if count0[i] == 5 or count1[i] == 5:
                    return False
            return True
        
        # Try color 0
        color[v] = 0
        saved = []
        for j in vertex_edges[v]:
            saved.append((j, count0[j], count1[j]))
            count0[j] += 1
            
        if backtrack():
            return True
            
        # Undo
        color[v] = -1
        for j, c0, c1 in saved:
            count0[j] = c0
            count1[j] = c1
            
        # Try color 1
        color[v] = 1
        saved = []
        for j in vertex_edges[v]:
            saved.append((j, count0[j], count1[j]))
            count1[j] += 1
            
        if backtrack():
            return True
            
        # Undo
        color[v] = -1
        for j, c0, c1 in saved:
            count0[j] = c0
            count1[j] = c1
            
        return False

    return backtrack()

def construct():
    start_time = time.time()
    time_limit = 110
    best_edges = None
    best_m = float('inf')
    best_n = 9
    
    # Try different n values
    for n in [9, 10, 11, 12]:
        if time.time() - start_time > time_limit:
            break
            
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with a reasonably dense graph to ensure it's likely non-2-colorable
        # If n=9, max 126. If n=11, max 462.
        # Let's start with 60% of edges or at least 40 edges.
        m_init = min(len(all_edges), max(40, int(len(all_edges) * 0.6)))
        
        # Try a few random subsets
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random edges
            edges = [list(e) for e in all_edges[:m_init]]
            random.shuffle(edges)
            
            # Check if non-2-colorable
            if not is_2_colorable(n, edges):
                # Found a non-2-colorable hypergraph
                current_edges = edges[:]
                current_m = len(current_edges)
                
                # Minimize edges
                # Shuffle to randomize removal order
                random.shuffle(current_edges)
                i = 0
                while i < len(current_edges):
                    if time.time() - start_time > time_limit:
                        break
                        
                    # Try removing edge i
                    temp_edges = current_edges[:i] + current_edges[i+1:]
                    
                    if not is_2_colorable(n, temp_edges):
                        # It is still non-2-colorable without this edge
                        current_edges = temp_edges
                        current_m -= 1
                        # Do not increment i
                    else:
                        # Edge is necessary
                        i += 1
                        
                if current_m < best_m:
                    best_m = current_m
                    best_edges = current_edges[:]
                    best_n = n
                    
                # If we hit the theoretical lower bound or a very good number, stop
                if best_m <= 30:
                    return {'n': best_n, 'edges': best_edges}
    
    if best_edges is None:
        # Fallback
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
