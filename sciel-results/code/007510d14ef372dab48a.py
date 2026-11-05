import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Returns False if no 2-coloring exists (non-2-colorable).
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    m = len(edges)
    # Map vertex to list of edge indices
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    # State
    colors = [-1] * n
    # edge_counts[e] = [count_0, count_1, count_unknown]
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        """
        Process queue of (vertex, color) assignments.
        Returns False if contradiction, True otherwise.
        Modifies colors and edge_counts.
        """
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            # Assign v to c
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                # Update counts
                edge_counts[e_idx][2] -= 1 # unknown
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                # Check contradiction: monochromatic
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation
                if cU == 1:
                    # One unknown left
                    # Find it
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        # Must be 1
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        # Must be 0
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        # Heuristic: Pick unassigned vertex with highest "pressure"
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                if cU == 1:
                    # This vertex is the last unknown in this edge.
                    if c0 == 4 or c1 == 4:
                        score += 10 # Strong constraint
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True # All assigned
            
        v = best_v
        
        # Try 0 then 1
        for c in [0, 1]:
            # Save state
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            # Tentative assignment
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            # Restore state
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    if propagate([]):
        return backtrack()
    return False

def greedy_set_cover(n):
    """
    Finds a small set of 5-edges such that every 2-coloring has at least one monochromatic edge.
    This guarantees the hypergraph is NOT 2-colorable.
    """
    # Universe: All 2-colorings. We fix vertex 0 to color 0 to break symmetry.
    # Number of colorings: 2^(n-1)
    num_colorings = 1 << (n - 1)
    
    # Generate all possible 5-edges
    all_edges = list(combinations(range(n), 5))
    
    # Precompute the bitset of colorings for which each edge is monochromatic.
    # S[e_idx] is an integer where bit i is 1 if coloring i makes edge e_idx monochromatic.
    S = [0] * len(all_edges)
    
    # Optimization: Precompute color bits for each vertex in each coloring?
    # Or just iterate colorings and update edges?
    # Given n is small (<=15), 2^(n-1) is <= 8192.
    # Edges <= 2002.
    # Total ops ~ 16M. This is fine in Python if optimized.
    
    # Let's represent coloring as a bitmask of n bits.
    # Coloring c: bit v is 1 if vertex v is color 1, 0 if color 0.
    # We fix vertex 0 to 0. So bit 0 is always 0.
    
    # For each coloring idx (0 to num_colorings-1):
    # The actual mask is: (idx << 1) | 0  ? No.
    # Let's map idx to mask.
    # idx bits 0..n-2 correspond to vertices 1..n-1.
    # mask = idx << 1. (Vertex 0 is 0).
    
    for idx in range(num_colorings):
        mask = idx << 1  # Vertex 0 is 0
        
        # For each edge, check if monochromatic
        # To speed up, we can iterate edges and check condition.
        for e_idx, e in enumerate(all_edges):
            # Check if all vertices in e have same color in mask
            # Get color of first vertex
            v0 = e[0]
            c0 = (mask >> v0) & 1
            
            mono = True
            for v in e[1:]:
                cv = (mask >> v) & 1
                if cv != c0:
                    mono = False
                    break
            
            if mono:
                S[e_idx] |= (1 << idx)
                
    # Greedy Set Cover
    uncovered = (1 << num_colorings) - 1
    selected_edges = []
    used = [False] * len(all_edges)
    
    while uncovered:
        best_e_idx = -1
        best_count = -1
        
        for e_idx in range(len(all_edges)):
            if used[e_idx]:
                continue
            # Count how many currently uncovered colorings this edge covers
            cover = S[e_idx] & uncovered
            count = cover.bit_count()
            if count > best_count:
                best_count = count
                best_e_idx = e_idx
        
        if best_e_idx == -1 or best_count == 0:
            # Should not happen if universe is covered
            break
        
        selected_edges.append(all_edges[best_e_idx])
        used[best_e_idx] = True
        uncovered &= ~S[best_e_idx]
        
    return [list(e) for e in selected_edges]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_edge_count = 1000
    
    # Try different n values
    # The optimal n for 5-uniform non-2-colorable graphs is likely around 12-14
    for n in [11, 12, 13, 14, 15]:
        if time.time() > deadline - 10.0:
            break
        
        try:
            edges = greedy_set_cover(n)
        except Exception:
            continue
            
        if len(edges) < best_edge_count:
            # Verify with backtracking to be absolutely sure
            # This is fast for small edge counts
            if not is_2_colorable(n, edges):
                best_edge_count = len(edges)
                best_result = {'n': n, 'edges': edges}
                
                # If we find a very good solution, we can stop
                if best_edge_count <= 51:
                    break
            else:
                # This shouldn't happen if set cover is correct, but just in case
                print(f"Warning: Set cover for n={n} produced 2-colorable graph?")
        
        if best_edge_count <= 51:
            break

    # Fallback if greedy set cover failed or found nothing (unlikely)
    if best_result is None:
        # Use a known small non-2-colorable graph or dense graph
        # n=9, all edges is 126.
        n_fb = 9
        all_edges_fb = list(combinations(range(n_fb), 5))
        return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
        
    return best_result
