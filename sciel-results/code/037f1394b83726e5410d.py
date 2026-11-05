import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Fast backtracking with unit propagation for 2-colorability of 5-uniform hypergraph.
    """
    if not edges:
        return True
        
    # Pre-process
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
    
    # Initialize counts
    for i, e in enumerate(edges):
        # All unknown initially, so count_unknown is 5, others 0.
        pass
        
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
                    continue # Already processed? No, we add to queue only when assigned.
            
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
                        continue # Should not happen if cU==1 and we just assigned v?
                    
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
        # Pressure = number of edges where it is the last unknown or second last?
        # Simple: Maximize number of edges with 4 colored vertices of same color.
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
                    # If c0==4 or c1==4, it's forced.
                    # Actually, if cU==1, the other 4 are colored.
                    # If they are all 0, v must be 1.
                    # If they are all 1, v must be 0.
                    # If mixed, no constraint yet.
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
        
        # Try 0 then 1 (or random order?)
        # Random order might help symmetry breaking
        colors_choice = [0, 1]
        random.shuffle(colors_choice)
        
        for c in colors_choice:
            # Save state
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            # Tentative assignment
            # We need to run propagation
            # Note: propagate modifies colors and edge_counts
            # We pass a queue with (v, c)
            # But we must ensure v is actually assigned in propagate.
            # My propagate checks if colors[v] != -1.
            # So we just add to queue.
            
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            # Restore state
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    # Start
    if propagate([]):
        return backtrack()
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Search for small non-2-colorable hypergraphs
    # n from 10 to 14
    # m from 30 to 50
    found = None
    
    for n in [10, 11, 12, 13, 14]:
        if time.time() > deadline - 10.0:
            break
        for m in range(30, 51):
            if time.time() > deadline - 10.0:
                break
            
            # Number of trials
            # Lower m is harder to find non-2-colorable.
            # Higher m is easier.
            # We want minimal m. So we try low m first.
            # But low m might take many trials.
            # Limit trials per (n,m) to keep moving.
            max_trials = 100 if m < 35 else 200
            
            for _ in range(max_trials):
                if time.time() > deadline - 10.0:
                    break
                
                # Generate random edges
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = all_edges[:m]
                
                if not is_2_colorable(n, edges):
                    found = {'n': n, 'edges': [list(e) for e in edges]}
                    break
            
            if found:
                break
        if found:
            break
            
    if found:
        return found
        
    # Fallback: Find a valid hypergraph quickly
    # Use n=8, which has 56 total edges.
    # A random subset of ~40-50 edges is very likely non-2-colorable.
    n_fb = 8
    all_edges_fb = list(combinations(range(n_fb), 5))
    random.shuffle(all_edges_fb)
    
    # Try m from 30 to 56
    for m_fb in range(30, 57):
        edges_fb = all_edges_fb[:m_fb]
        if not is_2_colorable(n_fb, edges_fb):
            return {'n': n_fb, 'edges': [list(e) for e in edges_fb]}
            
    # Absolute fallback (should not happen)
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
