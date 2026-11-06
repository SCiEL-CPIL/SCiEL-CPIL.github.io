import time
import random
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Build incidence
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    unassigned = [5] * m
    count0 = [0] * m
    count1 = [0] * m
    
    def propagate(v, c):
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        for e_idx in vertex_edges[v]:
            unassigned[e_idx] -= 1
            if c == 0:
                count0[e_idx] += 1
                if count0[e_idx] == 5:
                    return False
            else:
                count1[e_idx] += 1
                if count1[e_idx] == 5:
                    return False
            
            if unassigned[e_idx] == 1:
                e = edges[e_idx]
                # Find the single unassigned vertex
                for u in e:
                    if colors[u] == -1:
                        # Must be opposite color
                        if not propagate(u, 1 - c):
                            return False
                        break
            elif unassigned[e_idx] == 0:
                # Fully assigned, check validity
                if count0[e_idx] == 5 or count1[e_idx] == 5:
                    return False
        return True

    def backtrack():
        # Find edge with min unassigned > 0
        min_unassigned = 6
        candidate_vertices = []
        
        for i in range(m):
            if unassigned[i] > 0 and unassigned[i] < min_unassigned:
                min_unassigned = unassigned[i]
                candidate_vertices = edges[i]
                if min_unassigned == 2:
                    break
        
        if min_unassigned == 6:
            return True
            
        # Pick a vertex from candidate_vertices
        # Randomize to avoid bias
        v = random.choice(candidate_vertices)
        
        # Save state
        old_colors = colors[:]
        old_unassigned = unassigned[:]
        old_count0 = count0[:]
        old_count1 = count1[:]
        
        if propagate(v, 0):
            if backtrack():
                return True
        
        # Restore
        colors[:] = old_colors
        unassigned[:] = old_unassigned
        count0[:] = old_count0
        count1[:] = old_count1
        
        if propagate(v, 1):
            if backtrack():
                return True
        
        # Restore
        colors[:] = old_colors
        unassigned[:] = old_unassigned
        count0[:] = old_count0
        count1[:] = old_count1
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    def try_remove_edges(edges, n, time_limit):
        edges = [list(e) for e in edges]
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Shuffle order of removal attempts to avoid bias
            indices = list(range(len(edges)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                # Skip if index is out of range (edge might have been removed)
                if i >= len(edges):
                    continue
                remaining = edges[:i] + edges[i+1:]
                if not is_2_colorable(remaining, n):
                    edges = remaining
                    changed = True
                    break
        return edges

    # Try different n values
    for n in range(8, 13):
        if time.time() > deadline - 20:
            break
            
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Try different subset sizes
        for trial in range(20):
            if time.time() > deadline - 10:
                break
                
            # Pick a random subset size
            subset_size = min(len(all_edges), 80)
            subset = [list(e) for e in all_edges[:subset_size]]
            
            if not is_2_colorable(subset, n):
                # It is non-2-colorable, try to minimize
                reduced = try_remove_edges(subset, n, time_limit=time.time() + 5.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_n = n
                    best_edges = reduced
            
            # Try a different random shuffle for next trial
            random.shuffle(all_edges)

    return {'n': best_n, 'edges': best_edges}
