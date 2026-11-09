import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    if not edges:
        return True
        
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    count = [[0, 0] for _ in range(len(edges))]
    
    def propagate_from(v, colors, count):
        queue = [v]
        while queue:
            u = queue.pop()
            cu = colors[u]
            for e in vertex_edges[u]:
                count[e][cu] += 1
                if count[e][cu] == 5:
                    return False
                elif count[e][cu] == 4:
                    for w in edges[e]:
                        if colors[w] == -1:
                            colors[w] = 1 - cu
                            queue.append(w)
                            break
        return True

    def backtrack():
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            for e in range(len(edges)):
                if count[e][0] == 0 or count[e][1] == 0:
                    return False
            return True
        
        # Heuristic: pick vertex with most assigned neighbors
        # Or just first
        v = unassigned[0]
        
        # Try 0
        old_colors = colors[:]
        old_count = [c[:] for c in count]
        colors[v] = 0
        if propagate_from(v, colors, count):
            if backtrack():
                return True
        colors[:] = old_colors
        for i in range(len(count)):
            count[i] = old_count[i][:]
            
        # Try 1
        colors[v] = 1
        if propagate_from(v, colors, count):
            if backtrack():
                return True
        colors[:] = old_colors
        for i in range(len(count)):
            count[i] = old_count[i][:]
            
        return False

    if n > 0:
        colors[0] = 0
        if not propagate_from(0, colors, count):
            return False
        return backtrack()
    else:
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_n = 9
    
    fallback_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = fallback_edges
    best_n = 9
    
    for n in range(5, 27):
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            random.shuffle(all_edges)
            current_edges = []
            
            for edge in all_edges:
                if time.time() > deadline:
                    break
                
                candidate = current_edges + [edge]
                if not is_2_colorable(n, candidate):
                    current_edges = candidate
                    
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(len(current_edges)):
                            trial_edges = current_edges[:i] + current_edges[i+1:]
                            if not is_2_colorable(n, trial_edges):
                                current_edges = trial_edges
                                improved = True
                                break
                    
                    if best_edges is None or len(current_edges) < len(best_edges):
                        best_edges = [list(e) for e in current_edges]
                        best_n = n
                    break
    
    return {'n': best_n, 'edges': best_edges}
