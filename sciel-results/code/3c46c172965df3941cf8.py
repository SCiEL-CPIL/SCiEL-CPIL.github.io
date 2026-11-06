import itertools
import time

DEADLINE = 130.0

def _is_not_2_colorable(edges, n):
    # Fast backtracking for small n
    if not edges:
        return False
        
    degrees = [0] * n
    for e in edges:
        for v in e:
            degrees[v] += 1
            
    # Sort vertices by degree
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    colors = [-1] * n
    
    def propagate(colors):
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                uncolored = []
                for v in e:
                    if colors[v] == 0: c0 += 1
                    elif colors[v] == 1: c1 += 1
                    else: uncolored.append(v)
                if c0 == 5 or c1 == 5: return False
                if len(uncolored) == 1:
                    if c0 == 4:
                        colors[uncolored[0]] = 1
                        changed = True
                    elif c1 == 4:
                        colors[uncolored[0]] = 0
                        changed = True
        return True

    def bt(idx):
        if idx == n:
            return True
        v = order[idx]
        if colors[v] != -1:
            return bt(idx+1)
        
        # Heuristic: try the color that forces more?
        # Just try 0 then 1
        colors[v] = 0
        if propagate(colors):
            if bt(idx+1): return True
        colors[v] = -1
        
        colors[v] = 1
        if propagate(colors):
            if bt(idx+1): return True
        colors[v] = -1
        
        return False

    return not bt(0)

def construct():
    start_time = time.time()
    
    # Start with K_9^5
    n = 9
    all_edges = list(itertools.combinations(range(n), 5))
    current_edges = [list(e) for e in all_edges]
    
    # Iterative deletion
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE - 10:
            break
        improved = False
        # Shuffle to avoid bias? Or just iterate.
        # Iterate in reverse to avoid index shifting issues if we remove
        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            if _is_not_2_colorable(test_edges, n):
                # It is still non-2-colorable, so remove it
                current_edges.pop(i)
                improved = True
    
    return {'n': n, 'edges': current_edges}
