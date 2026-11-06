import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    def propagate(colors):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                reds = 0
                blues = 0
                unassigneds = 0
                unassigned_idx = -1
                
                for v in range(n):
                    if mask & (1 << v):
                        c = colors[v]
                        if c == 1:
                            reds += 1
                        elif c == 0:
                            blues += 1
                        else:
                            unassigneds += 1
                            unassigned_idx = v
                            
                if reds == 5 or blues == 5:
                    return False
                if reds == 4 and unassigneds == 1:
                    if colors[unassigned_idx] == -1:
                        colors[unassigned_idx] = 0
                        changed = True
                elif blues == 4 and unassigneds == 1:
                    if colors[unassigned_idx] == -1:
                        colors[unassigned_idx] = 1
                        changed = True
        return True

    def backtrack(colors):
        next_v = -1
        for i in range(n):
            if colors[i] == -1:
                next_v = i
                break
                
        if next_v == -1:
            return True
            
        for color in [0, 1]:
            new_colors = colors[:]
            new_colors[next_v] = color
            if propagate(new_colors):
                if backtrack(new_colors):
                    return True
        return False

    colors = [-1] * n
    colors[0] = 1
    if not propagate(colors):
        return False
    return backtrack(colors)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    best_edges = all_edges
    best_count = len(all_edges)
    
    # Greedy removal with random restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        edges = all_edges[:]
        random.shuffle(edges)
        
        # Try to remove each edge
        # We iterate from 0 to len(edges)-1
        # If we remove an edge, the list shrinks.
        # So we iterate carefully.
        i = 0
        while i < len(edges):
            if time.time() > deadline:
                break
                
            e = edges[i]
            # Try removing e at index i
            test_edges = edges[:i] + edges[i+1:]
            
            if is_2_colorable(n, test_edges):
                # Must keep e
                i += 1
            else:
                # Can remove e
                edges = test_edges
                # Do NOT increment i, because the next edge shifted into position i
                # needs to be tested.
                
        if len(edges) < best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return {'n': n, 'edges': best_edges}
