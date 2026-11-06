import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    # Try n=21 first, then 20
    for n in [21, 20]:
        if time.time() - start_time > timeout:
            break
            
        # Random restarts
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Initialize random graph
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
                        
            # Local search
            steps = 0
            max_steps = 1000
            while steps < max_steps:
                if time.time() - start_time > timeout:
                    break
                
                # Find a bad subset
                bad_subset = find_bad_subset(adj, n)
                if bad_subset is None:
                    # Check if valid
                    if is_valid(adj, n):
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        # This shouldn't happen if find_bad_subset is correct
                        # But just in case, break
                        break
                
                # Flip an edge in bad_subset to break regularity
                flipped = False
                # Shuffle edges in subset to randomize
                verts = list(bad_subset)
                random.shuffle(verts)
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u, v = verts[i], verts[j]
                        # Try flipping
                        adj[u][v] = not adj[u][v]
                        adj[v][u] = adj[u][v]
                        
                        # Check if this subset is now non-regular
                        if not is_regular_subset(verts, adj):
                            flipped = True
                            break
                        else:
                            # Revert
                            adj[u][v] = not adj[u][v]
                            adj[v][u] = adj[u][v]
                    
                if not flipped:
                    # No edge flip broke this subset?
                    # This can happen if the subset is "robust"?
                    # Actually, any regular graph on k>=2 vertices can be made non-regular by flipping one edge?
                    # If k=6, and it's regular, flipping an edge changes degrees of 2 vertices.
                    # If all degrees were d, they become d+1 and d-1. So it's non-regular.
                    # So flipped should always be True.
                    pass
                
                steps += 1

    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

def is_regular_subset(verts, adj):
    k = len(verts)
    if k < 2:
        return True
    degrees = [0] * k
    for i in range(k):
        for j in range(i+1, k):
            if adj[verts[i]][verts[j]]:
                degrees[i] += 1
                degrees[j] += 1
    return all(d == degrees[0] for d in degrees)

def find_bad_subset(adj, n):
    # Check sizes 6 to n
    for k in range(6, n+1):
        for comb in itertools.combinations(range(n), k):
            if is_regular_subset(comb, adj):
                return comb
    return None

def is_valid(adj, n):
    return find_bad_subset(adj, n) is None
