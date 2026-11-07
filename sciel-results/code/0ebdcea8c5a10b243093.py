import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 seconds deadline to be safe
    
    n = P['n']
    edges = P['edges']
    
    # Use a set for fast lookup and removal
    # Edges are undirected, so store as (min(u,v), max(u,v))
    E_uncov = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        E_uncov.add((u, v))
        
    # Adjacency list for fast neighbor lookup
    # adj[u] is a set of neighbors v such that (u,v) is in E_uncov
    adj = [set() for _ in range(n)]
    for u, v in E_uncov:
        adj[u].add(v)
        adj[v].add(u)
        
    bicliques = []
    
    while E_uncov:
        if time.time() > deadline:
            break
            
        best_cov = 0
        best_L = None
        best_R = None
        
        # Candidates: L of size 1, 2, 3
        # Size 1: L={u}, R=N(u)
        for u in range(n):
            if not adj[u]:
                continue
            R = set(adj[u])
            cov = len(R)
            if cov > best_cov:
                best_cov = cov
                best_L = [u]
                best_R = list(R)
                
        # Size 2: L={u,v}, R=N(u)∩N(v)\{u,v}
        for u in range(n):
            if not adj[u]:
                continue
            for v in adj[u]:
                # Ensure u < v to avoid duplicates
                if u >= v:
                    continue
                R = adj[u] & adj[v]
                R.discard(u)
                R.discard(v)
                if not R:
                    continue
                cov = 2 * len(R)
                if cov > best_cov:
                    best_cov = cov
                    best_L = [u, v]
                    best_R = list(R)
                    
        # Size 3: L={u,v,w}, R=N(u)∩N(v)∩N(w)\{u,v,w}
        # This might be expensive if done naively.
        # Number of triples is C(30,3) = 4060.
        # For each triple, intersection of 3 sets of size ~10 is fast.
        for u in range(n):
            if len(adj[u]) < 2:
                continue
            for v in adj[u]:
                if u >= v:
                    continue
                if len(adj[v]) < 2:
                    continue
                common_uv = adj[u] & adj[v]
                common_uv.discard(u)
                common_uv.discard(v)
                for w in common_uv:
                    # u < v < w
                    if v >= w:
                        continue
                    R = common_uv & adj[w]
                    R.discard(u)
                    R.discard(v)
                    R.discard(w)
                    if not R:
                        continue
                    cov = 3 * len(R)
                    if cov > best_cov:
                        best_cov = cov
                        best_L = [u, v, w]
                        best_R = list(R)
                        
        if best_cov == 0:
            # Fallback: pick any edge
            u, v = next(iter(E_uncov))
            best_L = [u]
            best_R = [v]
            best_cov = 1
            
        bicliques.append([best_L, best_R])
        
        # Remove covered edges
        for u in best_L:
            for v in best_R:
                edge = (min(u, v), max(u, v))
                if edge in E_uncov:
                    E_uncov.remove(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
    # Convert to required format: list of [left, right]
    # Ensure types are correct
    result = []
    for L, R in bicliques:
        result.append([[int(x) for x in L], [int(x) for x in R]])
        
    return result
