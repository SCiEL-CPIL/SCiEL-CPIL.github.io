"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 80.0
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
    
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    candidate_edges = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_edges:
                candidate_edges.append((u, v))
    
    def get_diameter(added):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
        max_d = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = [s]
            for node in q:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_d = max(max_d, dist[nb])
                        q.append(nb)
            if -1 in dist:
                return float('inf')
        return max_d
    
    def is_valid(added):
        # Check all added edges are cross-partition (ensures triangle-free)
        for u, v in added:
            if (u % 2) == (v % 2):
                return False
        return get_diameter(added) <= 4
    
    # Greedy: iteratively add the edge that most reduces diameter
    current_added = []
    remaining = list(range(len(candidate_edges)))
    
    # First, find a feasible solution (any k)
    # Then try to minimize k by checking if we can remove edges
    
    # Start with hub strategy as initial solution
    hub = 12
    initial = []
    for v in odd_verts:
        if (hub, v) not in base_edges:
            initial.append((hub, v))
    
    if is_valid(initial):
        current_added = initial
    else:
        # Fallback: add all candidates
        current_added = list(candidate_edges)
    
    # Local search: try to remove edges while maintaining validity
    improved = True
    while improved:
        improved = False
        for i in range(len(current_added)):
            if time.time() > deadline:
                break
            trial = current_added[:i] + current_added[i+1:]
            if is_valid(trial):
                current_added = trial
                improved = True
                break  # restart scan since list changed
    
    # Also try swapping: remove one, add another
    for i in range(len(current_added)):
        if time.time() > deadline:
            break
        for j in remaining:
            if time.time() > deadline:
                break
            e = candidate_edges[j]
            if e in current_added:
                continue
            trial = current_added[:i] + [e] + current_added[i+1:]
            if is_valid(trial):
                # Check if this reduces diameter or same count
                current_added = trial
                improved = True
                break
        if improved:
            break
    
    return [[int(u), int(v)] for u, v in current_added]
# EVOLVE-BLOCK-END
