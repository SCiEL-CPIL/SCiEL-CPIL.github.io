import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    deadline = time.time() + 80
    
    # Precompute base adjacency as list of lists
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify: 2 edges solution (hubs at 7 and 16, both even, connected to odd vertex 1)
    # This should give diameter <= 4 and be triangle-free
    solution = [[7, 1], [16, 1]]
    
    # Fast verification
    if _verify_fast(solution, n, adj):
        return solution
    
    # Fallback: search for valid 2-edge solution
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    # Try all pairs of (even, odd) edges
    for e1 in even_verts:
        if time.time() > deadline:
            break
        for o1 in odd_verts:
            if abs(e1 - o1) <= 1:  # skip if already connected
                continue
            for e2 in even_verts:
                if time.time() > deadline:
                    break
                for o2 in odd_verts:
                    if abs(e2 - o2) <= 1:
                        continue
                    # Check not duplicate
                    if (e1 == e2 and o1 == o2) or (e1 == o2 and o1 == e2):
                        continue
                    cand = [[e1, o1], [e2, o2]]
                    if _verify_fast(cand, n, adj):
                        return cand
    
    # Fallback: 7-edge solution
    return [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]

def _verify_fast(added_edges, n, adj):
    """Fast verification: build augmented adjacency, check triangle-free and diameter <= 4."""
    # Build augmented adjacency
    aug = [list(a) for a in adj]
    for u, v in added_edges:
        aug[u].append(v)
        aug[v].append(u)
    
    # Check triangle-free: for each edge (u,v), check no common neighbor
    # Actually, since we only add even-odd edges to a path (bipartite), it's automatically triangle-free
    # But let's verify for safety
    # For each vertex, check its neighbors don't share edges
    for i in range(n):
        ni = aug[i]
        for j in range(len(ni)):
            for k in range(j+1, len(ni)):
                if ni[k] in aug[ni[j]]:
                    return False
    
    # Check diameter <= 4 via BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= 4:
                continue
            du = dist[u] + 1
            for v in aug[u]:
                if dist[v] == -1:
                    dist[v] = du
                    queue.append(v)
        for d in dist:
            if d == -1 or d > 4:
                return False
    
    return True
