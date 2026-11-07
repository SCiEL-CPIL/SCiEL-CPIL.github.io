import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v: u, v = v, u
        base_edges.add((u, v))
    
    # Triangle-free constraint on a bipartite path: only edges between opposite parity are allowed.
    # Vertices 0..23: even = {0,2,...,22}, odd = {1,3,...,23}
    evens = [i for i in range(n) if i % 2 == 0]
    odds = [i for i in range(n) if i % 2 == 1]
    
    # Candidate edges: all cross-parity pairs not already in base
    candidates = []
    for u in evens:
        for v in odds:
            pair = (min(u, v), max(u, v))
            if pair not in base_edges:
                candidates.append(pair)
    
    def compute_diameter(edge_set):
        adj = [[] for _ in range(n)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
                        q.append(nb)
        return max_dist
    
    def is_valid(edge_set):
        # Check triangle-free (bipartite: all edges cross parity)
        for u, v in edge_set:
            if (u % 2) == (v % 2):
                return False
        # Check connected and diameter <= 4
        diam = compute_diameter(edge_set)
        if diam == -1:  # disconnected
            return False
        return diam <= 4
    
    current_edges = set(base_edges)
    added = []
    
    # Greedy: at each step, find the edge that most reduces diameter
    # If multiple, prefer the one that reduces the number of pairs at max distance
    def score_edge(edge_set):
        """Return (diameter, num_max_pairs) for tie-breaking."""
        adj = [[] for _ in range(n)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = 0
        num_max_pairs = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
                        q.append(nb)
        
        # Count pairs at max_dist
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            for dst in range(src + 1, n):
                if dist[dst] == max_dist:
                    num_max_pairs += 1
        
        return (max_dist, num_max_pairs)
    
    deadline = time.time() + 85  # 5s safety margin
    
    while time.time() < deadline:
        diam = compute_diameter(current_edges)
        if diam <= 4:
            break
        
        best_score = (float('inf'), float('inf'))
        best_cand = None
        
        for cand in candidates:
            if cand in current_edges:
                continue
            
            new_edges = current_edges | {cand}
            new_score = score_edge(new_edges)
            
            if new_score < best_score:
                best_score = new_score
                best_cand = cand
        
        if best_cand is None:
            break
        
        current_edges.add(best_cand)
        added.append(list(best_cand))
    
    # Try to remove edges (local improvement)
    deadline2 = time.time() + 88
    changed = True
    while changed and time.time() < deadline2:
        changed = False
        for i in range(len(added) - 1, -1, -1):
            edge = tuple(added[i])
            if edge in current_edges:
                new_edges = current_edges - {edge}
                if is_valid(new_edges):
                    current_edges = new_edges
                    added.pop(i)
                    changed = True
    
    return added
