"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_dist(adj, n):
    """Compute all-pairs shortest paths using BFS from each node."""
    dist = [[-1]*n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = [s]
        for node in queue:
            for nb in adj[node]:
                if dist[s][nb] == -1:
                    dist[s][nb] = dist[s][node] + 1
                    queue.append(nb)
    return dist

def _check_validity(added_edges, n=24):
    """Check triangle-free, connected, diameter <= 4."""
    base = P['edges']
    all_edges = base + added_edges
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    # Check connected & diameter
    dist = _bfs_dist(adj, n)
    max_d = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False  # not connected
            max_d = max(max_d, dist[i][j])
    return max_d <= 4

def construct():
    deadline = time.time() + 80
    n = 24
    # Candidate edges: all (even, odd) pairs not already in path
    base_set = set()
    for u, v in P['edges']:
        base_set.add((min(u,v), max(u,v)))
    
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    candidates = []
    for e in even_verts:
        for o in odd_verts:
            key = (min(e,o), max(e,o))
            if key not in base_set:
                candidates.append([e, o])
    
    # Greedy: start with hub solution, try to remove edges
    hub_edges = [[12, o] for o in [1,3,5,7,9,15,17,19,21,23]]
    if _check_validity(hub_edges, n):
        current = list(hub_edges)
    else:
        current = []
    
    # Try to remove each edge and see if still valid
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:i] + current[i+1:]
            if _check_validity(trial, n):
                current = trial
                improved = True
                break
        if time.time() > deadline:
            break
    
    # Try to add edges that allow removing more (swap)
    if time.time() < deadline - 10:
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for cand in candidates:
                if time.time() > deadline:
                    break
                if cand in current:
                    continue
                # Try replacing current[i] with cand
                trial = current[:i] + [cand] + current[i+1:]
                if _check_validity(trial, n):
                    # Check if we can then remove another edge
                    current = trial
                    improved2 = True
                    while improved2:
                        improved2 = False
                        for j in range(len(current)):
                            if time.time() > deadline:
                                break
                            trial2 = current[:j] + current[j+1:]
                            if _check_validity(trial2, n):
                                current = trial2
                                improved2 = True
                                break
                        if time.time() > deadline:
                            break
                    improved = True
                    break
    
    # Convert to proper int lists
    result = []
    for edge in current:
        result.append([int(edge[0]), int(edge[1])])
    return result
# EVOLVE-BLOCK-END
