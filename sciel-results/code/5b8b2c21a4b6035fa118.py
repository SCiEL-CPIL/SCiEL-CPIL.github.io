import time
import random
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _greedy_decompose(edges):
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick edge incident to vertex with highest remaining degree
        best_edge = None
        best_degree_sum = -1
        for e in remaining:
            u, v = e
            deg_sum = len(adj[u]) + len(adj[v])
            if deg_sum > best_degree_sum:
                best_degree_sum = deg_sum
                best_edge = e
        
        u, v = best_edge
        remaining.remove((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        # Extend from u
        curr = u
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def _try_merge(paths, adj, edges_set):
    """
    Try to merge two paths into one by finding a path in the graph 
    connecting an endpoint of path1 to an endpoint of path2, 
    such that the new path is simple and uses only edges in the graph.
    Note: This is a simplified merge that just checks if endpoints are adjacent 
    or if there's a short path. For a robust merge, we'd need to check 
    if the union of edges forms a simple path.
    """
    # For now, just check if any two paths can be merged by direct edge
    # or by a 2-edge path. This is a heuristic.
    n_paths = len(paths)
    for i in range(n_paths):
        for j in range(i+1, n_paths):
            p1 = paths[i]
            p2 = paths[j]
            # Try all 4 combinations of endpoints
            for e1 in [p1[0], p1[-1]]:
                for e2 in [p2[0], p2[-1]]:
                    if (min(e1, e2), max(e1, e2)) in edges_set:
                        # Direct merge
                        if e1 == p1[0]:
                            p1 = p1[::-1]
                        if e2 == p2[0]:
                            p2 = p2[::-1]
                        # Now p1 ends at e1, p2 starts at e2
                        if (min(e1, e2), max(e1, e2)) in edges_set:
                            new_path = p1 + p2[1:]
                            # Check simplicity
                            if len(set(new_path)) == len(new_path):
                                return i, j, new_path
    return None

def decompose(edges):
    if not edges:
        return []
    
    # Start with greedy
    paths = _greedy_decompose(edges)
    
    # Build adjacency and edge set for merging
    adj = defaultdict(set)
    edges_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add((min(u, v), max(u, v)))
    
    # Try to improve by merging
    improved = True
    while improved:
        improved = False
        merge_result = _try_merge(paths, adj, edges_set)
        if merge_result:
            i, j, new_path = merge_result
            # Remove i and j, add new_path
            if i > j:
                i, j = j, i
            paths.pop(j)
            paths.pop(i)
            paths.append(new_path)
            improved = True
    
    return paths

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             pass
             
    return results
