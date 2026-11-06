"""Rewritten program: greedy maximal simple path decomposition with backtracking fallback."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges):
    if not edges:
        return []
    # Build adjacency with edge IDs
    n = max(max(e) for e in edges) + 1
    adj = [[] for _ in range(n)]
    edge_set = set()
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
        edge_set.add((u, v))
    
    used = [False] * len(edges)
    paths = []
    
    # Greedy: repeatedly pick an unused edge, extend to maximal simple path
    # Sort edges to have deterministic order; pick lowest unused edge first
    remaining = list(range(len(edges)))
    # We'll maintain a set of unused edge IDs for quick lookup
    unused_edges = set(remaining)
    
    while unused_edges:
        # Pick an unused edge to start a new path
        # Heuristic: pick edge where both endpoints have low unused-degree (to avoid stranding)
        # Or just pick the smallest unused edge ID for determinism
        start_eid = min(unused_edges)
        u, v = edges[start_eid]
        
        # Try to extend in both directions from this edge
        # Build a path: try to extend from u backwards and from v forwards
        # Start with [u, v]
        path = [u, v]
        used[start_eid] = True
        unused_edges.discard(start_eid)
        
        # Extend from u end (go to neighbors of u other than v)
        def extend_from(vertex, prev, path_list, unused_set, used_arr, adj_local, edge_ids):
            # Try to extend path_list by adding a neighbor of vertex (not prev) via an unused edge
            # Returns True if extended, False otherwise
            # Greedy: prefer neighbor with highest unused-degree
            best_neighbor = -1
            best_eid = -1
            best_degree = -1
            for (nb, eid) in adj_local[vertex]:
                if eid in unused_set and nb != prev:
                    # Check if nb is not already in path (simple path)
                    if nb not in path_list:
                        # Count unused edges from nb
                        deg = sum(1 for (nb2, eid2) in adj_local[nb] if eid2 in unused_set)
                        if deg > best_degree:
                            best_degree = deg
                            best_neighbor = nb
                            best_eid = eid
            if best_neighbor >= 0:
                path_list.insert(0, best_neighbor)
                used_arr[best_eid] = True
                unused_set.discard(best_eid)
                return True
            return False
        
        def extend_to(vertex, prev, path_list, unused_set, used_arr, adj_local):
            # Extend path_list by appending a neighbor of vertex (not prev) via an unused edge
            best_neighbor = -1
            best_eid = -1
            best_degree = -1
            for (nb, eid) in adj_local[vertex]:
                if eid in unused_set and nb != prev:
                    if nb not in path_list:
                        deg = sum(1 for (nb2, eid2) in adj_local[nb] if eid2 in unused_set)
                        if deg > best_degree:
                            best_degree = deg
                            best_neighbor = nb
                            best_eid = eid
            if best_neighbor >= 0:
                path_list.append(best_neighbor)
                used_arr[best_eid] = True
                unused_set.discard(best_eid)
                return True
            return False
        
        # Extend from the u end (prepend)
        changed = True
        while changed:
            changed = extend_from(path[0], path[1], path, unused_edges, used, adj)
        
        # Extend from the v end (append)
        changed = True
        while changed:
            changed = extend_to(path[-1], path[-2], path, unused_edges, used, adj)
        
        paths.append(path)
    
    return paths

def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for (n, p, seed) in specs:
        edges = _instance(n, p, seed)
        paths = _decompose(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
