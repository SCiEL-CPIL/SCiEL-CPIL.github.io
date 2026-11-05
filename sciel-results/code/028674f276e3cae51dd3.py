import time
import random
import sys
from collections import defaultdict, deque
import numpy as np

_T0 = time.time()
_DEADLINE = 148.0
_MARGIN = 5.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set(tuple(sorted(e)) for e in zip(order, order[1:]))
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _shortest_paths(n, edges):
    # Return dist matrix
    dist = np.full((n, n), np.inf)
    for i in range(n):
        dist[i][i] = 0
    for u, v in edges:
        dist[u][v] = 1
        dist[v][u] = 1
    
    # Floyd Warshall
    for k in range(n):
        for i in range(n):
            if dist[i][k] == np.inf: continue
            for j in range(n):
                if dist[k][j] == np.inf: continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def _min_t_join(n, edges, odd_vertices):
    # Returns set of edges to remove
    if not odd_vertices:
        return set()
    
    T = odd_vertices
    t_size = len(T)
    if t_size == 0:
        return set()
    
    # Get distances between odd vertices
    dist = _shortest_paths(n, edges)
    
    # Build cost matrix for T
    # We need paths, not just costs.
    # But first, find matching.
    
    # If t_size is small, use DP
    # Map T to indices 0..t_size-1
    t_to_idx = {v: i for i, v in enumerate(T)}
    
    if t_size <= 20:
        # DP over subsets
        # dp[mask] = min cost to match vertices in mask
        # mask is bitmask of size t_size
        size = 1 << t_size
        dp = [np.inf] * size
        dp[0] = 0
        parent = [None] * size
        
        for mask in range(size):
            if dp[mask] == np.inf: continue
            # Find first unmatched vertex in mask
            i = -1
            for k in range(t_size):
                if not (mask & (1 << k)):
                    i = k
                    break
            if i == -1: continue
            
            for j in range(i + 1, t_size):
                if not (mask & (1 << j)):
                    new_mask = mask | (1 << i) | (1 << j)
                    cost = dist[T[i]][T[j]]
                    if dp[new_mask] > dp[mask] + cost:
                        dp[new_mask] = dp[mask] + cost
                        parent[new_mask] = (mask, i, j)
        
        # Reconstruct matching
        edges_to_remove = set()
        mask = (1 << t_size) - 1
        while mask != 0:
            prev_mask, i, j = parent[mask]
            u, v = T[i], T[j]
            # Add shortest path between u and v to edges_to_remove
            # We need the actual path, not just distance.
            # Re-run Dijkstra or use parent pointers from FW? FW doesn't store parents easily.
            # Let's store parent pointers in FW or just run BFS for each pair in the matching.
            mask = prev_mask
        
        # To get the actual edges, we need the paths.
        # We'll run BFS for each matched pair (u, v) to get the path.
        # But we need the list of pairs first.
        pairs = []
        mask = (1 << t_size) - 1
        while mask != 0:
            prev_mask, i, j = parent[mask]
            pairs.append((T[i], T[j]))
            mask = prev_mask
        
        # Find paths
        for u, v in pairs:
            # BFS to find path
            prev = [-1] * n
            prev[u] = u
            q = deque([u])
            while q:
                curr = q.popleft()
                if curr == v: break
                # neighbors
                # We need adjacency list
                # Let's build it once outside or pass it
                # For now, use dist matrix? No, need path.
                # We'll build adj list in the caller and pass it, or rebuild here.
                # Rebuilding is cheap for n=30.
                pass # Placeholder
        
        # Actually, let's just return the pairs and let the caller find paths.
        # Or better, implement a helper to get path.
        
        return pairs
    else:
        # Greedy matching
        pairs = []
        remaining = list(T)
        while remaining:
            u = remaining.pop(0)
            # Find closest v in remaining
            min_d = np.inf
            best_v = -1
            for v in remaining:
                if dist[u][v] < min_d:
                    min_d = dist[u][v]
                    best_v = v
            if best_v != -1:
                pairs.append((u, best_v))
                remaining.remove(best_v)
        return pairs

def _get_path(n, adj, u, v):
    prev = [-1] * n
    prev[u] = u
    q = deque([u])
    while q:
        curr = q.popleft()
        if curr == v: break
        for nei in adj[curr]:
            if prev[nei] == -1:
                prev[nei] = curr
                q.append(nei)
    
    path = []
    curr = v
    while curr != u:
        path.append(curr)
        curr = prev[curr]
    path.append(u)
    path.reverse()
    
    # Convert path to set of edges
    edges_in_path = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        edges_in_path.add((min(a, b), max(a, b)))
    return edges_in_path

def _decompose_eulerian(adj):
    # adj is dict of sets
    adj = {u: set(vs) for u, vs in adj.items()}
    cycles = []
    
    while any(len(vs) > 0 for vs in adj.values()):
        # Find start vertex with degree > 0
        start = None
        for u in adj:
            if len(adj[u]) > 0:
                start = u
                break
        if start is None: break
        
        # Find a cycle starting from start
        # Use DFS
        parent = {start: None}
        stack = [(start, iter(sorted(adj[start])))]
        order = [start]
        found_cycle = None
        
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v == parent.get(u):
                    continue
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    stack.append((v, iter(sorted(adj[v]))))
                    advanced = True
                    break
                else:
                    # Back edge
                    idx = order.index(v)
                    cyc = order[idx:]
                    if len(cyc) >= 3:
                        found_cycle = cyc
                        break
            if found_cycle:
                break
            if not advanced:
                stack.pop()
        
        if found_cycle is None:
            # Should not happen if Eulerian, but safety
            break
            
        # Remove cycle edges
        for i in range(len(found_cycle)):
            a = found_cycle[i]
            b = found_cycle[(i + 1) % len(found_cycle)]
            adj[a].discard(b)
            adj[b].discard(a)
        
        # Clean up empty sets
        adj = {u: vs for u, vs in adj.items() if vs}
        
        cycles.append(found_cycle)
        
    return cycles

def _decompose(edges):
    n = max((max(u, v) for u, v in edges), default=0) + 1
    present = set((min(u, v), max(u, v)) for u, v in edges)
    if not present:
        return []
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in present:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find odd vertices
    odd = [u for u in range(n) if len(adj[u]) % 2 == 1]
    
    # Find T-Join
    pairs = _min_t_join(n, present, odd)
    
    # Determine edges to remove
    edges_to_remove = set()
    for u, v in pairs:
        path_edges = _get_path(n, adj, u, v)
        edges_to_remove.update(path_edges)
    
    # Build Eulerian subgraph
    adj_euler = defaultdict(set)
    for u, v in present:
        if (u, v) not in edges_to_remove:
            adj_euler[u].add(v)
            adj_euler[v].add(u)
    
    # Decompose into cycles
    cycles = _decompose_eulerian(adj_euler)
    
    # Build pieces
    pieces = []
    used = set()
    
    for c in cycles:
        pieces.append(c)
        for i in range(len(c)):
            a = c[i]
            b = c[(i + 1) % len(c)]
            used.add((min(a, b), max(a, b)))
            
    # Singles
    for e in present:
        if e not in used:
            pieces.append([e[0], e[1]])
            used.add(e)
            
    return pieces

def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > _DEADLINE - _MARGIN:
            # Fill remaining
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                e2 = _instance(*spec2)
                results.append([[u, v] for u, v in e2])
            break
            
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
        
    return results
