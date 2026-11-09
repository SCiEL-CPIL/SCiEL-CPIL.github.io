"""Packing trees T1..T20 into K20."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    N = 20
    # Build K20 edge set
    all_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.add((i, j))
    # Sort trees by size descending for harder-first packing
    order = sorted(range(20), key=lambda k: len(trees[k]), reverse=True)
    
    def solve(order, idx, used_edges, assignments):
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        k = order[idx]
        edges = trees[k]
        if not edges:
            # T1: no edges, assign vertex 0 to any free vertex
            assignments[k] = [0]
            return solve(order, idx + 1, used_edges, assignments)
        
        # Backtracking: assign tree vertices to host vertices
        # Tree vertices: 0..k-1
        # We'll do a DFS with constraint propagation
        host = [-1] * k
        # Build tree adjacency
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a root: pick vertex 0
        # Order vertices by BFS for better constraint propagation
        visited = [False] * k
        queue = [0]
        visited[0] = True
        bfs_order = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    bfs_order.append(v)
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(bfs_order):
                # All assigned, check edges are free
                for u, v in edges:
                    e = (min(host[u], host[v]), max(host[u], host[v]))
                    if e in used_edges:
                        return False
                # Record assignment
                assignments[k] = list(host)
                used_edges_copy = set(used_edges)
                for u, v in edges:
                    used_edges_copy.add((min(host[u], host[v]), max(host[u], host[v])))
                if solve(order, idx + 1, used_edges_copy, assignments):
                    return True
                return False
            
            tv = bfs_order[pos]
            for hv in range(N):
                if hv in host:  # already used by another tree vertex
                    continue
                # Check: for each already-assigned neighbor, the edge must be free
                ok = True
                for nb in adj[tv]:
                    if host[nb] != -1:
                        e = (min(hv, host[nb]), max(hv, host[nb]))
                        if e in used_edges:
                            ok = False
                            break
                if not ok:
                    continue
                host[tv] = hv
                if backtrack(pos + 1):
                    return True
                host[tv] = -1
            return False
        
        # Try different starting vertices for root
        for start in range(N):
            host[0] = start
            if backtrack(1):
                return True
            host[0] = -1
        return False
    
    assignments = [None] * 20
    used_edges = set()
    
    if solve(order, 0, used_edges, assignments):
        result = []
        for k in range(20):
            if assignments[k] is None:
                result.append(None)
            else:
                result.append(list(assignments[k]))
        return result
    else:
        # Fallback: at least embed T20
        return [None] * 19 + [list(range(20))]
