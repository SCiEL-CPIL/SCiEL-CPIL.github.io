"""Packing trees T1..T20 into K20 using greedy edge-disjoint embedding."""
import time
import random
from random import Random
from collections import deque


def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _embed_tree(tree_edges, n=20, used_edges=None, rng=None, deadline=None):
    """
    Try to embed tree_edges (list of (u,v) pairs) into K_n.
    Returns a mapping dict from tree vertex -> host vertex,
    or None if fails.
    """
    if not tree_edges:
        return {}
    
    # Collect all tree vertices
    tree_verts = set()
    for u, v in tree_edges:
        tree_verts.add(u)
        tree_verts.add(v)
    tree_verts = sorted(tree_verts)
    k = len(tree_verts)
    
    # Build adjacency for the tree
    adj = {v: [] for v in tree_verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a root (a leaf or any vertex)
    root = tree_verts[0]
    
    # BFS to get an ordering
    order = [root]
    parent = {root: -1}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for w in adj[u]:
            if w not in parent:
                parent[w] = u
                queue.append(w)
                order.append(w)
    
    # Try multiple random attempts with backtracking
    max_attempts = 500
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline:
            return None
        
        mapping = {}
        host_used = set()
        success = True
        
        # Assign root
        available = [h for h in range(n) if h not in host_used]
        if not available:
            return None
        host_root = rng.choice(available)
        mapping[root] = host_root
        host_used.add(host_root)
        
        # Assign remaining vertices in BFS order with backtracking
        def backtrack(idx):
            if idx == len(order):
                return True
            if deadline and time.time() > deadline:
                return False
            
            v = order[idx]
            p = parent[v]
            host_p = mapping[p]
            
            # Need a host h such that (host_p, h) not in used_edges
            # and h not in host_used
            candidates = [h for h in range(n) if h not in host_used]
            rng.shuffle(candidates)
            
            for h in candidates:
                e = (min(host_p, h), max(host_p, h))
                if e not in used_edges:
                    mapping[v] = h
                    host_used.add(h)
                    if backtrack(idx + 1):
                        return True
                    del mapping[v]
                    host_used.remove(h)
            
            return False
        
        if backtrack(1):
            return mapping
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()
    result = [None] * 20
    
    # Embed trees in order of decreasing size (more edges first)
    # But we must return in original order
    order = list(range(20))
    # Sort by number of edges descending
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    rng = Random(42)
    
    for idx in order:
        if time.time() > deadline - 5.0:
            break
        k = idx + 1
        tree_edges = trees[idx]
        
        if not tree_edges:
            # Tree with 1 vertex (k=1)
            result[idx] = [0]
            continue
        
        # Try to embed
        mapping = _embed_tree(tree_edges, n, used_edges, rng, deadline)
        if mapping is not None:
            # Build the result list: position i is image of vertex i
            # Tree vertices are 0..k-1
            result_list = []
            for v in range(k):
                result_list.append(int(mapping[v]))
            result[idx] = result_list
            
            # Add edges to used_edges
            for u, v in tree_edges:
                hu, hv = mapping[u], mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
    
    # Ensure types are correct
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
# EVOLVE-BLOCK-END
