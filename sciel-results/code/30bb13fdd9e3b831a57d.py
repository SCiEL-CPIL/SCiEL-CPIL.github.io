"""Packing trees T1..T20 into K20 via backtracking with forward checking."""
import time
from random import Random

def _instance_trees(n=20, seed=611):
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

def _build_adjacency(edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v in adj:
        adj[v] = sorted(adj[v])
    return adj

def _embed_tree(tree_edges, n, used_edges_set, deadline, rng=None):
    if not tree_edges:
        return [0]

    k = len(tree_edges) + 1
    adj = _build_adjacency(tree_edges)
    root = max(range(k), key=lambda v: len(adj.get(v, [])))
    
    parent = [-1] * k
    children = [[] for _ in range(k)]
    order = []
    visited = {root}
    queue = [root]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for nb in adj.get(v, []):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = v
                children[v].append(nb)
                queue.append(nb)
    
    def try_root(root_host, rng_local):
        map_tree = [-1] * k
        used_hosts = {root_host}
        map_tree[root] = root_host
        order_rest = order[1:]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order_rest):
                return True
            v = order_rest[idx]
            p = parent[v]
            ph = map_tree[p]
            
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                if (ph, h) in used_edges_set or (h, ph) in used_edges_set:
                    continue
                candidates.append(h)
            
            if not candidates:
                return False
            
            valid = []
            for h in candidates:
                ok = True
                for ch in children[v]:
                    found = False
                    for h2 in range(n):
                        if h2 == h or h2 in used_hosts:
                            continue
                        if (h, h2) in used_edges_set or (h2, h) in used_edges_set:
                            continue
                        found = True
                        break
                    if not found:
                        ok = False
                        break
                if ok:
                    valid.append(h)
            
            if not valid:
                return False
            
            if rng_local:
                rng_local.shuffle(valid)
            
            for h in valid:
                map_tree[v] = h
                used_hosts.add(h)
                if backtrack(idx + 1):
                    return True
                used_hosts.remove(h)
                map_tree[v] = -1
            return False
        
        if backtrack(0):
            return map_tree[:]
        return None
    
    host_order = list(range(n))
    if rng:
        rng.shuffle(host_order)
    
    for rh in host_order:
        if time.time() > deadline:
            return None
        res = try_root(rh, rng)
        if res is not None:
            return res
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    trees = _instance_trees(n, 611)
    tree_indices = list(range(n))
    tree_indices.sort(key=lambda i: -i)
    used_edges_set = set()
    result = [None] * n
    rng = Random(42)
    
    for idx in tree_indices:
        if time.time() > deadline:
            break
        tree_edges = trees[idx]
        if not tree_edges:
            result[idx] = [0]
            continue
        emb = None
        emb = _embed_tree(tree_edges, n, used_edges_set, deadline, rng=None)
        if emb is None:
            for _ in range(30):
                if time.time() > deadline:
                    break
                emb = _embed_tree(tree_edges, n, used_edges_set, deadline, rng=rng)
                if emb is not None:
                    break
        if emb is None:
            continue
        for a, b in tree_edges:
            ha, hb = emb[a], emb[b]
            used_edges_set.add((ha, hb))
            used_edges_set.add((hb, ha))
        result[idx] = emb
    
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in result[i]])
    return final_result
