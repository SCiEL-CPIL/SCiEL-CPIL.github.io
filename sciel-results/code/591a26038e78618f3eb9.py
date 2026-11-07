from random import Random
import time

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
    start_time = time.time()
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    
    tree_adjs = []
    for idx in range(20):
        k = idx + 1
        adj = [[] for _ in range(k)]
        for (a, b) in trees[idx]:
            adj[a].append(b)
            adj[b].append(a)
        tree_adjs.append(adj)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 7919 + 611)
        
        order = list(range(19, -1, -1))
        
        available = set()
        for u in range(20):
            for v in range(u + 1, 20):
                available.add((u, v))
        
        result = [None] * 20
        success = True
        
        for idx in order:
            if time.time() > deadline:
                success = False
                break
            
            k = idx + 1
            if k == 1:
                result[idx] = [0]
                continue
            if k == 2:
                if not available:
                    success = False
                    break
                edge = list(available)[rng.randint(0, len(available) - 1)]
                result[idx] = [edge[0], edge[1]]
                available.discard(edge)
                continue
            
            adj = tree_adjs[idx]
            tree_edges = trees[idx]
            
            emb = None
            max_tries = 5
            for _ in range(max_tries):
                if time.time() > deadline:
                    break
                
                root = rng.randint(0, k - 1)
                root_host = rng.randint(0, 19)
                mapping = [None] * k
                mapping[root] = root_host
                used_edges = set()
                
                found = [False]
                nodes = [0]
                max_nodes = 10000
                
                def dfs(node, host):
                    if time.time() > deadline or found[0]:
                        return
                    nodes[0] += 1
                    if nodes[0] > max_nodes:
                        return
                    for nb in adj[node]:
                        if mapping[nb] is not None:
                            h = mapping[nb]
                            e = (min(host, h), max(host, h))
                            if e in used_edges or e not in available:
                                return
                        else:
                            for h in range(20):
                                if found[0]:
                                    return
                                if h == host:
                                    continue
                                if h in mapping:
                                    continue
                                e = (min(host, h), max(host, h))
                                if e in used_edges or e not in available:
                                    continue
                                mapping[nb] = h
                                used_edges.add(e)
                                dfs(nb, h)
                                used_edges.discard(e)
                                mapping[nb] = None
                                if found[0]:
                                    return
                            return
                    if all(m is not None for m in mapping):
                        found[0] = True
                        return
                
                dfs(root, root_host)
                if found[0]:
                    emb = mapping[:]
                    break
            
            if emb is None:
                success = False
                break
            
            result[idx] = emb
            for (a, b) in tree_edges:
                ha, hb = emb[a], emb[b]
                e = (min(ha, hb), max(ha, hb))
                available.discard(e)
        
        if success:
            valid = True
            used_edges = set()
            for idx in range(20):
                emb = result[idx]
                if emb is None:
                    valid = False
                    break
                if len(emb) != idx + 1:
                    valid = False
                    break
                if len(set(emb)) != idx + 1:
                    valid = False
                    break
                for v in emb:
                    if not isinstance(v, int) or v < 0 or v > 19:
                        valid = False
                        break
                if not valid:
                    break
                for (a, b) in trees[idx]:
                    ha, hb = emb[a], emb[b]
                    e = (min(ha, hb), max(ha, hb))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
            
            if valid:
                return [list(r) for r in result]
    
    return [None] * 20
