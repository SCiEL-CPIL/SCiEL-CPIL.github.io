import sys
sys.setrecursionlimit(10000)

def _generate_trees(n=20, seed=611):
    from random import Random
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

def _find_embedding(tree, k, n, used_edges):
    if k == 1:
        return [0]
    if k == 2:
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) not in used_edges:
                    return [a, b]
        return None

    adj = [[] for _ in range(k)]
    for u, v in tree:
        adj[u].append(v)
        adj[v].append(u)
    
    host = [-1] * k
    assigned = [False] * k
    used_hosts = set()

    def get_valid_hosts(v):
        valid = []
        for h in range(n):
            if h in used_hosts:
                continue
            ok = True
            for u in adj[v]:
                if assigned[u]:
                    hu = host[u]
                    edge = (min(h, hu), max(h, hu))
                    if edge in used_edges:
                        ok = False
                        break
            if ok:
                valid.append(h)
        return valid

    def backtrack(count):
        if count == k:
            return True
        
        # MRV Heuristic
        best_v = -1
        best_valids = None
        min_count = n + 1
        
        for v in range(k):
            if not assigned[v]:
                valids = get_valid_hosts(v)
                if len(valids) < min_count:
                    min_count = len(valids)
                    best_v = v
                    best_valids = valids
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break
        
        if best_v == -1:
            return False
            
        # Try valid hosts
        for h in best_valids:
            host[best_v] = h
            assigned[best_v] = True
            used_hosts.add(h)
            
            if backtrack(count + 1):
                return True
                
            host[best_v] = -1
            assigned[best_v] = False
            used_hosts.remove(h)
            
        return False

    # Start with a high-degree vertex to constrain the search space early
    start_v = max(range(k), key=lambda x: len(adj[x]))
    valids = get_valid_hosts(start_v)
    
    for h in valids:
        host[start_v] = h
        assigned[start_v] = True
        used_hosts.add(h)
        
        if backtrack(1):
            return host[:]
            
        host[start_v] = -1
        assigned[start_v] = False
        used_hosts.remove(h)
        
    return None

def construct():
    n = 20
    trees = _generate_trees(n)
    
    result = [None] * n
    used_edges = set()
    
    tree_indices = list(range(n - 1, -1, -1))
    
    for idx in tree_indices:
        k = idx + 1
        tree = trees[idx]
        
        if k == 1:
            result[idx] = None
            continue
        
        embedding = _find_embedding(tree, k, n, used_edges)
        if embedding is None:
            # If deterministic fails, it's likely impossible.
            # We return what we have.
            return result
        
        embedding = [int(x) for x in embedding]
        result[idx] = embedding
        
        for u, v in tree:
            hu = embedding[u]
            hv = embedding[v]
            edge = (min(hu, hv), max(hu, hv))
            used_edges.add(edge)
    
    return result
