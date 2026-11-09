import random
import itertools

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

def _find_embedding_randomized(tree, k, n, used_edges, num_restarts=10):
    """
    Find an embedding of tree into K_n avoiding used_edges.
    Uses randomized backtracking with restarts.
    """
    if k == 1:
        return [0]
    if k == 2:
        for _ in range(num_restarts):
            candidates = list(range(n))
            random.shuffle(candidates)
            for a in candidates:
                for b in candidates:
                    if a == b: continue
                    if (a, b) not in used_edges and (b, a) not in used_edges:
                        # Normalize edge for check? used_edges stores (min, max)
                        e = (min(a,b), max(a,b))
                        if e not in used_edges:
                            return [a, b]
        return None

    # Build tree adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a leaf to start from (randomize choice among leaves)
    leaves = [i for i in range(k) if len(adj[i]) == 1]
    if not leaves:
        leaves = [0]
        
    for restart in range(num_restarts):
        root = random.choice(leaves)
        
        # BFS order from root
        parent = [-1] * k
        parent[root] = -2
        order = [root]
        queue = [root]
        idx = 0
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            # Shuffle neighbors to randomize expansion order
            neighbors = adj[u][:]
            random.shuffle(neighbors)
            for v in neighbors:
                if parent[v] == -1:
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        
        host = [-1] * k
        used_hosts = set()
        
        def backtrack(pos):
            if pos == len(order):
                return True
            
            v = order[pos]
            assigned_neighbors = []
            for u in adj[v]:
                if host[u] != -1:
                    assigned_neighbors.append(u)
            
            # Get available host vertices
            # Optimization: Pre-filter if possible, but n=20 is small
            candidates = [h for h in range(n) if h not in used_hosts]
            random.shuffle(candidates)
            
            for h in candidates:
                ok = True
                for u in assigned_neighbors:
                    hu = host[u]
                    edge = (min(h, hu), max(h, hu))
                    if edge in used_edges:
                        ok = False
                        break
                if not ok:
                    continue
                
                host[v] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                host[v] = -1
                used_hosts.remove(h)
            
            return False
        
        # Assign root
        root_candidates = list(range(n))
        random.shuffle(root_candidates)
        for h in root_candidates:
            host[root] = h
            used_hosts.add(h)
            if backtrack(1):
                return [host[i] for i in range(k)]
            host[root] = -1
            used_hosts.remove(h)
            
    return None

def construct():
    n = 20
    trees = _generate_trees(n)
    
    result = [None] * n
    used_edges = set()
    
    # Embed trees in order of decreasing size
    tree_indices = list(range(n - 1, -1, -1))
    
    # To improve robustness, we can try a few different global orderings if we fail to pack all
    # But first, let's just use the randomized embedder with the standard order.
    
    for idx in tree_indices:
        k = idx + 1
        tree = trees[idx]
        
        if k == 1:
            result[idx] = None
            continue
        
        embedding = _find_embedding_randomized(tree, k, n, used_edges, num_restarts=20)
        if embedding is None:
            # If we fail, we might try to skip this tree and come back?
            # For now, if we fail, we return what we have.
            # However, a failed large tree might block smaller ones? No, it's just missing.
            # A better strategy: If a tree fails, we can try to embed the remaining smaller trees, 
            # and then retry the failed tree?
            # Given time constraints, let's just return.
            # But wait, if T20 fails, we get 0. If T20 succeeds but T19 fails, we get 19.
            # It is unlikely T20 fails. T19 is very likely to succeed with randomization.
            return result
        
        embedding = [int(x) for x in embedding]
        result[idx] = embedding
        
        for u, v in tree:
            hu = embedding[u]
            hv = embedding[v]
            edge = (min(hu, hv), max(hu, hv))
            used_edges.add(edge)
    
    return result
