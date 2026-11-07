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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    n = 20
    
    # Try to find a valid packing using a simple heuristic
    # Process trees in order of increasing k
    used_edges = set()
    result = [None] * n
    
    for idx in range(n):
        k = idx + 1
        if k == 1:
            result[idx] = None
            continue
        
        tree_edges = trees[idx]
        if not tree_edges:
            result[idx] = None
            continue
        
        # Try to embed this tree
        # Use a simple greedy approach
        embedded = False
        
        # Try identity first
        valid = True
        for (u, v) in tree_edges:
            edge = (min(u, v), max(u, v))
            if edge in used_edges:
                valid = False
                break
        
        if valid:
            for (u, v) in tree_edges:
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
            result[idx] = list(range(k))
            embedded = True
        else:
            # Try a few random permutations
            rng = Random(idx * 1000 + 42)
            for _ in range(100):
                perm = list(range(n))
                rng.shuffle(perm)
                perm = perm[:k]
                valid = True
                for (u, v) in tree_edges:
                    mu = perm[u]
                    mv = perm[v]
                    edge = (min(mu, mv), max(mu, mv))
                    if edge in used_edges:
                        valid = False
                        break
                if valid:
                    for (u, v) in tree_edges:
                        mu = perm[u]
                        mv = perm[v]
                        edge = (min(mu, mv), max(mu, mv))
                        used_edges.add(edge)
                    result[idx] = list(perm)
                    embedded = True
                    break
        
        if not embedded:
            return [None] * n
    
    return result
