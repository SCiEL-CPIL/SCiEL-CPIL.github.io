import random

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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

def construct():
    trees = instance_trees()
    n = 20
    
    def try_pack(seed_val):
        rng = random.Random(seed_val)
        used_edges = set()
        results = [None] * n
        
        def embed_tree(k, edges):
            if k == 1:
                return [0]
            if k == 2:
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) not in used_edges:
                            return [u, v]
                return None
            
            deg = [0] * k
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            vertices = list(range(k))
            vertices.sort(key=lambda x: deg[x], reverse=True)
            
            mapping = [-1] * k
            mapping[vertices[0]] = 0
            
            def backtrack(idx):
                if idx == k:
                    return True
                
                v = vertices[idx]
                hosts = list(range(n))
                rng.shuffle(hosts)
                
                for h in hosts:
                    if h in mapping:
                        continue
                    
                    valid = True
                    for u, w in edges:
                        if u == v:
                            if mapping[w] != -1:
                                a, b = sorted((h, mapping[w]))
                                if (a, b) in used_edges:
                                    valid = False
                                    break
                        elif w == v:
                            if mapping[u] != -1:
                                a, b = sorted((h, mapping[u]))
                                if (a, b) in used_edges:
                                    valid = False
                                    break
                            
                    if valid:
                        mapping[v] = h
                        if backtrack(idx + 1):
                            return True
                        mapping[v] = -1
                        
                return False

            if backtrack(1):
                return mapping
            return None

        for k in range(20, 0, -1):
            edges = trees[k-1]
            res = embed_tree(k, edges)
            if res:
                results[k-1] = res
                for u, v in edges:
                    a, b = sorted((res[u], res[v]))
                    used_edges.add((a, b))
            else:
                # If we fail, we can't easily recover without backtracking over previous trees
                # So we just return what we have so far?
                # Or we signal failure to try a different seed
                return None, results, k
        
        return True, results, 0

    # Try a few seeds
    for s in range(100):
        success, res, fail_k = try_pack(s)
        if success:
            return res
        else:
            # If we failed, we might still have a partial result
            # But a partial result from a failed seed might be worse than a successful one
            # We want the best score.
            # Let's keep the best partial result seen so far
            # But since we return None for failed trees, the score is sum of edges of successful trees
            pass

    # If all seeds fail to pack all, we should return the best partial result
    # The loop above discards partial results if success is False.
    # Let's modify to keep track of the best score.
    
    best_score = -1
    best_res = [None] * 20
    
    for s in range(100):
        success, res, fail_k = try_pack(s)
        # Calculate score
        score = 0
        for k in range(1, 21):
            if res[k-1] is not None:
                # Number of edges in T_k is k-1
                score += k - 1
        if score > best_score:
            best_score = score
            best_res = res
        if success:
            return res
            
    return best_res
