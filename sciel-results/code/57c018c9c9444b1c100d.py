import time
from random import Random
import sys

sys.setrecursionlimit(10000)

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
    deadline = start_time + 140.0
    
    trees = instance_trees()
    n = 20
    
    # Pack in descending order
    # We will try multiple strategies if the first one fails
    # Strategy 1: Deterministic MRV
    # Strategy 2: Randomized MRV
    
    def try_pack(strategy_seed=None):
        used_edges = set()
        embeddings = [None] * n
        rng = Random(strategy_seed) if strategy_seed is not None else None
        
        def pack_tree(tree_idx):
            k = tree_idx + 1
            if k == 1:
                # T1: Single vertex. Map to 0.
                return [0]
            if k == 2:
                # T2: Single edge. Find any unused edge.
                # To be consistent with "distinct host vertices", we just pick u, v.
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges:
                            used_edges.add((u, v))
                            return [u, v]
                return None

            tree_edges = trees[tree_idx]
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            mapping = {}
            
            def backtrack(placed_count):
                if time.time() > deadline:
                    return False
                
                if placed_count == k:
                    return True
                
                candidates = []
                for v in range(k):
                    if v not in mapping:
                        placed_neighbors = sum(1 for u in adj[v] if u in mapping)
                        candidates.append((placed_neighbors, len(adj[v]), v))
                
                if not candidates:
                    return False
                
                candidates.sort(key=lambda x: (-x[0], -x[1]))
                
                # Randomize tie-breaking if rng is provided
                if rng:
                    # Group by score, then shuffle within groups
                    # Simple approach: shuffle candidates that have the same score
                    # For simplicity, just shuffle the whole list? No, that breaks MRV.
                    # We can shuffle the candidates with the same (placed_neighbors, degree)
                    # Let's just pick the first one for now, or randomize the valid_hosts.
                    pass
                
                _, _, v = candidates[0]
                
                used_hosts = set(mapping.values())
                valid_hosts = []
                for h in range(n):
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in adj[v]:
                        if u in mapping:
                            hu = mapping[u]
                            edge = (min(hu, h), max(hu, h))
                            if edge in used_edges:
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)
                
                # Symmetry breaking for T20 (first packed)
                if tree_idx == 19 and len(used_edges) == 0:
                    if v == 0 and placed_count == 0:
                        valid_hosts = [0]
                    elif placed_count == 1 and 0 in mapping and mapping[0] == 0 and 0 in adj[v]:
                        valid_hosts = [h for h in valid_hosts if h == 1]
                        if not valid_hosts:
                            return False
                    elif placed_count == 2 and 0 in mapping and mapping[0] == 0 and 1 in mapping and mapping[1] == 1:
                        # Try to fix a third vertex if possible?
                        # If v is neighbor of 1, fix to 2?
                        if 1 in adj[v]:
                            valid_hosts = [h for h in valid_hosts if h == 2]
                            if not valid_hosts:
                                return False

                # Randomize order of valid hosts if rng
                if rng:
                    rng.shuffle(valid_hosts)
                
                for h in valid_hosts:
                    mapping[v] = h
                    if backtrack(placed_count + 1):
                        return True
                    del mapping[v]
                
                return False
            
            if backtrack(0):
                for u, v in tree_edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    edge = (min(hu, hv), max(hu, hv))
                    used_edges.add(edge)
                return [mapping[i] for i in range(k)]
            else:
                return None

        for i in range(n - 1, -1, -1):
            res = pack_tree(i)
            if res is None:
                return None
            embeddings[i] = res
        
        return embeddings

    # Try deterministic first
    result = try_pack(None)
    if result is not None:
        return result
    
    # Try randomized restarts
    for seed in range(1, 10):
        if time.time() > deadline - 5.0:
            break
        result = try_pack(seed)
        if result is not None:
            return result
            
    # If all fail, return best effort? 
    # The task requires 20 entries. If we can't pack all, we should return what we can?
    # "Entry k-1 is None or a list".
    # If we return None for some, they are not counted.
    # Let's return a list of Nones to be safe, or try to pack as many as possible.
    # Given the knowledge, it should succeed.
    return [None] * n
