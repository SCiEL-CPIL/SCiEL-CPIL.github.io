import random

def construct():
    trees = instance_trees(20, 611)
    
    # Precompute adjacency
    tree_adj = [{} for _ in range(20)]
    for i, edges in enumerate(trees):
        for u, v in edges:
            tree_adj[i].setdefault(u, []).append(v)
            tree_adj[i].setdefault(v, []).append(u)
        for u in range(len(trees[i])):
            if u not in tree_adj[i]:
                tree_adj[i][u] = []

    def try_packing(order):
        used_edges = set()
        embeddings = [None] * 20
        
        for idx in order:
            k = len(trees[idx])
            if k == 1:
                embeddings[idx] = [0]
                continue
            
            tv_order = sorted(range(k), key=lambda x: -len(tree_adj[idx][x]))
            mapping = [-1] * k
            host_used = set()
            
            def backtrack(t_idx):
                if t_idx == k:
                    return True
                
                tv = tv_order[t_idx]
                
                for hv in range(20):
                    if hv in host_used:
                        continue
                    
                    valid = True
                    for nb in tree_adj[idx][tv]:
                        if mapping[nb] != -1:
                            e1, e2 = (mapping[nb], hv) if mapping[nb] < hv else (hv, mapping[nb])
                            if (e1, e2) in used_edges:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    mapping[tv] = hv
                    host_used.add(hv)
                    
                    if backtrack(t_idx + 1):
                        return True
                    
                    mapping[tv] = -1
                    host_used.remove(hv)
                
                return False
            
            if not backtrack(0):
                return None
            
            embeddings[idx] = [mapping[i] for i in range(k)]
            for u, v in trees[idx]:
                hu, hv = embeddings[idx][u], embeddings[idx][v]
                e1, e2 = (hu, hv) if hu < hv else (hv, hu)
                used_edges.add((e1, e2))
        
        return embeddings

    # Try greedy order: largest first
    order = list(range(20))
    order.sort(key=lambda i: -len(trees[i]))
    
    result = try_packing(order)
    
    if result is None:
        # Try a few random orders
        for _ in range(10):
            random_order = list(range(20))
            random.shuffle(random_order)
            result = try_packing(random_order)
            if result is not None:
                break
                
    if result is None:
        # Return the seed solution as fallback
        return [None] * 19 + [list(range(20))]
        
    return result
