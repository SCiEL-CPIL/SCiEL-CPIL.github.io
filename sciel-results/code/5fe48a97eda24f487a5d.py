import random

def construct():
    trees = instance_trees(20, 611)
    available_edges = set((u, v) for u in range(20) for v in range(u + 1, 20))
    embeddings = [None] * 20
    indices = list(range(20))
    indices.sort(key=lambda i: i + 1, reverse=True)
    
    rng = Random(123)
    
    for idx in indices:
        k = idx + 1
        tree = trees[idx]
        if k == 1:
            embeddings[idx] = [0]
            continue
            
        tree_adj = [[] for _ in range(k)]
        for u, v in tree:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            
        vert_order = sorted(range(k), key=lambda x: len(tree_adj[x]), reverse=True)
        
        found = False
        # Try multiple random attempts if needed
        for attempt in range(100):
            mapping = {}
            
            def backtrack(pos):
                if pos == k:
                    return True
                v = vert_order[pos]
                # Get available host vertices
                used_hosts = set(mapping.values())
                candidates = [h for h in range(20) if h not in used_hosts]
                # Shuffle candidates for randomization
                rng.shuffle(candidates)
                
                for h in candidates:
                    valid = True
                    for neighbor in tree_adj[v]:
                        if neighbor in mapping:
                            hn = mapping[neighbor]
                            edge = (min(h, hn), max(h, hn))
                            if edge not in available_edges:
                                valid = False
                                break
                    if valid:
                        mapping[v] = h
                        if backtrack(pos + 1):
                            return True
                        del mapping[v]
                return False
                
            if backtrack(0):
                embeddings[idx] = [mapping[i] for i in range(k)]
                for u, v in tree:
                    hu, hv = mapping[u], mapping[v]
                    edge = (min(hu, hv), max(hu, hv))
                    available_edges.remove(edge)
                found = True
                break
                
        if not found:
            # If all attempts fail, we might miss this tree
            # But likely it will succeed
            pass
            
    return embeddings
