import time
from random import Random

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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree structures
    tree_info = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        if k == 1:
            tree_info.append({
                'k': 1,
                'edges': [],
                'adj': [[]],
                'deg': [0],
                'mcf_order': [0]
            })
            continue
        
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        
        # Most Constrained First: Sort vertices by degree descending
        mcf_order = list(range(k))
        mcf_order.sort(key=lambda x: -deg[x])
        
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'mcf_order': mcf_order
        })
    
    # Sort trees by size descending (Largest First)
    order = list(range(n))
    order.sort(key=lambda x: tree_info[x]['k'], reverse=True)
    
    best_solution = None
    
    # Try multiple random restarts
    for restart in range(20000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 7919 + 13)
        
        # avail[i][j] = True if edge (i,j) is available
        avail = [[True] * n for _ in range(n)]
        
        used_hosts = [False] * n
        current_solution = [None] * n
        success = True
        
        for t_idx in order:
            info = tree_info[t_idx]
            k = info['k']
            
            if k == 1:
                unused = [v for v in range(n) if not used_hosts[v]]
                if not unused:
                    success = False
                    break
                h = rng.choice(unused)
                current_solution[t_idx] = [h]
                used_hosts[h] = True
                continue
            
            mapping = [-1] * k
            assigned = [False] * k
            mcf_order = info['mcf_order']
            adj = info['adj']
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == k:
                    return True
                
                tv = mcf_order[idx]
                
                # Find candidates
                candidates = []
                # Get list of unused hosts to iterate
                unused_hosts = [v for v in range(n) if not used_hosts[v]]
                
                for h in unused_hosts:
                    # Check constraints from already assigned neighbors
                    valid = True
                    for nb in adj[tv]:
                        if assigned[nb]:
                            nh = mapping[nb]
                            if not avail[h][nh]:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Heuristic: Score based on available edges to other unused hosts
                    # Higher score = more connected = more flexible
                    score = 0
                    for u in unused_hosts:
                        if u == h:
                            continue
                        if avail[h][u]:
                            score += 1
                    
                    # Add random noise to break ties and encourage exploration
                    score += rng.random() * 0.5
                    candidates.append((score, h))
                
                if not candidates:
                    return False
                
                # Sort by score descending
                candidates.sort(key=lambda x: -x[0])
                
                for score, h in candidates:
                    mapping[tv] = h
                    used_hosts[h] = True
                    assigned[tv] = True
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    mapping[tv] = -1
                    used_hosts[h] = False
                    assigned[tv] = False
                
                return False
            
            if not backtrack(0):
                success = False
                break
            
            # Mark edges as used
            for u, v in info['edges']:
                gu = mapping[u]
                gv = mapping[v]
                avail[gu][gv] = False
                avail[gv][gu] = False
            
            current_solution[t_idx] = mapping[:]
        
        if success:
            # Quick verification
            used_edges_set = set()
            valid = True
            for t_idx in range(n):
                sol = current_solution[t_idx]
                if sol is None:
                    valid = False
                    break
                if len(set(sol)) != len(sol):
                    valid = False
                    break
                for u, v in tree_info[t_idx]['edges']:
                    gu = sol[u]
                    gv = sol[v]
                    e = (min(gu, gv), max(gu, gv))
                    if e in used_edges_set:
                        valid = False
                        break
                    used_edges_set.add(e)
                if not valid:
                    break
            
            if valid:
                return current_solution
    
    return [None] * n
