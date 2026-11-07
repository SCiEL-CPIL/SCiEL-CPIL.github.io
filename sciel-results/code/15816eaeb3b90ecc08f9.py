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
    deadline = start_time + 135.0
    n = 20
    trees = instance_trees(n, 611)
    
    tree_adj = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    def check_time():
        return time.time() < deadline
    
    def embed_tree(tree_idx, used_edges_set, rng):
        k = tree_idx + 1
        if k == 1:
            return [0]
        
        adj = tree_adj[tree_idx]
        edges = trees[tree_idx]
        
        # Build available neighbor sets for host graph
        avail_nb = [set(range(n)) - {i} for i in range(n)]
        for (a, b) in used_edges_set:
            if a > b:
                a, b = b, a
            avail_nb[a].discard(b)
            avail_nb[b].discard(a)
        
        root_t = max(range(k), key=lambda v: len(adj[v]))
        
        parent = [-1] * k
        bfs_order = [root_t]
        visited = [False] * k
        visited[root_t] = True
        for node in bfs_order:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = node
                    bfs_order.append(nb)
        
        root_cands = sorted(range(n), key=lambda g: len(avail_nb[g]), reverse=True)
        
        for root_g in root_cands:
            if not check_time():
                return None
            
            mapping = [-1] * k
            mapping[root_t] = root_g
            used_verts = {root_g}
            
            def dfs(pos):
                if not check_time():
                    return False
                if pos == len(bfs_order):
                    return True
                
                t_node = bfs_order[pos]
                p_node = parent[t_node]
                p_g = mapping[p_node]
                
                cands = avail_nb[p_g] - used_verts
                if not cands:
                    return False
                
                # Score candidates
                def score(g):
                    s = 0
                    for nb in adj[t_node]:
                        if nb != p_node and mapping[nb] != -1:
                            nb_g = mapping[nb]
                            if nb_g in avail_nb[g]:
                                s += 10
                    s += len(avail_nb[g] - used_verts) * 0.1
                    return s
                
                cands_list = sorted(cands, key=score, reverse=True)
                
                for g in cands_list:
                    mapping[t_node] = g
                    used_verts.add(g)
                    if dfs(pos + 1):
                        return True
                    used_verts.remove(g)
                    mapping[t_node] = -1
                
                return False
            
            if dfs(1):
                valid = True
                for u, v in edges:
                    e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    if e in used_edges_set:
                        valid = False
                        break
                if valid:
                    return mapping[:]
        
        return None
    
    best_solution = [None] * n
    best_value = 0
    rng = Random(123)
    
    for attempt in range(500):
        if not check_time():
            break
        
        order = list(range(n - 1, -1, -1))
        for i in range(len(order) - 1):
            if rng.random() < 0.2:
                j = i + 1 + rng.randint(0, min(2, len(order) - i - 2))
                order[i], order[j] = order[j], order[i]
        
        used_edges = set()
        solution = [None] * n
        success = True
        
        for t_idx in order:
            if not check_time():
                success = False
                break
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [0]
                continue
            mapping = embed_tree(t_idx, used_edges, rng)
            if mapping is None:
                success = False
                break
            for u, v in trees[t_idx]:
                used_edges.add((min(mapping[u], mapping[v]), max(mapping[u], mapping[v])))
            solution[t_idx] = mapping
        
        if success:
            val = sum(len(trees[i]) for i in range(n))
            if val > best_value:
                best_value = val
                best_solution = solution[:]
            if val >= 190:
                break
    
    if best_value == 0:
        return [None] * 19 + [list(range(20))]
    return best_solution
