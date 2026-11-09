import time
from random import Random

def _generate_trees(n=20, seed=611):
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
    deadline = start_time + 140
    
    trees = _generate_trees(20, 611)
    used_edges = set()
    result = [None] * 20
    
    for k in range(20):
        if time.time() > deadline:
            break
        
        tree_edges = trees[k]
        
        if not tree_edges:
            result[k] = [0]
            continue
        
        # Find embedding using backtracking
        k_verts = len(tree_edges) + 1
        adj = [[] for _ in range(k_verts)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find leaves
        leaves = [i for i in range(k_verts) if len(adj[i]) == 1]
        
        # Try multiple random starts
        rng = Random(123 + k)
        found = None
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            start_leaf = rng.choice(leaves) if leaves else 0
            
            # Build BFS order
            order = []
            visited = [False] * k_verts
            queue = [start_leaf]
            visited[start_leaf] = True
            while queue:
                curr = queue.pop(0)
                order.append(curr)
                for nb in adj[curr]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            
            # Backtrack assignment
            assign = {}
            assign[start_leaf] = rng.randint(0, 19)
            
            def bt(idx):
                if time.time() > deadline:
                    return False
                if idx == len(order):
                    # Verify
                    for u, v in tree_edges:
                        e = (min(assign[u], assign[v]), max(assign[u], assign[v]))
                        if e in used_edges:
                            return False
                    return True
                
                tv = order[idx]
                # Find assigned neighbor
                parent = None
                for nb in adj[tv]:
                    if nb in assign:
                        parent = nb
                        break
                
                if parent is not None:
                    ph = assign[parent]
                    for ch in range(20):
                        if ch == ph:
                            continue
                        e = (min(ph, ch), max(ph, ch))
                        if e in used_edges:
                            continue
                        if ch in assign.values():
                            continue
                        assign[tv] = ch
                        if bt(idx + 1):
                            return True
                        del assign[tv]
                else:
                    used_h = set(assign.values())
                    for ch in range(20):
                        if ch in used_h:
                            continue
                        assign[tv] = ch
                        if bt(idx + 1):
                            return True
                        del assign[tv]
                return False
            
            if bt(1):
                found = [0] * k_verts
                for tv, hv in assign.items():
                    found[tv] = hv
                break
        
        if found:
            for u, v in tree_edges:
                e = (min(found[u], found[v]), max(found[u], found[v]))
                used_edges.add(e)
            result[k] = found
    
    return [None if r is None else [int(x) for x in r] for r in result]
