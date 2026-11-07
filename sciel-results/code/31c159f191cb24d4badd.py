import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    N = 20
    
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    # Precompute tree structures
    tree_info = []
    for i in range(20):
        k = i + 1
        edges = trees[i]
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': [[]], 'root_order': [0], 'parent': [-2]})
            continue
        if k == 2:
            tree_info.append({'k': 2, 'edges': [(0, 1)], 'adj': [[1], [0]], 'root_order': [0, 1], 'parent': [-2, 0]})
            continue
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v); adj[v].append(u)
        # BFS from 0
        parent = [-1] * k
        parent[0] = -2
        order = [0]
        q = deque([0])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if parent[y] == -1:
                    parent[y] = x
                    q.append(y)
                    order.append(y)
        tree_info.append({'k': k, 'edges': edges, 'adj': adj, 'root_order': order, 'parent': parent})
    
    def try_pack(seed_val, time_limit):
        rng = Random(seed_val)
        used = set()
        result = [None] * 20
        
        # Process trees in order of decreasing size for constraint propagation
        order_idx = list(range(20))
        order_idx.sort(key=lambda i: -(i + 1))
        
        for idx in order_idx:
            info = tree_info[idx]
            k = info['k']
            
            if k == 1:
                # Single vertex, any unused host
                for h in range(N):
                    if h not in used:
                        result[idx] = [h]
                        break
                else:
                    return None
                continue
            
            if k == 2:
                # Find any unused edge
                placed = False
                for a in range(N):
                    for b in range(a + 1, N):
                        e = (a, b)
                        if e not in used:
                            result[idx] = [a, b]
                            used.add(e)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    return None
                continue
            
            # General tree: backtracking with MRV heuristic and forward checking
            adj = info['adj']
            parent = info['parent']
            order = info['root_order']
            
            assignment = [-1] * k
            used_hosts = [False] * N
            
            # Count how many free edges each host has
            def count_free_edges(h):
                cnt = 0
                for other in range(N):
                    if other == h:
                        continue
                    e = edge_key(h, other)
                    if e not in used:
                        cnt += 1
                return cnt
            
            def dfs(idx_v):
                if time.time() > time_limit:
                    return False
                if idx_v == k:
                    return True
                
                v = order[idx_v]
                pv = parent[v]
                
                # Get candidate hosts
                candidates = []
                for h in range(N):
                    if used_hosts[h]:
                        continue
                    if pv >= 0:
                        ph = assignment[pv]
                        e = edge_key(h, ph)
                        if e in used:
                            continue
                    candidates.append(h)
                
                if not candidates:
                    return False
                
                # Sort candidates by number of free edges (ascending) - most constrained first
                candidates.sort(key=lambda h: count_free_edges(h))
                
                for h in candidates:
                    assignment[v] = h
                    used_hosts[h] = True
                    if dfs(idx_v + 1):
                        return True
                    used_hosts[h] = False
                
                assignment[v] = -1
                return False
            
            if dfs(0):
                result[idx] = [assignment[i] for i in range(k)]
                for (u, v) in info['edges']:
                    used.add(edge_key(assignment[u], assignment[v]))
            else:
                return None
        
        # Verify
        seen = set()
        for i in range(20):
            r = result[i]
            if r is None:
                if i == 0:
                    continue
                return None
            if len(set(r)) != len(r):
                return None
            for x in r:
                if not (0 <= x <= 19):
                    return None
            for (u, v) in trees[i]:
                e = edge_key(r[u], r[v])
                if e in seen:
                    return None
                seen.add(e)
        
        return result
    
    # Try multiple seeds with increasing time allocation
    best_result = None
    best_edges = 0
    for s in range(2000):
        elapsed = time.time() - (deadline - 145.0)
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate more time to later attempts as we get closer to deadline
        time_per_attempt = min(remaining * 0.3, 2.0)
        res = try_pack(s, time.time() + time_per_attempt)
        if res is not None:
            # Count edges
            edge_count = 0
            for i in range(20):
                if res[i] is not None:
                    edge_count += len(trees[i])
            
            if edge_count > best_edges:
                best_edges = edge_count
                best_result = res
                
                # If we found a complete solution, return it
                if edge_count == 190:
                    out = []
                    for i in range(20):
                        r = res[i]
                        if r is None:
                            out.append(None)
                        else:
                            out.append([int(x) for x in r])
                    return out
    
    # Return best partial solution found
    if best_result is not None:
        out = []
        for i in range(20):
            r = best_result[i]
            if r is None:
                out.append(None)
            else:
                out.append([int(x) for x in r])
        return out
    
    # Fallback: return seed solution
    return [None] * 19 + [list(range(20))]
