import time
import random
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
    deadline = start_time + 140.0
    
    trees = instance_trees(20, 611)
    
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: i, reverse=True)
    
    best_results = [None] * 20
    best_edge_count = 0
    
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        current_used_edges = set()
        current_results = [None] * 20
        success = True
        current_edge_count = 0
        
        for idx in tree_indices:
            if time.time() > deadline:
                success = False
                break
                
            k = idx + 1
            tree_edges = trees[idx]
            
            if k == 1:
                current_results[idx] = [0]
                continue
                
            if k == 2:
                found = False
                hosts = list(range(20))
                random.shuffle(hosts)
                for h1 in hosts:
                    if found: break
                    for h2 in hosts:
                        if h1 == h2: continue
                        edge = (min(h1, h2), max(h1, h2))
                        if edge not in current_used_edges:
                            current_results[idx] = [h1, h2]
                            current_used_edges.add(edge)
                            found = True
                            break
                    if found: break
                if not found:
                    success = False
                    break
                current_edge_count += 1
                continue
                
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            order = list(range(k))
            order.sort(key=lambda x: len(adj[x]), reverse=True)
            
            assignment = [-1] * k
            
            def backtrack(v_idx, used_edges_local, assignment_local):
                if v_idx == k:
                    return True
                
                v = order[v_idx]
                hosts = list(range(20))
                random.shuffle(hosts)
                
                for h in hosts:
                    if h in assignment_local:
                        continue
                    
                    valid = True
                    for u in adj[v]:
                        if assignment_local[u] != -1:
                            a = assignment_local[u]
                            edge = (min(h, a), max(h, a))
                            if edge in used_edges_local:
                                valid = False
                                break
                            
                    if not valid:
                        continue
                    
                    assignment_local[v] = h
                    if backtrack(v_idx + 1, used_edges_local, assignment_local):
                        return True
                    assignment_local[v] = -1
                    
                return False
            
            if backtrack(0, current_used_edges, assignment):
                current_results[idx] = assignment
                for u, v in tree_edges:
                    a = assignment[u]
                    b = assignment[v]
                    edge = (min(a, b), max(a, b))
                    if edge in current_used_edges:
                        success = False
                        break
                    current_used_edges.add(edge)
                if not success:
                    break
                current_edge_count += len(tree_edges)
            else:
                success = False
                break
                
        if current_edge_count > best_edge_count:
            best_edge_count = current_edge_count
            best_results = [r[:] if r is not None else None for r in current_results]
            
        if success:
            break

    final_results = []
    for r in best_results:
        if r is None:
            final_results.append(None)
        else:
            final_results.append([int(x) for x in r])
            
    return final_results
