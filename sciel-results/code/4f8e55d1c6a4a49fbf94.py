# EVOLVE-BLOCK-START
import time
import random
from collections import deque

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
    deadline = time.time() + 115.0  # safety margin
    n = 20
    trees = instance_trees(n, 611)
    
    # Track used edges
    used_edges = set()
    result = [None] * n
    
    # Embed trees from largest to smallest
    order = list(range(n - 1, -1, -1))
    
    for idx in order:
        if time.time() > deadline:
            break
        k = idx + 1
        tree_edges = trees[idx]
        
        if k == 1:
            result[idx] = [0]
            continue
        
        if k == 2:
            # Single edge
            found = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if (u, v) not in used_edges and (v, u) not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add((u, v))
                        used_edges.add((v, u))
                        found = True
                        break
                if found:
                    break
            continue
        
        # For k >= 3
        tree_adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            tree_adj[a].append(b)
            tree_adj[b].append(a)
            
        # Backtracking
        embedding = [-1] * k
        
        def backtrack(assigned_count):
            if time.time() > deadline:
                return False
            
            if assigned_count == k:
                return True
            
            # Select next tree vertex to assign
            # Heuristic: Pick unassigned vertex with max assigned neighbors
            best_v = -1
            max_assigned_neighbors = -1
            for v in range(k):
                if embedding[v] != -1:
                    continue
                cnt = sum(1 for u in tree_adj[v] if embedding[u] != -1)
                if cnt > max_assigned_neighbors:
                    max_assigned_neighbors = cnt
                    best_v = v
            
            if best_v == -1:
                return False # Should not happen
            
            # Try host vertices for best_v
            for h in range(n):
                # Check validity against assigned neighbors
                valid = True
                for u in tree_adj[best_v]:
                    if embedding[u] != -1:
                        uh = embedding[u]
                        if (uh, h) in used_edges or (h, uh) in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                embedding[best_v] = h
                
                if backtrack(assigned_count + 1):
                    return True
                
                embedding[best_v] = -1
                
            return False
        
        if backtrack(0):
            result[idx] = embedding[:]
            for a, b in tree_edges:
                ua, ub = embedding[a], embedding[b]
                used_edges.add((ua, ub))
                used_edges.add((ub, ua))
        else:
            # Failed to embed
            pass

    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append(list(result[i]))
    
    return final_result
# EVOLVE-BLOCK-END
