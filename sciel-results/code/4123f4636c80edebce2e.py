import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
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

    trees = instance_trees(20, 611)
    
    def get_host_edge(u, v):
        return (min(u, v), max(u, v))
        
    def compute_cost(perm):
        edge_counts = defaultdict(int)
        for k in range(1, 20): # T2 to T20 (indices 1 to 19 in perm)
            edges = trees[k] # trees[1] is T2
            for u, v in edges:
                hu = perm[k][u]
                hv = perm[k][v]
                e = get_host_edge(hu, hv)
                edge_counts[e] += 1
        
        cost = 0
        for count in edge_counts.values():
            if count > 1:
                cost += count - 1
        return cost

    def initialize():
        perm = [None] * 20
        perm[0] = [0] # T1, doesn't matter
        for k in range(1, 20):
            hosts = list(range(20))
            random.shuffle(hosts)
            perm[k] = hosts[:k+1]
        return perm

    best_cost = 190
    best_perm = None
    
    # Run multiple restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        perm = initialize()
        current_cost = compute_cost(perm)
        
        if current_cost == 0:
            best_perm = perm
            break
            
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            if iterations > 5000: # Limit iterations per restart
                break
                
            # Choose move type
            move_type = random.choice([0, 1])
            
            if move_type == 0:
                # Swap within same tree
                k = random.randint(1, 19)
                if k+1 < 2: continue # T2 has 2 vertices, can swap
                v1, v2 = random.sample(range(k+1), 2)
                
                old_h1 = perm[k][v1]
                old_h2 = perm[k][v2]
                
                perm[k][v1] = old_h2
                perm[k][v2] = old_h1
                
                new_cost = compute_cost(perm)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        best_perm = perm
                        break
                else:
                    perm[k][v1] = old_h1
                    perm[k][v2] = old_h2
            else:
                # Swap between two trees
                k1, k2 = random.sample(range(1, 20), 2)
                v1 = random.randint(0, k1)
                v2 = random.randint(0, k2)
                
                old_h1 = perm[k1][v1]
                old_h2 = perm[k2][v2]
                
                perm[k1][v1] = old_h2
                perm[k2][v2] = old_h1
                
                new_cost = compute_cost(perm)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        best_perm = perm
                        break
                else:
                    perm[k1][v1] = old_h1
                    perm[k2][v2] = old_h2

        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = perm

    if best_perm is None:
        # Fallback: return initial random or seed-like
        best_perm = initialize()
        
    result = []
    for k in range(20):
        if k == 0:
            result.append(None)
        else:
            result.append([int(v) for v in best_perm[k]])
            
    return result
