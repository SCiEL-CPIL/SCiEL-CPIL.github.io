import random
import numpy as np

def construct():
    n = 20
    best_edges_count = 0
    best_edges = []
    
    def is_k33_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = np.logical_and(np.logical_and(adj[i], adj[j]), adj[k])
                    if np.sum(common) >= 3:
                        return False
        return True

    def find_violation(adj):
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = np.logical_and(np.logical_and(adj[i], adj[j]), adj[k])
                    if np.sum(common) >= 3:
                        return i, j, k, np.where(common)[0]
        return None

    def run_heuristic(seed):
        nonlocal best_edges_count, best_edges
        random.seed(seed)
        p = 0.65
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Phase 1: Remove edges
        max_removals = 100
        removals = 0
        while removals < max_removals:
            violation = find_violation(adj)
            if violation is None:
                break
            i, j, k, commons = violation
            c = commons[random.randint(0, len(commons)-1)]
            v = [i, j, k][random.randint(0, 2)]
            adj[v, c] = 0
            adj[c, v] = 0
            removals += 1
            
        # Phase 2: Add edges
        for _ in range(10):
            added_any = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
            random.shuffle(pairs)
            for i, j in pairs:
                adj[i, j] = 1
                adj[j, i] = 1
                if is_k33_free(adj):
                    added_any = True
                else:
                    adj[i, j] = 0
                    adj[j, i] = 0
            if not added_any:
                break
                
        edge_count = np.sum(adj) // 2
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] == 1:
                        edges.append([i, j])
            best_edges = edges

    for seed in range(20):
        run_heuristic(seed)
        
    return best_edges
