import time
import random
import numpy as np
from itertools import combinations

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return edges

def _is_regular_subgraph(adj_masks, verts):
    k = len(verts)
    mask = 0
    for v in verts:
        mask |= (1 << v)
    first_deg = (adj_masks[verts[0]] & mask).bit_count()
    if (k * first_deg) % 2 != 0:
        return False
    for v in verts[1:]:
        if (adj_masks[v] & mask).bit_count() != first_deg:
            return False
    return True

def _has_regular_subgraph(adj_masks, n, min_size=6):
    for k in range(min_size, n + 1):
        for verts in combinations(range(n), k):
            if _is_regular_subgraph(adj_masks, verts):
                return True
    return False

def _greedy_construct(n, deadline, seed=42):
    rng = random.Random(seed)
    adj_masks = [0] * n
    
    for i in range(min(5, n)):
        for j in range(i + 1, min(5, n)):
            if rng.random() < 0.5:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    for v in range(5, n):
        if time.time() > deadline:
            return None
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            candidate = [0] * n
            for u in range(v):
                if rng.random() < 0.5:
                    candidate[v] |= (1 << u)
                    candidate[u] |= (1 << v)
            
            valid = True
            for k in range(6, v + 1):
                for others in combinations(range(v), k - 1):
                    verts = list(others) + [v]
                    if _is_regular_subgraph(candidate, verts):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for u in range(v):
                    if (candidate[v] >> u) & 1:
                        adj_masks[v] |= (1 << u)
                        adj_masks[u] |= (1 << v)
                break
        else:
            return None
    
    if _has_regular_subgraph(adj_masks, n):
        return None
    
    return adj_masks

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_adj_masks = []
    
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
        
        n = target_n
        found = False
        
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            
            adj_masks = _greedy_construct(n, deadline - 5, seed=seed)
            
            if adj_masks is not None:
                best_n = n
                best_adj_masks = adj_masks
                found = True
                break
        
        if found:
            break
    
    if best_n == 0:
        for target_n in range(19, 5, -1):
            if time.time() > deadline - 3:
                break
            n = target_n
            for seed in range(20):
                if time.time() > deadline - 2:
                    break
                adj_masks = _greedy_construct(n, deadline - 2, seed=seed)
                if adj_masks is not None:
                    best_n = n
                    best_adj_masks = adj_masks
                    break
            if best_n > 0:
                break
    
    if best_n == 0:
        best_n = 5
        best_adj_masks = [0] * 5
    
    edges = _edges_from_adj_masks(best_adj_masks, best_n)
    return {'n': best_n, 'edges': edges}
