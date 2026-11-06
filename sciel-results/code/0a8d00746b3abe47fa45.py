"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def _is_regular_subset(neigh, subset):
    smask = 0
    for v in subset:
        smask |= (1 << v)
    first = (neigh[subset[0]] & smask).bit_count()
    for v in subset[1:]:
        if (neigh[v] & smask).bit_count() != first:
            return False
    return True

def _find_violations_sampled(neigh, n, num_samples=100):
    vertices = list(range(n))
    violations = []
    for _ in range(num_samples):
        k = random.randint(6, n)
        subset = tuple(random.sample(vertices, k))
        if _is_regular_subset(neigh, subset):
            violations.append(subset)
    return violations

def _full_validate(neigh, n):
    vertices = list(range(n))
    for k in range(6, n + 1):
        for subset in combinations(vertices, k):
            if _is_regular_subset(neigh, subset):
                return False
    return True

def _flip_edge(adj, neigh, u, v):
    if adj[u, v]:
        adj[u, v] = 0
        adj[v, u] = 0
        neigh[u] &= ~(1 << v)
        neigh[v] &= ~(1 << u)
    else:
        adj[u, v] = 1
        adj[v, u] = 1
        neigh[u] |= (1 << v)
        neigh[v] |= (1 << u)

def construct():
    deadline = time.time() + 140
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        n = 20
        p = random.uniform(0.35, 0.65)
        adj = np.zeros((n, n), dtype=np.uint8)
        neigh = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    neigh[i] |= (1 << j)
                    neigh[j] |= (1 << i)
        
        for it in range(5000):
            if time.time() > deadline:
                break
            
            violations = _find_violations_sampled(neigh, n, num_samples=100)
            
            if not violations:
                if _full_validate(neigh, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                break
            
            viol = random.choice(violations)
            fixed = False
            for _ in range(100):
                u, v = random.sample(viol, 2)
                _flip_edge(adj, neigh, u, v)
                if not _is_regular_subset(neigh, viol):
                    fixed = True
                    break
                else:
                    _flip_edge(adj, neigh, u, v)
            
            if not fixed:
                u, v = random.sample(range(n), 2)
                _flip_edge(adj, neigh, u, v)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        n = 19
        p = random.uniform(0.35, 0.65)
        adj = np.zeros((n, n), dtype=np.uint8)
        neigh = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    neigh[i] |= (1 << j)
                    neigh[j] |= (1 << i)
        
        for it in range(3000):
            if time.time() > deadline:
                break
            violations = _find_violations_sampled(neigh, n, num_samples=100)
            if not violations:
                if _full_validate(neigh, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                break
            viol = random.choice(violations)
            fixed = False
            for _ in range(50):
                u, v = random.sample(viol, 2)
                _flip_edge(adj, neigh, u, v)
                if not _is_regular_subset(neigh, viol):
                    fixed = True
                    break
                else:
                    _flip_edge(adj, neigh, u, v)
            if not fixed:
                u, v = random.sample(range(n), 2)
                _flip_edge(adj, neigh, u, v)
    
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
