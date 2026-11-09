import time
import numpy as np
import random
from collections import defaultdict

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    best_edges = []
    best_count = -1
    
    def get_edges_from_A(A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    edges.append([int(i), int(j)])
        return edges

    def count_edges(A):
        return int(A.sum() // 2)

    def is_valid(A):
        degs = A.sum(axis=1)
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[degs[i]].append(i)
            
        A2 = A @ A
        A3 = A2 @ A
        
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    if A3[s, t] > 0:
                        return False, (s, t)
        return True, None

    def optimize(A):
        # Phase 1: Remove edges to fix violations
        for _ in range(200):
            valid, viol = is_valid(A)
            if valid:
                break
            s, t = viol
            # Find a path
            found = False
            # Randomize order of u, v to explore different solutions
            us = list(range(n))
            vs = list(range(n))
            random.shuffle(us)
            random.shuffle(vs)
            for u in us:
                if A[s, u] == 0: continue
                for v in vs:
                    if A[u, v] == 0: continue
                    if A[v, t] == 0: continue
                    # Remove (s, u)
                    A[s, u] = 0
                    A[u, s] = 0
                    found = True
                    break
                if found: break
            if not found:
                break
                
        # Phase 2: Add edges greedily
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 0:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        changed = True
        while changed:
            changed = False
            for i, j in non_edges:
                if A[i, j] == 1:
                    continue
                A[i, j] = 1
                A[j, i] = 1
                valid, _ = is_valid(A)
                if not valid:
                    A[i, j] = 0
                    A[j, i] = 0
                else:
                    changed = True
                    
        return A

    # Run multiple trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        random.seed(trial)
        # Start with complete graph
        A = np.ones((n, n), dtype=int)
        np.fill_diagonal(A, 0)
        
        # Maybe start with a random sparse graph to escape local minima?
        # But complete graph is a good start for "remove" phase.
        
        A_opt = optimize(A)
        
        valid, _ = is_valid(A_opt)
        if valid:
            count = count_edges(A_opt)
            if count > best_count:
                best_count = count
                best_edges = get_edges_from_A(A_opt)
        
        # Try starting from a different initial state?
        # e.g. Random graph with p=0.5
        if trial % 5 == 0:
             A_rand = (np.random.rand(n, n) < 0.5).astype(int)
             np.fill_diagonal(A_rand, 0)
             A_rand = (A_rand + A_rand.T) // 2 # Symmetrize
             A_opt = optimize(A_rand)
             valid, _ = is_valid(A_opt)
             if valid:
                 count = count_edges(A_opt)
                 if count > best_count:
                     best_count = count
                     best_edges = get_edges_from_A(A_opt)

    if not best_edges:
        # Fallback: Return a known valid small graph if search fails
        # Empty graph is valid (0 edges)
        return []
        
    return best_edges
