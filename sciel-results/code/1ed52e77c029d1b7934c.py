import time
import itertools
import random

def construct():
    deadline = time.time() + 140
    n = 16
    
    all_4sets = list(itertools.combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    disjoint_pairs = []
    for i, A in enumerate(all_4sets):
        A_set = set(A)
        for j, B in enumerate(all_4sets):
            if i != j and not (A_set & set(B)):
                disjoint_pairs.append((i, j, A, B))
                
    best_edges = []
    best_count = 0
    
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        adj = [0] * n
        miss = [[15] * num_4sets for _ in range(n)]
        
        def update_miss(u, v, add_edge):
            for b_idx in range(num_4sets):
                B = all_4sets[b_idx]
                if v in B:
                    pos = B.index(v)
                    mask = 1 << pos
                    if add_edge:
                        miss[u][b_idx] &= ~mask
                    else:
                        miss[u][b_idx] |= mask
            for a_idx in range(num_4sets):
                A = all_4sets[a_idx]
                if u in A:
                    pos = A.index(u)
                    mask = 1 << pos
                    if add_edge:
                        miss[v][a_idx] &= ~mask
                    else:
                        miss[v][a_idx] |= mask

        def is_violation(i, j, A, B):
            m0 = miss[A[0]][j]
            m1 = miss[A[1]][j]
            m2 = miss[A[2]][j]
            m3 = miss[A[3]][j]
            
            if (m0 & m1) or (m0 & m2) or (m0 & m3) or (m1 & m2) or (m1 & m3) or (m2 & m3):
                return False
            if m0 > 1 or m1 > 1 or m2 > 1 or m3 > 1:
                return False
                
            total = bin(m0).count('1') + bin(m1).count('1') + bin(m2).count('1') + bin(m3).count('1')
            return total <= 4

        def check_valid():
            for i, j, A, B in disjoint_pairs:
                if is_violation(i, j, A, B):
                    return False
            return True

        random.seed(42 + attempt)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        current_edges = []
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            if adj[u] & (1 << v):
                continue
                
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            update_miss(u, v, True)
            
            if check_valid():
                current_edges.append((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                update_miss(u, v, False)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return [[u, v] for u, v in best_edges]
