import time
import random
import numpy as np

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0
    
    A = np.zeros((n, n), dtype=int)
    neighbors = [0] * n
    
    def has_path_len_4(u, v):
        Nu = neighbors[u]
        Nv = neighbors[v]
        
        for b in range(n):
            if b == u or b == v:
                continue
            Nb = neighbors[b]
            cu = Nu & Nb
            cv = Nv & Nb
            if cu and cv:
                if cu == cv:
                    # If they are the same set, we can only avoid a==c if there are at least 2 choices
                    if cu & (cu - 1) != 0:
                        return True
                    # else single choice, a==c forced, no simple path
                else:
                    return True
        return False

    # Initialize with random graph p=0.3
    random.seed(42)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.3:
                if not has_path_len_4(u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
    
    best_count = int(np.sum(A)) // 2
    best_A = A.copy()
    best_neighbors = neighbors[:]
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline and T > T_min:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        if A[u, v] == 1:
            # Try remove
            # Delta E = 1 (cost increases)
            # Accept if exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                A[u, v] = 0
                A[v, u] = 0
                neighbors[u] &= ~(1 << v)
                neighbors[v] &= ~(1 << u)
        else:
            # Try add
            if not has_path_len_4(u, v):
                A[u, v] = 1
                A[v, u] = 1
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
        
        T *= alpha
        
        curr_count = int(np.sum(A)) // 2
        if curr_count > best_count:
            best_count = curr_count
            best_A = A.copy()
            best_neighbors = neighbors[:]
            
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                final_edges.append([u, v])
                
    return final_edges
