import numpy as np
import time

def construct():
    n = 31
    m = 91
    start_time = time.time()
    time_limit = 140.0

    def count_c4(A):
        d = A.sum(axis=1)
        A2 = A @ A
        A4 = A2 @ A2
        tr_A4 = np.trace(A4)
        sum_d2 = np.sum(d**2)
        return int((tr_A4 - 2 * sum_d2 + 2 * m) / 8)

    # 1. Greedy Construction
    A = np.zeros((n, n), dtype=np.int8)
    current_score = 0
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 0:
                    A[i,j] = 1
                    A[j,i] = 1
                    score = count_c4(A)
                    A[i,j] = 0
                    A[j,i] = 0
                    
                    if score < best_score:
                        best_score = score
                        best_edge = (i, j)
        
        if best_edge is not None:
            u, v = best_edge
            A[u,v] = 1
            A[v,u] = 1
            current_score = best_score

    edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i,j] == 1]

    # 2. Local Search
    while time.time() - start_time < time_limit:
        e1 = edges[np.random.randint(len(edges))]
        u1, v1 = e1
        
        added = False
        for _ in range(20):
             u2 = np.random.randint(n)
             v2 = np.random.randint(n)
             if u2 == v2: continue
             if u2 > v2: u2, v2 = v2, u2
             if A[u2,v2] == 0 and (u2,v2) != (u1,v1):
                 A[u1,v1] = 0
                 A[v1,u1] = 0
                 A[u2,v2] = 1
                 A[v2,u2] = 1
                 
                 new_score = count_c4(A)
                 
                 if new_score <= current_score:
                     current_score = new_score
                     edges.remove(e1)
                     edges.append((u2,v2))
                     added = True
                     break
                 else:
                     A[u1,v1] = 1
                     A[v1,u1] = 1
                     A[u2,v2] = 0
                     A[v2,u2] = 0

    return [list(e) for e in edges]
