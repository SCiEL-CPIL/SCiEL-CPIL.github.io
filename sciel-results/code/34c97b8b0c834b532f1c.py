import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We'll build the decomposition incrementally
    # Each term is a triple (u_vec, v_vec, w_vec) where:
    # u_vec is length n*m, v_vec is length m*p, w_vec is length p*n
    
    terms = []
    
    # Define the 4 Strassen blocks
    # Each block: (i_set, j_set, k_set) where each is a tuple of 2 indices
    blocks = [
        ((0, 1), (0, 1), (0, 1)),
        ((0, 1), (0, 1), (2, 3)),
        ((0, 1), (0, 1), (4, 5)),
        ((0, 1), (0, 1), (6, 7)),
    ]
    
    # Track which (i,j,k) triples are covered by Strassen blocks
    strassen_covered = set()
    for (i_set, j_set, k_set) in blocks:
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    strassen_covered.add((i, j, k))
    
    # Add Strassen terms for each block
    # Strassen vectors for 2x2x2 (local indices 0,1)
    # u_r[i_local, j_local], v_r[j_local, k_local], w_r[k_local, i_local]
    # The 7 Strassen rank-1 tensors:
    strassen_u = [
        [1, 0, 0, 1],  # r=1: A00 + A11
        [0, 0, 1, 1],  # r=2: A10 + A11
        [1, 0, 0, 0],  # r=3: A00
        [0, 0, 0, 1],  # r=4: A11
        [1, 1, 0, 0],  # r=5: A00 + A01
        [-1, 0, 1, 0], # r=6: A10 - A00
        [0, 1, 0, -1], # r=7: A01 - A11
    ]
    strassen_v = [
        [1, 0, 0, 1],  # r=1: B00 + B11
        [1, 0, 0, 0],  # r=2: B00
        [0, 1, 0, -1], # r=3: B01 - B11
        [0, 0, 1, -1], # r=4: B10 - B00
        [0, 0, 0, 1],  # r=5: B11
        [1, 1, 0, 0],  # r=6: B00 + B01
        [0, 0, 1, 1],  # r=7: B10 + B11
    ]
    strassen_w = [
        [1, 0, 0, 1],  # r=1: C00 + C11
        [0, 0, 1, 0],  # r=2: C10
        [0, 1, 0, 0],  # r=3: C01
        [-1, 0, 0, 1], # r=4: -C00 + C11
        [0, 1, 0, 0],  # r=5: C01
        [0, 0, 1, 0],  # r=6: C10
        [1, 0, 0, 0],  # r=7: C00
    ]
    
    for (i_set, j_set, k_set) in blocks:
        i0, i1 = i_set
        j0, j1 = j_set
        k0, k1 = k_set
        
        for r in range(7):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Map local indices to global
            # u: local (li, lj) -> global (i, j)
            # li=0 -> i0, li=1 -> i1
            # lj=0 -> j0, lj=1 -> j1
            for li in range(2):
                for lj in range(2):
                    i_global = i0 if li == 0 else i1
                    j_global = j0 if lj == 0 else j1
                    u[i_global * m + j_global] = strassen_u[r][li * 2 + lj]
            
            # v: local (lj, lk) -> global (j, k)
            # lj=0 -> j0, lj=1 -> j1
            # lk=0 -> k0, lk=1 -> k1
            for lj in range(2):
                for lk in range(2):
                    j_global = j0 if lj == 0 else j1
                    k_global = k0 if lk == 0 else k1
                    v[j_global * p + k_global] = strassen_v[r][lj * 2 + lk]
            
            # w: local (lk, li) -> global (k, i)
            # lk=0 -> k0, lk=1 -> k1
            # li=0 -> i0, li=1 -> i1
            for lk in range(2):
                for li in range(2):
                    k_global = k0 if lk == 0 else k1
                    i_global = i0 if li == 0 else i1
                    w[k_global * n + i_global] = strassen_w[r][lk * 2 + li]
            
            terms.append((u, v, w))
    
    # Add schoolbook terms for all (i,j,k) NOT covered by Strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    terms.append((u, v, w))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            if u[idx] != 0:
                U[idx][r] = u[idx]
        for idx in range(m * p):
            if v[idx] != 0:
                V[idx][r] = v[idx]
        for idx in range(p * n):
            if w[idx] != 0:
                W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}
