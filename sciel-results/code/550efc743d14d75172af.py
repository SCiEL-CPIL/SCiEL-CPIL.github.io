import time
import sys

def construct():
    deadline = time.time() + 145.0
    n, m, p = 3, 5, 6
    
    # Initialize matrices with zeros
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    # We don't know rank yet, so we will build lists of terms and then assemble.
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    def add_schoolbook(i, j, k):
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        add_term(u, v, w)

    def add_strassen(i0, i1, j0, j1, k0, k1):
        # Term 1: (A00+A11)(B00+B11)
        u1 = {(i0, j0): 1, (i1, j1): 1}
        v1 = {(j0, k0): 1, (j1, k1): 1}
        w1 = {(k0, i0): 1, (k1, i1): 1}
        add_term(u1, v1, w1)
        
        # Term 2: (A10+A11)B00
        u2 = {(i1, j0): 1, (i1, j1): 1}
        v2 = {(j0, k0): 1}
        w2 = {(k0, i1): 1, (k1, i1): -1}
        add_term(u2, v2, w2)
        
        # Term 3: A00(B10-B11)
        u3 = {(i0, j0): 1}
        v3 = {(j1, k0): 1, (j1, k1): -1}
        w3 = {(k0, i0): 1}
        add_term(u3, v3, w3)
        
        # Term 4: A11(B01-B00)
        u4 = {(i1, j1): 1}
        v4 = {(j0, k1): 1, (j0, k0): -1}
        w4 = {(k1, i1): 1}
        add_term(u4, v4, w4)
        
        # Term 5: (A00+A01)B11
        u5 = {(i0, j0): 1, (i0, j1): 1}
        v5 = {(j1, k1): 1}
        w5 = {(k1, i1): 1, (k0, i0): -1}
        add_term(u5, v5, w5)
        
        # Term 6: (A10-A00)(B00+B01)
        u6 = {(i1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0, k1): 1}
        w6 = {(k1, i1): 1}
        add_term(u6, v6, w6)
        
        # Term 7: (A01-A11)(B10+B11)
        u7 = {(i0, j1): 1, (i1, j1): -1}
        v7 = {(j1, k0): 1, (j1, k1): 1}
        w7 = {(k0, i0): 1, (k1, i1): -1}
        add_term(u7, v7, w7)

    # Partitions
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                if len(i_blk) == 2 and len(j_blk) == 2 and len(k_blk) == 2:
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    add_strassen(i0, i1, j0, j1, k0, k1)
                else:
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                add_schoolbook(i, j, k)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    # Verification
    # Check if sum_c U[i,m+j,c] * V[j,p+k,c] * W[k,n+i,c] == 1 for all i,j,k
    # Using integer arithmetic to be safe, but values are small.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    # Fallback to schoolbook if verification fails
                    # This should not happen if the Strassen implementation is correct
                    return _schoolbook_fallback(n, m, p)
                
    return {"U": U, "V": V, "W": W}

def _schoolbook_fallback(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
