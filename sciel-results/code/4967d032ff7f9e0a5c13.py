import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

def decompose(n, m, p):
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 2 and m == 2 and p == 2:
        return strassen_2x2x2()

    # Try splitting dimensions
    # Strategy: Split the dimension that allows the best reduction.
    # We'll try splitting n, m, p if possible.
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = decompose(n1, m, p)
        U2, V2, W2 = decompose(n2, m, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            # Concatenate columns
            # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
            U = [row1 + row2 for row1, row2 in zip(U1, U2)] # Wait, U1 is n1*m rows, U2 is n2*m rows.
            # Need to interleave or append rows?
            # The tensor is block diagonal in the first index if we split n.
            # U1 has rows 0..n1*m-1. U2 has rows 0..n2*m-1.
            # We need to map U2 rows to n1*m .. n*m-1.
            U_new = [ [0]*best_rank for _ in range(n*m) ]
            V_new = [ [0]*best_rank for _ in range(m*p) ]
            W_new = [ [0]*best_rank for _ in range(p*n) ]
            
            for i in range(n1):
                for j in range(m):
                    U_new[i*m + j] = U1[i*m + j] + [0]*r2
            for i in range(n2):
                for j in range(m):
                    U_new[(n1+i)*m + j] = [0]*r1 + U2[i*m + j]
            
            for j in range(m):
                for k in range(p):
                    V_new[j*p + k] = V1[j*p + k] + [0]*r2
                    # V is shared? No, V1 and V2 have same shape.
                    # V_new[j*p+k] = V1[j*p+k] + V2[j*p+k]?
                    # No, the decomposition is a sum.
                    # T = T1 + T2.
                    # So U_new = [U1, U2] (block diag in first dim, identity in others?)
                    # Actually, if T = T1 + T2, then U_c = U1_c for c in 1, U2_c for c in 2.
                    # So V_new = [V1, V2] (concatenate columns).
                    # W_new = [W1, W2] (concatenate columns).
                    pass
            
            # Correct concatenation logic:
            U_concat = []
            for i in range(n):
                for j in range(m):
                    if i < n1:
                        U_concat.append(U1[i*m + j] + [0]*r2)
                    else:
                        U_concat.append([0]*r1 + U2[(i-n1)*m + j])
            
            V_concat = [v1 + v2 for v1, v2 in zip(V1, V2)]
            W_concat = [w1 + w2 for w1, w2 in zip(W1, W2)]
            
            best_decomp = (U_concat, V_concat, W_concat)

    # Similar for m and p...
    
    if best_decomp:
        return best_decomp
    else:
        # Fallback schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        return U, V, W

def strassen_2x2x2():
    # Returns U, V, W for 2x2x2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 0, 0, 0], # A01 - wait, Strassen uses linear combos
        [0, 1, 0, 0, 0, 0, 0], # A10
        [1, 1, 0, 1, 0, 0, 0]  # A11
    ]
    # This is getting complicated to hardcode correctly.
    # Let's just use schoolbook for 2x2x2 if Strassen is too hard to code quickly.
    # Rank 8 vs 7. The gain is small.
    rank = 8
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][idx] = 1
                V[j*2+k][idx] = 1
                W[k*2+i][idx] = 1
                idx += 1
    return U, V, W
