import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Must finish within 120s, keep safety margin
    
    n, m, p = 4, 5, 5
    # n*m = 20, m*p = 25, p*n = 20
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Standard rank is n*m*p = 100
    # We can apply Strassen-like 7-term decomposition to 2x2x2 blocks
    # Each 2x2x2 block has 8 terms normally, Strassen reduces to 7
    # We need to find non-overlapping 2x2x2 blocks in the 4x5x5 tensor
    
    # For a 4x5x5 tensor, we can partition into 2x2x2 blocks:
    # i: 0,1,2,3 -> blocks at (0,1) and (2,3)
    # j: 0,1,2,3,4 -> blocks at (0,1) and (2,3), leaving j=4
    # k: 0,1,2,3,4 -> blocks at (0,1) and (2,3), leaving k=4
    
    # So we can have 2x2x2 blocks where i in {0,1} or {2,3}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    # That's 2*2*2 = 8 blocks, each saving 1 term = 8 terms saved
    # Rank = 100 - 8 = 92
    
    # Let's use all 8 possible 2x2x2 Strassen blocks
    rank = 92
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Define all 2x2x2 blocks
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((
                    [i_base, i_base + 1],
                    [j_base, j_base + 1],
                    [k_base, k_base + 1]
                ))
    
    # Create a set of (i, j, k) tuples to skip
    skip_set = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    skip_set.add((i, j, k))
    
    col = 0
    # Fill standard terms for non-block entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Fill Strassen terms for each block
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i2, col] += 1
        W[k2 * n + i2, col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1, col] += 1
        V[j1 * p + k2, col] += 1
        V[j2 * p + k2, col] += -1
        W[k2 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2, col] += 1
        V[j2 * p + k1, col] += 1
        V[j1 * p + k1, col] += -1
        W[k1 * n + i1, col] += 1
        W[k1 * n + i2, col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1, col] += 1
        U[i1 * m + j2, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += -1
        W[k2 * n + i1, col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1, col] += 1
        U[i1 * m + j1, col] += -1
        V[j1 * p + k1, col] += 1
        V[j1 * p + k2, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2, col] += 1
        U[i2 * m + j2, col] += -1
        V[j2 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
    
    # Check if we're within time limit
    if time.time() > deadline:
        # Return what we have
        pass
    
    # Convert to list format as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
