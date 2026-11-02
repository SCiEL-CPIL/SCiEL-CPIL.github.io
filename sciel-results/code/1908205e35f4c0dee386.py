import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Rank-28 Strassen decomposition: apply rank-7 Strassen to each of 4 column-pairs.
    # This is provably optimal: rank(M_{2,2,8}) = 28.
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2x2 matrix multiplication:
    # Each product has:
    #   'A': dict {(i,j): coeff} for A[i,j]
    #   'B': dict {(j,k_sub): coeff} for B[j, k_sub] where k_sub in {0,1}
    #   'C': dict {(i, k_sub): coeff} for C[i, k_sub] where k_sub in {0,1}
    strassen_products = [
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    # Apply Strassen to each of the 4 column-pairs (k=0,1), (k=2,3), (k=4,5), (k=6,7)
    for q in range(p // 2):
        k1 = 2 * q  # first column of the pair
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # global term index
            
            # Fill U: U[i*m + j, t]
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: V[j*p + k_global, t]
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: W[k*n + i, t] (official trace convention)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verify correctness: the tensor decomposition should satisfy
    # sum_r U[a, r] * V[b, r] * W[c, r] = delta_{j,j'} * delta_{k,k'}
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # i.e., it equals 1 iff j==j' and k==k' (and i==i' is implicit in the W index)
    
    # Efficient verification: build the tensor T[a,b,c] and check against the target
    # Target: T[a,b,c] = 1 iff (a%m == b//p) and (b%p == c//n)
    #         (and a//m == c%n, but that's automatically satisfied by construction)
    
    # Use einsum for efficient tensor computation
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is equivalent to: for each r, outer product U[:,r] @ V[:,r].T @ W[:,r].T
    # But more efficiently: T = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    # a = i*m+j (0..3), b = j'*p+k (0..15), c = k'*n+i' (0..15)
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    a = i * m + j
                    b = jp * p + k
                    c = k * n + i
                    if j == jp:
                        T_target[a, b, c] = 1.0
    
    # Check correctness
    max_err = np.max(np.abs(T_computed - T_target))
    if max_err > 1e-10:
        raise ValueError(f"Decomposition incorrect, max error: {max_err}")
    
    # Check if time is running out
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
