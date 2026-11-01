"""Matrix multiplication 2,3,10: Strassen-based rank-55 decomposition."""
DIMENSIONS = [2, 3, 10]

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for M_{2,2,2} using Strassen's algorithm.
    
    U indices: i*m+j = i*2+j, so: 00->0, 01->1, 10->2, 11->3
    V indices: j*p+k = j*2+k, so: 00->0, 01->1, 10->2, 11->3
    W indices: k*n+i = k*2+i, so: 00->0, 01->1, 10->2, 11->3
    
    Strassen's 7 products for C = A@B where A,B are 2x2:
    M0 = (A00+A11)(B00+B11)
    M1 = (A10+A11)B00
    M2 = A00(B01-B11)
    M3 = A11(B10-B00)
    M4 = (A00+A01)B11
    M5 = (A10-A00)(B00+B01)
    M6 = (A01-A11)(B10+B11)
    
    C00 = M0+M3-M4+M6
    C01 = M2+M4
    C10 = M1+M3
    C11 = M0-M1+M2+M5
    """
    # Each product gives a rank-1 decomposition contribution.
    # Product t: U_vec * V_vec * W_vec where
    #   sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    
    # M0 = (A00+A11)(B00+B11)
    # U: A00+A11 means U[0]=1, U[3]=1 (indices 0 and 3 for i=0,j=0 and i=1,j=1)
    # V: B00+B11 means V[0]=1, V[3]=1
    # W: contributes to C00 and C11
    #   C00 += M0, C11 += M0
    #   C00 is at W index k=0,i=0 -> 0*2+0=0
    #   C11 is at W index k=1,i=1 -> 1*2+1=3
    # So W[0]=1, W[3]=1
    
    # M1 = (A10+A11)B00
    # U: A10+A11: U[2]=1, U[3]=1
    # V: B00: V[0]=1
    # W: C10 += M1, C11 += -M1
    #   C10: k=0,i=1 -> 0*2+1=1, so W[1]=1
    #   C11: k=1,i=1 -> 1*2+1=3, so W[3]=-1
    
    # M2 = A00(B01-B11)
    # U: A00: U[0]=1
    # V: B01-B11: V[1]=1, V[3]=-1
    # W: C01 += M2, C11 += M2
    #   C01: k=1,i=0 -> 1*2+0=2, W[2]=1
    #   C11: k=1,i=1 -> 3, W[3]=1
    
    # M3 = A11(B10-B00)
    # U: A11: U[3]=1
    # V: B10-B00: V[2]=1, V[0]=-1
    # W: C00 += M3, C10 += M3
    #   C00: W[0]=1
    #   C10: W[1]=1
    
    # M4 = (A00+A01)B11
    # U: A00+A01: U[0]=1, U[1]=1
    # V: B11: V[3]=1
    # W: C00 += -M4, C01 += M4
    #   C00: W[0]=-1
    #   C01: W[2]=1
    
    # M5 = (A10-A00)(B00+B01)
    # U: A10-A00: U[2]=1, U[0]=-1
    # V: B00+B01: V[0]=1, V[1]=1
    # W: C11 += M5
    #   C11: W[3]=1
    
    # M6 = (A01-A11)(B10+B11)
    # U: A01-A11: U[1]=1, U[3]=-1
    # V: B10+B11: V[2]=1, V[3]=1
    # W: C00 += M6
    #   C00: W[0]=1
    
    products = [
        # (U_vec, V_vec, W_vec)
        # M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M1
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # M2
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    return products


def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # Should equal: T[i*m+j, j*p+k, k*n+i] = 1 else 0
    
    # Use numpy for efficiency
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # This is: T = U @ diag? No, it's a tensor contraction.
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # = (U * V^T) contracted with W... 
    # Actually: for each t, outer(U[:,t], V[:,t], W[:,t]) summed
    # Efficiently: T = np.einsum('at,bt,ct->abc', U_arr, V_arr, W_arr)
    
    T = np.einsum('at,bt,ct->abc', U_arr, V_arr, W_arr)
    
    # Expected tensor
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    diff = np.max(np.abs(T - expected))
    return diff < 1e-9


def construct():
    import time
    start = time.time()
    DEADLINE = start + 100  # Safety margin
    
    n, m, p = 2, 3, 10
    
    # Split m=3 into m1=2, m2=1
    m1, m2 = 2, 1
    
    # For M_{n,m1,p} = M_{2,2,10}: tile p into 5 blocks of 2
    # Each 2x2x2 block uses Strassen (rank 7)
    # Total: 5 * 7 = 35
    
    # For M_{n,m2,p} = M_{2,1,10}: schoolbook rank = 2*1*10 = 20
    # Total: 35 + 20 = 55
    
    total_rank = 55
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    strassen_products = _strassen_2x2x2()
    
    # M_{2,2,10}: for each block of 2 columns in p
    # Block b (0 to 4): columns 2b and 2b+1 of B
    # For block b, the sub-problem is M_{2,2,2} where:
    #   A is the same 2x2 matrix (columns 0,1 of A)
    #   B_block is B[0:2, 2b:2b+2] (2x2)
    #   C_block is C[:, 2b:2b+2] (2x2)
    #
    # In the full problem:
    #   U indices: i*m+j where i in {0,1}, j in {0,1} -> 0,1,2,3
    #   V indices: j*p+k where j in {0,1}, k in {2b, 2b+1} -> 2b, 2b+1, 10+2b, 10+2b+1
    #   W indices: k*n+i where k in {2b,2b+1}, i in {0,1} -> 4b, 4b+1, 4b+2, 4b+3
    #
    # Wait, W index is k*n+i. For k=2b: 2b*2+0=4b, 2b*2+1=4b+1
    # For k=2b+1: (2b+1)*2+0=4b+2, (2b+1)*2+1=4b+3
    #
    # V index: j*p+k. For j=0: k -> 2b, 2b+1. For j=1: 10+2b, 10+2b+1.
    #
    # In the Strassen decomposition for M_{2,2,2}:
    #   U indices (local): 0,1,2,3 (i*2+j)
    #   V indices (local): 0,1,2,3 (j*2+k) where k in {0,1}
    #   W indices (local): 0,1,2,3 (k*2+i) where k in {0,1}
    #
    # Mapping local -> global for block b:
    #   U_local[i*2+j] -> U_global[i*m+j] = i*3+j  (since m=3)
    #   V_local[j*2+k] -> V_global[j*p+(2b+k)] = j*10+2b+k
    #   W_local[k*2+i] -> W_global[(2b+k)*n+i] = (2b+k)*2+i
    
    for b in range(5):
        base_rank = b * 7
        for t, (u_dict, v_dict, w_dict) in enumerate(strassen_products):
            r = base_rank + t
            # U mapping: local index li -> global index i*3+j where li=i*2+j
            for li, val in u_dict.items():
                i = li // 2
                j = li % 2
                gi = i * m + j  # i*3+j
                U[gi][r] += val
            # V mapping: local index lv -> global index j*10+(2b+k) where lv=j*2+k
            for lv, val in v_dict.items():
                j = lv // 2
                k = lv % 2
                gv = j * p + (2 * b + k)
                V[gv][r] += val
            # W mapping: local index lw -> global index (2b+k)*2+i where lw=k*2+i
            for lw, val in w_dict.items():
                k = lw // 2
                i = lw % 2
                gw = (2 * b + k) * n + i
                W[gw][r] += val
    
    # M_{2,1,10}: schoolbook
    # U indices: i*m+2 = i*3+2 for i in {0,1} -> 2, 5
    # V indices: 2*p+k = 20+k for k in {0,...,9} -> 20,21,...,29
    # W indices: k*n+i for k in {0,...,9}, i in {0,1}
    
    base_rank = 35
    col = base_rank
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1.0   # i*3+2
            V[2 * p + k][col] = 1.0    # 20+k
            W[k * n + i][col] = 1.0     # k*2+i
            col += 1
    
    # Verify
    if time.time() < DEADLINE:
        valid = _verify(U, V, W)
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
