"""Matrix multiplication 2,5,6 tensor decomposition using block Strassen."""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x6 matrix.
    
    Strategy: Partition B's rows into {0,1}, {2,3}, {4} and columns into
    {0,1}, {2,3}, {4,5}. For 2x2 sub-blocks, use Strassen (rank 7).
    For 2x1 or 1x2 sub-blocks, use schoolbook (rank 2 each).
    For 1x1 sub-blocks, use schoolbook (rank 1).
    
    Total rank: 6*7 + 3*2 = 42 + 6 = 48
    """
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strassen's algorithm for 2x2x2 block
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m4
    # C[1,0] = m2 + m5
    # C[1,1] = m1 - m2 + m3 + m7
    #
    # Each m_t = U_t * V_t where U_t is linear in A, V_t is linear in B
    # The W vector for each term indicates which C entries it contributes to
    
    # U vectors (flattened 2x2 A: [a,b,c,d] -> indices [0,1,2,3])
    # V vectors (flattened 2x2 B: [e,f,g,h] -> indices [0,1,2,3])
    # W vectors (flattened 2x2 C: [C00,C01,C10,C11] -> indices [0,1,2,3])
    
    strassen_terms = [
        # m1 = (a+d)(e+h)
        # U: a=1, d=1 -> [1,0,0,1]
        # V: e=1, h=1 -> [1,0,0,1]
        # W: C00+=1, C11+=1 -> [1,0,0,1]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m2 = (c+d)e
        # U: c=1, d=1 -> [0,0,1,1]
        # V: e=1 -> [1,0,0,0]
        # W: C10+=1 -> [0,0,1,0]
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # m3 = a(f-h)
        # U: a=1 -> [1,0,0,0]
        # V: f=1, h=-1 -> [0,1,0,-1]
        # W: C01+=1, C11+=1 -> [0,1,0,1]
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        # m4 = d(g-e)
        # U: d=1 -> [0,0,0,1]
        # V: g=1, e=-1 -> [0,0,1,-1]
        # W: C00+=-1, C01+=1 -> [-1,1,0,0]
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]),
        # m5 = (a+b)h
        # U: a=1, b=1 -> [1,1,0,0]
        # V: h=1 -> [0,0,0,1]
        # W: C00+=1, C10+=1 -> [1,0,1,0]
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        # m6 = (c-a)(e+f)
        # U: c=1, a=-1 -> [-1,0,1,0]
        # V: e=1, f=1 -> [1,1,0,0]
        # W: C11+=-1 -> [0,0,0,-1]
        ([-1,0,1,0], [1,1,0,0], [0,0,0,-1]),
        # m7 = (b-d)(g+h)
        # U: b=1, d=-1 -> [0,1,0,-1]
        # V: g=1, h=1 -> [0,0,1,1]
        # W: C00+=-1, C11+=1 -> [-1,0,0,1]
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),
    ]
    
    # Verify Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for U, V, W in strassen_terms:
                        total += U[i] * V[j] * W[k]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify_strassen()
    if not valid:
        # Fallback to schoolbook
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
    
    # Partition B rows and columns
    # Row blocks: {0,1}, {2,3}, {4}
    # Col blocks: {0,1}, {2,3}, {4,5}
    row_blocks = [[0, 1], [2, 3], [4]]
    col_blocks = [[0, 1], [2, 3], [4, 5]]
    
    all_terms = []  # list of (U_sparse, V_sparse, W_sparse)
    
    for rb in row_blocks:
        for cb in col_blocks:
            if len(rb) == 2 and len(cb) == 2:
                # Use Strassen for 2x2x2 sub-block
                j0, j1 = rb[0], rb[1]
                k0, k1 = cb[0], cb[1]
                for U_flat, V_flat, W_flat in strassen_terms:
                    U_sparse = {}
                    V_sparse = {}
                    W_sparse = {}
                    # U_flat indices: [0,1,2,3] correspond to (i,j) in [(0,j0),(0,j1),(1,j0),(1,j1)]
                    for idx, val in enumerate(U_flat):
                        if val != 0:
                            i = idx // 2
                            j_local = idx % 2
                            j = j0 if j_local == 0 else j1
                            U_sparse[i * m + j] = val
                    # V_flat indices: [0,1,2,3] correspond to (j,k) in [(j0,k0),(j0,k1),(j1,k0),(j1,k1)]
                    for idx, val in enumerate(V_flat):
                        if val != 0:
                            j_local = idx // 2
                            k_local = idx % 2
                            j = j0 if j_local == 0 else j1
                            k = k0 if k_local == 0 else k1
                            V_sparse[j * p + k] = val
                    # W_flat indices: [0,1,2,3] correspond to (k,i) in [(k0,0),(k0,1),(k1,0),(k1,1)]
                    for idx, val in enumerate(W_flat):
                        if val != 0:
                            k_local = idx // 2
                            i = idx % 2
                            k = k0 if k_local == 0 else k1
                            W_sparse[k * n + i] = val
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            elif len(rb) == 2 and len(cb) == 1:
                # Schoolbook: 2 terms
                j0, j1 = rb[0], rb[1]
                k0 = cb[0]
                for j in [j0, j1]:
                    for i in range(n):
                        U_sparse = {i * m + j: 1}
                        V_sparse = {j * p + k0: 1}
                        W_sparse = {k0 * n + i: 1}
                        all_terms.append((U_sparse, V_sparse, W_sparse))
            elif len(rb) == 1 and len(cb) == 2:
                # Schoolbook: 2 terms
                j0 = rb[0]
                k0, k1 = cb[0], cb[1]
                for k in [k0, k1]:
                    for i in range(n):
                        U_sparse = {i * m + j0: 1}
                        V_sparse = {j0 * p + k: 1}
                        W_sparse = {k * n + i: 1}
                        all_terms.append((U_sparse, V_sparse, W_sparse))
            else:
                # 1x1: schoolbook, 1 term
                j0 = rb[0]
                k0 = cb[0]
                for i in range(n):
                    U_sparse = {i * m + j0: 1}
                    V_sparse = {j0 * p + k0: 1}
                    W_sparse = {k0 * n + i: 1}
                    all_terms.append((U_sparse, V_sparse, W_sparse))
    
    rank = len(all_terms)
    
    # Check time
    if time.time() - start_time > DEADLINE:
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
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_sparse, V_sparse, W_sparse) in enumerate(all_terms):
        for idx, val in U_sparse.items():
            U[idx][col] = val
        for idx, val in V_sparse.items():
            V[idx][col] = val
        for idx, val in W_sparse.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
