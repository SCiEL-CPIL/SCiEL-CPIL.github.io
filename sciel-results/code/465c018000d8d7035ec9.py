import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 120.0  # Safety margin before 150s limit

    # Target: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # With n=2, m=2, p=9: U is (4, R), V is (18, R), W is (18, R)

    size_u = n * m  # 4
    size_v = m * p  # 18
    size_w = p * n  # 18

    def build_result(U, V, W):
        """Build and validate the result. Returns dict or None if invalid."""
        R = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = max(err, abs(val - 1.0))
        if err > 1e-8:
            return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_schoolbook(rank_override=None):
        """Standard schoolbook decomposition with rank n*m*p = 36."""
        R = rank_override if rank_override is not None else n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build a block decomposition:
    # Split k into blocks of size 2: k in {0,1}, {2,3}, {4,5}, {6,7}, and k=8
    # For each block of 2, use Strassen's 7-term decomposition for 2x2*2x2
    # For the last column (k=8), use schoolbook (4 terms)
    # Total rank: 4*7 + 4 = 32

    # Strassen 7-term decomposition for 2x2x2
    # C00 = M1 + M3 - M5 - M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M7
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)  -- not used in standard Strassen, let's use M7 instead
    # M7 = (A01-A11)(B10+B11)
    #
    # Let me define the 7 terms with their U, V, W coefficients:
    # U is indexed by (i,j) -> i*2+j: [A00, A01, A10, A11]
    # V is indexed by (j,k) -> j*2+k: [B00, B01, B10, B11]
    # W is indexed by (k,i) -> k*2+i: [C00, C01, C10, C11]
    
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,-1,0,1]  (C00+=1, C01-=1, C11+=1)
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,1]  (C10+=1, C11+=1)
    # M3: U=[1,0,0,0], V=[0,0,1,-1], W=[1,0,1,0]  (C00+=1, C10+=1)
    # M4: U=[0,0,0,1], V=[0,1,0,-1], W=[0,1,0,1]  (C01+=1, C11+=1)
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]  (C00-=1, C01+=1)
    # M6: U=[0,0,1,-1], V=[0,0,1,1], W=[-1,0,1,0]  (C00-=1, C10+=1) -- this is M7 in some notations
    # M7: U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,1]  (C00-=1, C11+=1)
    
    # Let me verify this decomposition numerically by constructing the tensor.
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,-1,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]),    # M2
        ([1,0,0,0], [0,0,1,-1], [1,0,1,0]),   # M3
        ([0,0,0,1], [0,1,0,-1], [0,1,0,1]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([0,0,1,-1], [0,0,1,1], [-1,0,1,0]),  # M6 (acts as M7 for C10)
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),  # M7
    ]
    
    # Verify the Strassen decomposition
    T_check = np.zeros((2,2,2))
    for u, v, w in strassen_terms:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_check[i,j,k] += u[i*2+j] * v[j*2+k] * w[k*2+i]
    
    # Expected: T[i,j,k] = 1 for all i,j,k in {0,1}
    if not np.allclose(T_check, 1.0):
        # Fallback to schoolbook if Strassen verification fails
        return build_schoolbook()
    
    def build_block_strassen():
        """Build a block decomposition with Strassen for 2-column blocks and schoolbook for the last column."""
        # 4 blocks of 2 columns each (k=0,1; k=2,3; k=4,5; k=6,7) + 1 single column (k=8)
        n_blocks = 4
        R = n_blocks * 7 + 4  # 32
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        col = 0
        for block in range(n_blocks):
            k0 = block * 2
            for u, v, w in strassen_terms:
                for idx in range(4):
                    i_l = idx // 2
                    j_l = idx % 2
                    U[i_l*m + j_l, col] = u[idx]
                    
                    # Map local V index (j*2+k_local) to global
                    # Local V is [B00, B01, B10, B11] -> indices 0,1,2,3
                    # Global V index: j*p + k0 + k_local
                    j_v = idx // 2
                    k_v = idx % 2
                    V[j_v*p + k0 + k_v, col] = v[idx]
                    
                    # Map local W index (k*2+i) to global
                    # Local W is [C00, C01, C10, C11] -> indices 0,1,2,3
                    # Global W index: (k0 + k_local)*n + i
                    k_w = idx // 2
                    i_w = idx % 2
                    W[(k0 + k_w)*n + i_w, col] = w[idx]
                col += 1
        
        # Last column (k=8)
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j*p + 8, col] = 1
                W[8*n + i, col] = 1
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result = build_block_strassen()
    
    # Validate
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    validated = build_result(U, V, W)
    if validated is not None:
        return validated
    
    # Fallback to schoolbook
    return build_schoolbook()
