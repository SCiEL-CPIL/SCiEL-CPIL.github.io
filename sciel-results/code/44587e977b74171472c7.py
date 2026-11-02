import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U, V, W shapes: (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M1 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M2 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M3 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M4 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W[k*2+i] for C[i,k]
    # C00 = M0 + M3 - M5 - M6
    W[0, 0] = 1; W[0, 3] = 1; W[0, 5] = -1; W[0, 6] = -1
    # C01 = M4 + M5
    W[1, 4] = 1; W[1, 5] = 1
    # C10 = M1 + M2
    W[2, 1] = 1; W[2, 2] = 1
    # C11 = M0 - M1 + M2 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_rank49_4x4x4():
    """
    Build rank 49 decomposition for 4x4x4 using recursive Strassen.
    """
    U22, V22, W22 = _strassen_2x2x2()
    r = 49
    U = np.zeros((16, r), dtype=float)
    V = np.zeros((16, r), dtype=float)
    W = np.zeros((16, r), dtype=float)
    
    # Strassen block definitions for 4x4
    # s=0: A_blocks=[(0,0,1), (1,1,1)], B_blocks=[(0,0,1), (1,1,1)], C_blocks=[(0,0,1), (0,1,0), (1,0,0), (1,1,1)]
    # We define for each s:
    #   A_comb: list of (block_row, block_col, coeff)
    #   B_comb: list of (block_row, block_col, coeff)
    #   C_comb: list of (block_row, block_col, coeff)
    
    strassen_blocks = [
        # M0
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (0,1,0), (1,0,0), (1,1,1)] # Wait, C is sum of Ms.
            # Actually, C_comb should be the coefficients of M_s in each C_block.
            # C00 += M0, C11 += M0.
            # So for s=0: C00 coeff 1, C11 coeff 1.
        },
        # M1
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)] # C10 += M1, C11 -= M1
        },
        # M2
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(1,0,1), (1,1,1)] # C10 += M2, C11 += M2
        },
        # M3
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(0,0,1), (1,1,1)] # C00 += M3, C11 += M3
        },
        # M4
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,1,1)] # C01 += M4
        },
        # M5
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(0,0,-1), (0,1,1)] # C00 -= M5, C01 += M5
        },
        # M6
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,-1), (1,1,1)] # C00 -= M6, C11 += M6
        }
    ]
    
    # Correct C coefficients based on standard Strassen:
    # C00 = M0 + M3 - M5 - M6
    # C01 = M4 + M5
    # C10 = M1 + M2
    # C11 = M0 - M1 + M2 + M6
    
    # Re-define C coefficients properly
    c_coeffs = [
        {(0,0): 1, (1,1): 1}, # M0
        {(1,0): 1, (1,1): -1}, # M1
        {(1,0): 1, (1,1): 1}, # M2
        {(0,0): 1, (1,1): 1}, # M3
        {(0,1): 1}, # M4
        {(0,0): -1, (0,1): 1}, # M5
        {(0,0): -1, (1,1): 1} # M6
    ]
    
    for s in range(7):
        A_comb = strassen_blocks[s]['A']
        B_comb = strassen_blocks[s]['B']
        
        for t in range(7):
            col = s * 7 + t
            
            # Fill U
            for (br, bc, coeff) in A_comb:
                # Block A[br, bc] corresponds to global rows 2*br..2*br+1, cols 2*bc..2*bc+1
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2 * br + i_local
                        j_global = 2 * bc + j_local
                        # U22[i_local*2 + j_local, t] is the coefficient for A[i_local, j_local] in M_s
                        u_val = U22[i_local * 2 + j_local, t]
                        if u_val != 0:
                            U[i_global * 4 + j_global, col] += coeff * u_val
            
            # Fill V
            for (br, bc, coeff) in B_comb:
                # Block B[br, bc] corresponds to global rows 2*br..2*br+1, cols 2*bc..2*bc+1
                for i_local in range(2): # row of B
                    for j_local in range(2): # col of B
                        i_global = 2 * br + i_local
                        j_global = 2 * bc + j_local
                        v_val = V22[i_local * 2 + j_local, t]
                        if v_val != 0:
                            V[i_global * 4 + j_global, col] += coeff * v_val
            
            # Fill W
            for (cr, cc, coeff) in c_coeffs[s].items():
                # Block C[cr, cc] corresponds to global rows 2*cr..2*cr+1, cols 2*cc..2*cc+1
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2 * cr + i_local
                        j_global = 2 * cc + j_local
                        # W index is k*n + i. Here n=4.
                        # W[j_global * 4 + i_global, col]
                        w_val = W22[j_local * 2 + i_local, t]
                        if w_val != 0:
                            W[j_global * 4 + i_global, col] += coeff * w_val
                            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, trials=10, atol=1e-10):
    """Verify a tensor decomposition is correct."""
    r = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for t in range(r):
            Ut = U[:, t].reshape(n, m)
            Vt = V[:, t].reshape(m, p)
            Wt = W[:, t].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += Wt[k, i] * np.dot(Ut[i, :], Vt[:, k])
        
        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def time_remaining():
        return 145 - (time.time() - start_time)
        
    try:
        U44, V44, W44 = _build_rank49_4x4x4()
        rank_44 = U44.shape[1]
        
        if not _verify_decomposition(U44, V44, W44, 4, 4, 4, trials=5):
            raise ValueError("4x4x4 verification failed")
            
        rank_school = 4 * 4 * 3
        total_rank = rank_44 + rank_school
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        U[:, :rank_44] = U44
        for j in range(4):
            for k in range(4):
                V[j*7+k, :rank_44] = V44[j*4+k, :rank_44]
        W[:16, :rank_44] = W44
        
        col = rank_44
        for i in range(n):
            for j in range(m):
                for k in range(4, p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    
        if col != total_rank:
            raise ValueError(f"Column mismatch: {col} vs {total_rank}")
            
        if not _verify_decomposition(U, V, W, n, m, p, trials=5):
            raise ValueError("Full decomposition verification failed")
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception:
        pass
        
    # Fallback
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
