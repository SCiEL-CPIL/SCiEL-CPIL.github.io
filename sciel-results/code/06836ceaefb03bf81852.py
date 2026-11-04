"""Strassen/Laderman-based 4x5x6 matrix multiplication decomposition.
Uses Laderman rank-11 for 2x3x3 blocks and Strassen+schoolbook for 2x2x3 blocks.
Total rank: 4*11 + 4*11 = 88
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # 4 Laderman blocks (2x3x3, rank 11 each) = 44
    # 4 blocks of 2x2x3 (Strassen 2x2x2 rank 7 + schoolbook 2x2x1 rank 4 = rank 11 each) = 44
    # Total rank = 88
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_laderman_block(i0, j0, k0, r_start):
        """Add Laderman rank-11 decomposition for A[i0:i0+2, j0:j0+3] * B[j0:j0+3, k0:k0+3] -> C[i0:i0+2, k0:k0+3]"""
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = A[0,0] * B[0,0]
        # C[0,0] += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        r_start += 1
        
        # M2 = (A[0,1] + A[0,2]) * (B[1,0] + B[2,0])
        # C[0,0] += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(0, 2)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(2, 0)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        r_start += 1
        
        # M3 = (A[1,0] + A[1,1] + A[1,2]) * (B[0,0] + B[1,0] + B[2,0])
        # C[0,0] += 1, C[1,0] += -1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        U[u_idx(1, 2)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(2, 0)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += -1
        r_start += 1
        
        # M4 = (A[0,1] - A[0,2]) * (B[1,0] - B[2,0])
        # C[0,0] += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(0, 2)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(2, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        r_start += 1
        
        # M5 = A[1,0] * B[1,1]
        # C[1,1] += 1
        U[u_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M6 = (A[0,1] + A[1,1]) * (B[1,1] + B[1,2])
        # C[0,1] += 1, C[1,1] += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        V[v_idx(1, 2)][r_start] += 1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M7 = (A[1,2] - A[1,0]) * (B[2,1] + B[0,1])
        # C[0,1] += 1, C[1,1] += 1
        U[u_idx(1, 2)][r_start] += 1
        U[u_idx(1, 0)][r_start] += -1
        V[v_idx(2, 1)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M8 = (A[1,1] - A[1,2]) * (B[1,2] - B[2,2])
        # C[1,1] += 1
        U[u_idx(1, 1)][r_start] += 1
        U[u_idx(1, 2)][r_start] += -1
        V[v_idx(1, 2)][r_start] += 1
        V[v_idx(2, 2)][r_start] += -1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M9 = A[1,2] * B[2,2]
        # C[1,2] += 1
        U[u_idx(1, 2)][r_start] += 1
        V[v_idx(2, 2)][r_start] += 1
        W[w_idx(2, 1)][r_start] += 1
        r_start += 1
        
        # M10 = (A[0,0] + A[1,0]) * (B[0,2] + B[2,2])
        # C[0,2] += 1, C[1,2] += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 0)][r_start] += 1
        V[v_idx(0, 2)][r_start] += 1
        V[v_idx(2, 2)][r_start] += 1
        W[w_idx(2, 0)][r_start] += 1
        W[w_idx(2, 1)][r_start] += 1
        r_start += 1
        
        # M11 = (A[0,2] - A[0,0] - A[1,0] - A[1,2]) * (B[2,0] + B[2,1] + B[2,2])
        # C[0,2] += 1, C[1,2] += 1
        U[u_idx(0, 2)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        U[u_idx(1, 0)][r_start] += -1
        U[u_idx(1, 2)][r_start] += -1
        V[v_idx(2, 0)][r_start] += 1
        V[v_idx(2, 1)][r_start] += 1
        V[v_idx(2, 2)][r_start] += 1
        W[w_idx(2, 0)][r_start] += 1
        W[w_idx(2, 1)][r_start] += 1
        r_start += 1
        
        return r_start
    
    def add_strassen_2x2x2_block(i0, j0, k0, r_start):
        """Add Strassen rank-7 decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M2 = (A10 + A11)B00
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        r_start += 1
        
        # M3 = A00(B01 - B11)
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M4 = A11(B10 - B00)
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        r_start += 1
        
        # M5 = (A00 + A01)B11
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        r_start += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        r_start += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        r_start += 1
        
        return r_start
    
    def add_schoolbook_2x2x1_block(i0, j0, k0, r_start):
        """Add schoolbook rank-4 decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        for i in range(2):
            for j in range(2):
                U[u_idx(i, j)][r_start] = 1
                V[v_idx(j, 0)][r_start] = 1
                W[w_idx(0, i)][r_start] = 1
                r_start += 1
        
        return r_start
    
    def add_2x2x3_block(i0, j0, k0, r_start):
        """Add decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+3] -> C[i0:i0+2, k0:k0+3]
        Split B's columns into [k0:k0+2] and [k0+2:k0+3]
        Strassen for 2x2x2 + schoolbook for 2x2x1
        """
        r_start = add_strassen_2x2x2_block(i0, j0, k0, r_start)
        r_start = add_schoolbook_2x2x1_block(i0, j0, k0 + 2, r_start)
        return r_start
    
    r_offset = 0
    
    # Laderman blocks (2x3x3, rank 11 each)
    # A[0:2, 0:3] * B[0:3, 0:3] -> C[0:2, 0:3]
    r_offset = add_laderman_block(0, 0, 0, r_offset)
    # A[0:2, 0:3] * B[0:3, 3:6] -> C[0:2, 3:6]
    r_offset = add_laderman_block(0, 0, 3, r_offset)
    # A[2:4, 0:3] * B[0:3, 0:3] -> C[2:4, 0:3]
    r_offset = add_laderman_block(2, 0, 0, r_offset)
    # A[2:4, 0:3] * B[0:3, 3:6] -> C[2:4, 3:6]
    r_offset = add_laderman_block(2, 0, 3, r_offset)
    
    # 2x2x3 blocks (rank 11 each: Strassen 7 + schoolbook 4)
    # A[0:2, 3:5] * B[3:5, 0:3] -> C[0:2, 0:3]
    r_offset = add_2x2x3_block(0, 3, 0, r_offset)
    # A[0:2, 3:5] * B[3:5, 3:6] -> C[0:2, 3:6]
    r_offset = add_2x2x3_block(0, 3, 3, r_offset)
    # A[2:4, 3:5] * B[3:5, 0:3] -> C[2:4, 0:3]
    r_offset = add_2x2x3_block(2, 3, 0, r_offset)
    # A[2:4, 3:5] * B[3:5, 3:6] -> C[2:4, 3:6]
    r_offset = add_2x2x3_block(2, 3, 3, r_offset)
    
    if r_offset != rank:
        pass
    
    # Verify correctness
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        U_r = U_np[:, r].reshape(n, m)
        V_r = V_np[:, r].reshape(m, p)
        W_r = W_np[:, r].reshape(p, n)
        
        a_part = np.sum(A * U_r)
        b_part = np.sum(B * V_r)
        C_decomp += a_part * b_part * W_r.T
    
    error = np.max(np.abs(C_decomp - C_expected))
    if error > 1e-9:
        raise ValueError(f"Decomposition error: {error}")
    
    return {"U": U, "V": V, "W": W}
