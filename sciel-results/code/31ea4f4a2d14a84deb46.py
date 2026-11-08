import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    def strassen_decomposition():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
        V = [[0.0] * rank for _ in range(m * p)]  # 16 rows
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows
        
        def add_strassen(col_offset, A_block, B_block, C_block_W_indices):
            a00_idx = A_block[(0,0)]
            a01_idx = A_block[(0,1)]
            a10_idx = A_block[(1,0)]
            a11_idx = A_block[(1,1)]
            
            b00_idx = B_block[(0,0)]
            b01_idx = B_block[(0,1)]
            b10_idx = B_block[(1,0)]
            b11_idx = B_block[(1,1)]
            
            c00_w = C_block_W_indices[(0,0)]
            c01_w = C_block_W_indices[(0,1)]
            c10_w = C_block_W_indices[(1,0)]
            c11_w = C_block_W_indices[(1,1)]
            
            # M1 = (A00+A11)*(B00+B11)
            U[a00_idx][col_offset] += 1.0
            U[a11_idx][col_offset] += 1.0
            V[b00_idx][col_offset] += 1.0
            V[b11_idx][col_offset] += 1.0
            W[c00_w][col_offset] += 1.0
            W[c11_w][col_offset] += 1.0
            
            # M2 = (A10+A11)*B00
            U[a10_idx][col_offset+1] += 1.0
            U[a11_idx][col_offset+1] += 1.0
            V[b00_idx][col_offset+1] += 1.0
            W[c10_w][col_offset+1] += 1.0
            
            # M3 = A00*(B10-B11)
            U[a00_idx][col_offset+2] += 1.0
            V[b10_idx][col_offset+2] += 1.0
            V[b11_idx][col_offset+2] += -1.0
            W[c01_w][col_offset+2] += 1.0
            W[c11_w][col_offset+2] += 1.0
            
            # M4 = A11*(B01-B00)
            U[a11_idx][col_offset+3] += 1.0
            V[b01_idx][col_offset+3] += 1.0
            V[b00_idx][col_offset+3] += -1.0
            W[c00_w][col_offset+3] += 1.0
            W[c10_w][col_offset+3] += 1.0
            
            # M5 = (A00+A01)*B11
            U[a00_idx][col_offset+4] += 1.0
            U[a01_idx][col_offset+4] += 1.0
            V[b11_idx][col_offset+4] += 1.0
            W[c00_w][col_offset+4] += -1.0
            W[c01_w][col_offset+4] += 1.0
            
            # M6 = (A10-A00)*(B00+B01)
            U[a10_idx][col_offset+5] += 1.0
            U[a00_idx][col_offset+5] += -1.0
            V[b00_idx][col_offset+5] += 1.0
            V[b01_idx][col_offset+5] += 1.0
            W[c11_w][col_offset+5] += 1.0
            
            # M7 = (A01-A11)*(B10+B11)
            U[a01_idx][col_offset+6] += 1.0
            U[a11_idx][col_offset+6] += -1.0
            V[b10_idx][col_offset+6] += 1.0
            V[b11_idx][col_offset+6] += 1.0
            W[c00_w][col_offset+6] += 1.0
            W[c11_w][col_offset+6] += -1.0
            
            return col_offset + 7
        
        A1 = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
        A2 = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
        
        B11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
        B12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
        B21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
        B22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
        
        C1_W = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        C2_W = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        col = 0
        col = add_strassen(col, A1, B11, C1_W)
        col = add_strassen(col, A2, B21, C1_W)
        col = add_strassen(col, A1, B12, C2_W)
        col = add_strassen(col, A2, B22, C2_W)
        
        return {"U": U, "V": V, "W": W}

    def verify_solution(U, V, W):
        # Quick verification with integer inputs to avoid floating point issues
        # Test with A = [[1,0,1,0],[0,1,0,1]] and B = Identity-like
        # Actually, just test one specific case
        A = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
        B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        C_true = A @ B
        
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        C_tensor = np.zeros((n, p))
        for r in range(rank):
            a_proj = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_proj[i] += A[i, j] * U[i * m + j, r]
            b_proj = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_proj[k] += B[j, k] * V[j * p + k, r]
            for i in range(n):
                for k in range(p):
                    C_tensor[i, k] += W[k * n + i, r] * a_proj[i] * b_proj[k]
        
        return np.allclose(C_tensor, C_true, atol=1e-9)

    sol = strassen_decomposition()
    if verify_solution(sol["U"], sol["V"], sol["W"]):
        return sol
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
