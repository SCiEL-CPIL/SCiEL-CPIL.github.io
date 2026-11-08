import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    # Output convention: W[k*n + i, r] where i is row of C, k is col of C
    
    def schoolbook_solution():
        rank = n * m * p  # 32
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
    
    def verify_solution(U, V, W, max_samples=200):
        """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,n+i,r] == A[i,j]*B[j,k] 
        for the matrix multiplication tensor."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = A[i,j]*B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, W index is k*n+i per the convention W[k*n+i, r]
        
        np.random.seed(42)
        for _ in range(max_samples):
            # Random A and B
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B  # n x p = 2 x 4
            
            # Compute C via tensor: C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,k]*V[j*p+k,r]) * W[k*n+i,r]
            # Actually the standard form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])
            # But that's not right. The tensor decomposition is:
            # T = sum_r U_r ⊗ V_r ⊗ W_r
            # where T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # And the matrix multiplication says: C[i,k] = sum_j A[i,j]*B[j,k]
            # So we need: for all A,B, sum_j A[i,j]*B[j,k] = sum_r (sum_j A[i,j]*U[i*m+j,r])*(sum_j B[j,k]*V[j*p+k,r])*W[k*n+i,r]
            # Hmm, that's not quite the standard tensor form either.
            # 
            # Let me reconsider. The matrix multiplication tensor M has entries M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor decomposition: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (using W[k*n+i,r])
            # Then: C[i,k] = sum_{j} A[i,j]*B[j,k]*M[i,j,k] = sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r]) * W[k,i,r]
            # Wait no. If M[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r], then:
            # C[i,k] = sum_j A[i,j]*B[j,k]*M[i,j,k]
            #         = sum_j A[i,j]*B[j,k]*sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
            #         = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r])
            
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
            
            if not np.allclose(C_tensor, C_true, atol=1e-9, rtol=1e-9):
                return False
        return True
    
    def strassen_decomposition():
        """
        Rank-28 decomposition using Strassen for 2x2 blocks.
        
        A is 2x4, split as A = [A1 | A2] where A1, A2 are 2x2
        B is 4x4, split as B = [[B11, B12], [B21, B22]] where each is 2x2
        C is 2x4, split as C = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
        
        For C1 = A1*B11 + A2*B21:
          Use Strassen for A1*B11 (7 products) + Strassen for A2*B21 (7 products) = 14 products
        For C2 = A1*B12 + A2*B22:
          Use Strassen for A1*B12 (7 products) + Strassen for A2*B22 (7 products) = 14 products
        Total: 28 products
        """
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
        V = [[0.0] * rank for _ in range(m * p)]  # 16 rows
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows
        
        # Strassen for X*Y (both 2x2): 7 products
        # M1 = (X00+X11)*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = X00*(Y10-Y11)
        # M4 = X11*(Y01-Y00)
        # M5 = (X00+X01)*Y11
        # M6 = (X10-X00)*(Y00+Y01)
        # M7 = (X01-X11)*(Y10+Y11)
        # Z00 = M1+M4-M5+M7, Z01 = M3+M5, Z10 = M2+M4, Z11 = M1-M2+M3+M6
        
        # Indexing for A (2x4): A[i,j] where i in {0,1}, j in {0,1,2,3}
        # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
        # For A1: elements are A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
        # For A2: elements are A[0,2], A[0,3], A[1,2], A[1,3] -> indices 4,5,6,7
        
        # Indexing for B (4x4): B[j,k] where j in {0..3}, k in {0..3}
        # B11 = B[0:2, 0:2]: B[0,0],B[0,1],B[1,0],B[1,1] -> indices 0,1,4,5
        # B12 = B[0:2, 2:4]: B[0,2],B[0,3],B[1,2],B[1,3] -> indices 2,3,6,7
        # B21 = B[2:4, 0:2]: B[2,0],B[2,1],B[3,0],B[3,1] -> indices 8,9,12,13
        # B22 = B[2:4, 2:4]: B[2,2],B[2,3],B[3,2],B[3,3] -> indices 10,11,14,15
        
        # Indexing for C (2x4) -> W index is k*n+i = k*2+i
        # C1 = C[0:2, 0:2]: C[0,0],C[0,1],C[1,0],C[1,1] -> W indices: 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3
        #   i.e., W[k*n+i]: C[i,k] -> W[k*2+i]
        #   C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
        # C2 = C[0:2, 2:4]: C[0,2],C[0,3],C[1,2],C[1,3]
        #   C[0,2]->W[2*2+0]=W[4], C[0,3]->W[3*2+0]=W[6], C[1,2]->W[2*2+1]=W[5], C[1,3]->W[3*2+1]=W[7]
        
        def add_strassen(col_offset, A_block, B_block, C_block_W_indices):
            """
            Add Strassen products for A_block * B_block to C_block.
            A_block: dict mapping (i,j) -> U index (i*m+j)
            B_block: dict mapping (j,k) -> V index (j*p+k)
            C_block_W_indices: dict mapping (i,k) -> W index (k*n+i)
            """
            # A_block elements: A00, A01, A10, A11
            a00_idx = A_block[(0,0)]
            a01_idx = A_block[(0,1)]
            a10_idx = A_block[(1,0)]
            a11_idx = A_block[(1,1)]
            
            # B_block elements: B00, B01, B10, B11
            b00_idx = B_block[(0,0)]
            b01_idx = B_block[(0,1)]
            b10_idx = B_block[(1,0)]
            b11_idx = B_block[(1,1)]
            
            # C_block W indices: C00, C01, C10, C11
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
        
        # A1 = A[0:2, 0:2]: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        A1 = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
        # A2 = A[0:2, 2:4]: (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7
        A2 = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
        
        # B11 = B[0:2, 0:2]: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
        B11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
        # B12 = B[0:2, 2:4]: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
        B12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
        # B21 = B[2:4, 0:2]: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
        B21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
        # B22 = B[2:4, 2:4]: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
        B22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
        
        # C1 W indices: C[i,k] -> W[k*2+i]
        # C1[0,0]->W[0], C1[0,1]->W[2], C1[1,0]->W[1], C1[1,1]->W[3]
        C1_W = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        # C2 W indices: C2[0,0]->W[4], C2[0,1]->W[6], C2[1,0]->W[5], C2[1,1]->W[7]
        C2_W = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        # C1 = A1*B11 + A2*B21
        col = 0
        col = add_strassen(col, A1, B11, C1_W)  # products 0-6
        col = add_strassen(col, A2, B21, C1_W)  # products 7-13
        
        # C2 = A1*B12 + A2*B22
        col = add_strassen(col, A1, B12, C2_W)  # products 14-20
        col = add_strassen(col, A2, B22, C2_W)  # products 21-27
        
        return {"U": U, "V": V, "W": W}
    
    # First try Strassen rank-28
    try:
        strassen_sol = strassen_decomposition()
        if verify_solution(strassen_sol["U"], strassen_sol["V"], strassen_sol["W"], max_samples=100):
            return strassen_sol
    except Exception:
        pass
    
    # Fallback to schoolbook
    return schoolbook_solution()
# EVOLVE-BLOCK-END
