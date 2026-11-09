import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A (j=0,1), B (j=0,1; k=0,1), C (k=0,1)
    # Strassen 2x2x2 decomposition
    # U_s: 4x7, rows correspond to A00, A01, A10, A11
    # V_s: 4x7, rows correspond to B00, B01, B10, B11
    # W_s: 4x7, rows correspond to C00, C01, C10, C11
    
    # Standard Strassen decomposition for 2x2 matrices:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U matrix (for A = [[A00, A01], [A10, A11]])
    # Row 0: A00 -> coefficients in [m1, m2, m3, m4, m5, m6, m7]
    #   m1: 1, m2: 0, m3: 1, m4: 0, m5: 1, m6: -1, m7: 0
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 0, 0, 1],    # A01
        [0, 1, 0, 0, 1, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    
    # V matrix (for B = [[B00, B01], [B10, B11]])
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    
    # W matrix (for C = [[C00, C01], [C10, C11]])
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    def add_strassen_block(u_offset, v_offset, w_offset, rank_offset):
        # u_offset: row index of A00 in U
        # v_offset: row index of B00 in V
        # w_offset: row index of C00 in W
        
        # A: A00 at u_offset, A01 at u_offset+1, A10 at u_offset+3, A11 at u_offset+4
        u_map = [u_offset, u_offset + 1, u_offset + 3, u_offset + 4]
        
        # B: B00 at v_offset, B01 at v_offset+1, B10 at v_offset+4, B11 at v_offset+5
        v_map = [v_offset, v_offset + 1, v_offset + 4, v_offset + 5]
        
        # C: C00 at w_offset, C01 at w_offset+1, C10 at w_offset+2, C11 at w_offset+3
        w_map = [w_offset, w_offset + 1, w_offset + 2, w_offset + 3]
        
        for r in range(7):
            col = rank_offset + r
            for s_idx in range(4):
                U[u_map[s_idx]][col] += U_s[s_idx][r]
                V[v_map[s_idx]][col] += V_s[s_idx][r]
                W[w_map[s_idx]][col] += W_s[s_idx][r]
    
    # Block 1: A rows (i=0,1), cols (j=0,1); B rows (j=0,1), cols (k=0,1); C rows (k=0,1), cols (i=0,1)
    # u_offset = 0 (A00 at U[0])
    # v_offset = 0 (B00 at V[0])
    # w_offset = 0 (C00 at W[0])
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A rows (i=0,1), cols (j=0,1); B rows (j=0,1), cols (k=2,3); C rows (k=2,3), cols (i=0,1)
    # u_offset = 0 (same A)
    # v_offset = 2 (B00 at V[2], which is j=0, k=2)
    # w_offset = 4 (C00 at W[4], which is k=2, i=0)
    add_strassen_block(0, 2, 4, 7)
    
    # Block 3: A rows (i=0,1), col (j=2); B row (j=2), cols (k=0..3); C rows (k=0..3), cols (i=0,1)
    # This is a rank-1 decomposition: C_ki = A_i2 * B_2k
    col = 11
    # A: i=0, j=2 -> U[2, 11] = 1; i=1, j=2 -> U[5, 11] = 1
    U[2][col] = 1
    U[5][col] = 1
    
    # B: j=2, k=0..3 -> V[2*4+k, 11] = 1
    for k in range(4):
        V[2*4 + k][col] = 1
        
    # C: k=0..3, i=0..1 -> W[k*2+i, 11] = 1
    for k in range(4):
        for i in range(2):
            W[k*2 + i][col] = 1
    
    # Verify the decomposition
    # C = A * B
    # C[k*n+i, j*m+k] = sum_j A[i*m+j, j*p+k] * ... wait, let me re-check the convention
    # The output factor follows the official trace convention W[k*n+i, r]
    # So W is (p*n, rank), indexed as W[k*n + i, r]
    # U is (n*m, rank), indexed as U[i*m + j, r]
    # V is (m*p, rank), indexed as V[j*p + k, r]
    
    # The tensor identity is:
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ... no, that's not right
    
    # Actually, for matrix multiplication C = A*B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # The decomposition is:
    # C[i, k] = sum_r (sum_j U[i*m+j, r] * B[j, k]) * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, that's not standard. Let me think again.
    
    # The standard tensor decomposition for matrix multiplication is:
    # C[i, k] = sum_r U[i, r] * V[r, k]  ... no, that's for a single matrix product
    
    # For the tensor [A, B, C], the decomposition is:
    # T[i, j, k, l, m, n] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Where T represents the matrix multiplication tensor:
    # T[i, j, k, l, m, n] = 1 if (i=m and j=l and k=n) or similar, representing A[m,l]*B[l,n]=C[m,n]
    
    # Actually, the correct form is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ... this doesn't look right
    
    # Let me just verify numerically
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for r in range(rank):
                # Sum over j
                for j in range(m):
                    total += U[i*m + j][r] * V[j*p + k][r]
            C_computed[i, k] = total * W[k*n + i][0] if rank > 0 else 0
    
    # Wait, I'm confusing myself. Let me re-read the problem statement.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    
    # I think the decomposition is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    
    # No, that's still not standard. Let me try a different interpretation.
    
    # Actually, I think the correct interpretation is:
    # The tensor is T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[A_idx, B_idx, C_idx] = 1 if the multiplication is valid
    
    # For matrix multiplication: C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor has indices: (i, j) for A, (j, k) for B, (i, k) for C
    # T[(i,j), (j,k), (i,k)] = 1
    
    # The decomposition is:
    # T[(i,j), (j,k), (i,k)] = sum_r U[(i,j), r] * V[(j,k), r] * W[(i,k), r]
    
    # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  for all i, j, k
    
    # And for invalid indices, T = 0 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let me verify this
    max_error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                expected = 1.0
                max_error = max(max_error, abs(total - expected))
    
    # The max error should be 0 if the decomposition is correct
    # If it's not 0, the decomposition is wrong
    
    if max_error > 1e-10:
        # Fall back to a known correct decomposition
        # Use the trivial rank-24 decomposition
        rank = 24
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m + j][r] = 1
                    V[j*p + k][r] = 1
                    W[k*n + i][r] = 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
