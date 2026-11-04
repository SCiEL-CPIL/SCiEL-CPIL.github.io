import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Kronecker product decomposition:
    # Split 6x6 matrices into 3x3 blocks of 2x2 sub-matrices.
    # Outer level: 3x3 schoolbook (27 terms)
    # Inner level: 2x2 Strassen (7 terms)
    # Total rank: 27 * 7 = 189
    #
    # Trace convention: W[k*n + i, r] where k is column index, i is row index
    # So C[i,j] is at W_flat[j*6 + i]
    
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=int)  # 36 x 189
    V = np.zeros((m * p, rank), dtype=int)  # 36 x 189
    W = np.zeros((p * n, rank), dtype=int)  # 36 x 189
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # For A_block = [a00, a01; a10, a11], B_block = [b00, b01; b10, b11]:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # A block (i,k): rows 2*i..2*i+1, cols 2*k..2*k+1 in A (6x6)
                    # Flattened index: row*6 + col
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    # B block (k,j): rows 2*k..2*k+1, cols 2*j..2*j+1 in B (6x6)
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    # C block (i,j): rows 2*i..2*i+1, cols 2*j..2*j+1 in C (6x6)
                    # Trace convention: C[r,c] is at W_flat[c*6 + r]
                    c00 = (2*j) * 6 + (2*i)      # C[2*i, 2*j]
                    c01 = (2*j+1) * 6 + (2*i)    # C[2*i, 2*j+1]
                    c10 = (2*j) * 6 + (2*i+1)    # C[2*i+1, 2*j]
                    c11 = (2*j+1) * 6 + (2*i+1)  # C[2*i+1, 2*j+1]
                    
                    if s == 0:  # M1 = (a00 + a11)(b00 + b11)
                        U[a00, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1   # C00 += M1
                        W[c11, idx] = 1   # C11 += M1
                    elif s == 1:  # M2 = (a10 + a11)b00
                        U[a10, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1   # C10 += M2
                        W[c11, idx] = -1  # C11 -= M2
                    elif s == 2:  # M3 = a00(b01 - b11)
                        U[a00, idx] = 1
                        V[b01, idx] = 1
                        V[b11, idx] = -1
                        W[c01, idx] = 1   # C01 += M3
                        W[c11, idx] = 1   # C11 += M3
                    elif s == 3:  # M4 = a11(b10 - b00)
                        U[a11, idx] = 1
                        V[b10, idx] = 1
                        V[b00, idx] = -1
                        W[c00, idx] = 1   # C00 += M4
                        W[c10, idx] = 1   # C10 += M4
                    elif s == 4:  # M5 = (a00 + a01)b11
                        U[a00, idx] = 1
                        U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1  # C00 -= M5
                        W[c01, idx] = 1   # C01 += M5
                    elif s == 5:  # M6 = (a10 - a00)(b00 + b01)
                        U[a10, idx] = 1
                        U[a00, idx] = -1
                        V[b00, idx] = 1
                        V[b01, idx] = 1
                        W[c11, idx] = 1   # C11 += M6
                    elif s == 6:  # M7 = (a01 - a11)(b10 + b11)
                        U[a01, idx] = 1
                        U[a11, idx] = -1
                        V[b10, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1   # C00 += M7
                    
                    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verify the decomposition is correct
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        
        C_expected = A_test @ B_test
        
        # Verify: C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * V[k*p+j, r])
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        # Use einsum: C[i,j] = sum_{r,k} W[j,n+i,r] * U[i,m+k,r] * V[k,p+j,r]
        # Reshape for einsum
        U_3d = U_arr.reshape(n, m, rank)   # U[i, k, r]
        V_3d = V_arr.reshape(m, p, rank)   # V[k, j, r]
        W_3d = W_arr.reshape(p, n, rank)   # W[j, i, r]
        
        # prod[i,j,r] = sum_k U[i,k,r] * V[k,j,r]
        prod = np.einsum('ikr,kjr->ijr', U_3d, V_3d)
        # C[i,j] = sum_r prod[i,j,r] * W[j,i,r]
        C_computed = np.einsum('ijr,jir->ij', prod, W_3d)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
