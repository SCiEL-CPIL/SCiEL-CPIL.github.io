import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Strategy: 6x6 = 3x3 grid of 2x2 blocks
    # For each (i,k,j) block position, apply Strassen (rank 7) on 2x2 sub-blocks
    # Total rank = 3*3*3*7 = 189
    #
    # W convention: W[j*n+i, r] where C[i,j] is the output element
    # U[i*m+k, r], V[k*p+j, r]
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # 2x2 sub-block indices
                a00 = (2*i) * 6 + 2*k
                a01 = (2*i) * 6 + 2*k + 1
                a10 = (2*i+1) * 6 + 2*k
                a11 = (2*i+1) * 6 + 2*k + 1
                
                b00 = (2*k) * 6 + 2*j
                b01 = (2*k) * 6 + 2*j + 1
                b10 = (2*k+1) * 6 + 2*j
                b11 = (2*k+1) * 6 + 2*j + 1
                
                # C output: C[row, col] where row = 2i or 2i+1, col = 2j or 2j+1
                # W index = col*n + row = col*6 + row
                c00 = (2*j) * 6 + (2*i)
                c01 = (2*j+1) * 6 + (2*i)
                c10 = (2*j) * 6 + (2*i+1)
                c11 = (2*j+1) * 6 + (2*i+1)
                
                # Strassen's 7 products for 2x2 matrix multiplication
                # C = A*B where A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
                # C = [[c00,c01],[c10,c11]]
                #
                # M0 = (a00+a11)*(b00+b11) -> c00+=M0, c11+=M0
                # M1 = (a10+a11)*b00       -> c10+=M1, c11-=M1
                # M2 = a00*(b01-b11)       -> c01+=M2, c11+=M2
                # M3 = a11*(b10-b00)       -> c00+=M3, c10+=M3
                # M4 = (a00+a01)*b11       -> c00-=M4, c01+=M4
                # M5 = (a10-a00)*(b00+b01) -> c11+=M5
                # M6 = (a01-a11)*(b10+b11) -> c00+=M6
                
                for s in range(7):
                    if s == 0:
                        U[a00, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1; W[c11, idx] = 1
                    elif s == 1:
                        U[a10, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1; W[c11, idx] = -1
                    elif s == 2:
                        U[a00, idx] = 1
                        V[b01, idx] = 1; V[b11, idx] = -1
                        W[c01, idx] = 1; W[c11, idx] = 1
                    elif s == 3:
                        U[a11, idx] = 1
                        V[b10, idx] = 1; V[b00, idx] = -1
                        W[c00, idx] = 1; W[c10, idx] = 1
                    elif s == 4:
                        U[a00, idx] = 1; U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1; W[c01, idx] = 1
                    elif s == 5:
                        U[a10, idx] = 1; U[a00, idx] = -1
                        V[b00, idx] = 1; V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:
                        U[a01, idx] = 1; U[a11, idx] = -1
                        V[b10, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verification with the stated W convention: W[j*n+i, r]
    # C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * V[k*p+j, r])
    if time.time() < deadline - 10:
        np.random.seed(12345)
        A_test = np.random.randint(-3, 3, (n, m)).astype(float)
        B_test = np.random.randint(-3, 3, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_f = U.astype(float)
        V_f = V.astype(float)
        W_f = W.astype(float)
        
        # M[i,j,r] = sum_k U[i*m+k, r] * V[k*p+j, r]
        U_r = U_f.reshape(n, m, rank)
        V_r = V_f.reshape(m, p, rank)
        M = np.einsum('ikr,kjr->ijr', U_r, V_r)
        
        # C[i,j] = sum_r M[i,j,r] * W[j*n+i, r]
        W_r = W_f.reshape(p, n, rank)
        C_computed = np.einsum('ijr,jir->ij', M, W_r)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Try alternative: W[i*n+j, r]
            W_r_alt = W_f.reshape(n, p, rank)
            C_alt = np.einsum('ijr,ijr->ij', M, W_r_alt)
            if np.allclose(C_alt, C_expected, atol=1e-9):
                W_perm = W_r_alt.transpose(1, 0, 2).reshape(p * n, rank)
                return {"U": U.tolist(), "V": V.tolist(), "W": W_perm.astype(int).tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
