import numpy as np
import time

def _verify_matmul(U, V, W, n, m, p, trials=10, seed=42):
    """
    Verify the matrix multiplication identity:
    C[i, k] = sum_j A[i, j] B[j, k]
    Reconstructed:
    C[i, k] = sum_{j, r} W[k*n+i, r] * (sum_j U[i*m+j, r] A[i, j]) * (sum_j V[j*p+k, r] B[j, k])
    Wait, the standard bilinear form reconstruction is:
    C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k]) * W[k, i, r]?
    No, the factors are:
    U is (n*m, rank) -> U[i, j, r]
    V is (m*p, rank) -> V[j, k, r]
    W is (p*n, rank) -> W[k, i, r]
    
    The term is: sum_r U[i, j, r] V[j, k, r] W[k, i, r] A[i, j] B[j, k]
    This sums over i, j, k?
    No, C[i, k] is the output.
    The identity is:
    C[i, k] = sum_j A[i, j] B[j, k]
    Decomposition:
    C[i, k] = sum_r W[k, i, r] * ( sum_j U[i, j, r] A[i, j] ) * ( sum_j V[j, k, r] B[j, k] )
    """
    U_t = np.array(U).reshape(n, m, -1)
    V_t = np.array(V).reshape(m, p, -1)
    W_t = np.array(W).reshape(p, n, -1)
    
    np.random.seed(seed)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute reconstructed C
        C_recon = np.zeros((n, p))
        for r in range(len(U[0])):
            # U[:, :, r] is (n, m)
            # V[:, :, r] is (m, p)
            # W[:, :, r] is (p, n)
            
            # The bilinear form is:
            # M_r = (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
            # Note: The sums are over j.
            # Let a_r = sum_j U[i,j,r] A[i,j] -> This is a vector of length n?
            # No, U[i,j,r] depends on i.
            # So for each i, u_i = sum_j U[i,j,r] A[i,j].
            # For each k, v_k = sum_j V[j,k,r] B[j,k].
            
            # Then the contribution to C[i,k] is W[k,i,r] * u_i * v_k.
            
            u_vec = np.sum(U_t[:, :, r] * A, axis=1) # Shape (n,)
            v_vec = np.sum(V_t[:, :, r] * B, axis=0) # Shape (p,)
            
            # W_t[:, :, r] is (p, n). W[k, i, r].
            # C_recon[i, k] += W[k, i, r] * u_vec[i] * v_vec[k]
            
            # Outer product u_vec (n,) and v_vec (p,) gives (n, p)
            # But we need to multiply by W[k, i, r].
            # C_recon[i, k] = sum_r W[k, i, r] u_vec[i] v_vec[k]
            
            # We can do:
            # C_recon += np.outer(u_vec, v_vec) * W_t.T[:, :, r]
            # W_t is (p, n, R). W_t[:, :, r] is (p, n).
            # W_t.T[:, :, r] is (n, p).
            
            C_recon += np.outer(u_vec, v_vec) * W_t.T[:, :, r]
            
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
            
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    n, m, p = 2, 2, 11
    
    # Rank 39: 5 Strassen pairs (5*7=35) + 1 Schoolbook col (4)
    R = 39
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    rank_offset = 0
    
    # Strassen Coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # U coeffs (A parts)
    # U[i, j, r]
    # A00 is i=0, j=0. A10 is i=1, j=0.
    # A11 is i=1, j=1. A01 is i=0, j=1.
    
    u_strassen = [
        # M1
        [ [1, 0], [0, 1] ], # A00, A11
        # M2
        [ [0, 0], [1, 1] ], # A10, A11
        # M3
        [ [1, 0], [0, 0] ], # A00
        # M4
        [ [0, 0], [0, 1] ], # A11
        # M5
        [ [1, 1], [0, 0] ], # A00, A10
        # M6
        [ [-1, 0], [1, 0] ], # -A00, A11
        # M7
        [ [0, -1], [0, 1] ]  # -A01, A11
    ]
    
    # V coeffs (B parts)
    # V[j, k, r]
    # B00 is j=0, k=0. B10 is j=1, k=0.
    # B01 is j=0, k=1. B11 is j=1, k=1.
    
    v_strassen = [
        # M1
        [ [1, 1], [0, 0] ], # B00, B01? No.
        # M1 = (A00+A11)(B00+B11). B part: B00 (j=0,k=0) + B11 (j=1,k=1).
        # So V[0,0]=1, V[1,1]=1.
        
        # Let's re-define V carefully.
        # M1: B00 + B11 -> V[0,0]=1, V[1,1]=1
        # M2: B00 -> V[0,0]=1
        # M3: B10 - B11 -> V[1,0]=1, V[1,1]=-1
        # M4: B01 - B00 -> V[0,1]=1, V[0,0]=-1
        # M5: B11 -> V[1,1]=1
        # M6: B00 + B01 -> V[0,0]=1, V[0,1]=1
        # M7: B10 + B11 -> V[1,0]=1, V[1,1]=1
    ]
    
    v_strassen = [
        # M1
        [ [1, 0], [0, 1] ], # V[0,0]=1, V[1,1]=1
        # M2
        [ [1, 0], [0, 0] ], # V[0,0]=1
        # M3
        [ [0, 0], [1, -1] ], # V[1,0]=1, V[1,1]=-1
        # M4
        [ [-1, 1], [0, 0] ], # V[0,0]=-1, V[0,1]=1
        # M5
        [ [0, 0], [0, 1] ], # V[1,1]=1
        # M6
        [ [1, 1], [0, 0] ], # V[0,0]=1, V[0,1]=1
        # M7
        [ [0, 0], [1, 1] ]  # V[1,0]=1, V[1,1]=1
    ]
    
    # W coeffs (C parts)
    # W[k, i, r]
    # C00 is k=0, i=0. C01 is k=1, i=0.
    # C10 is k=0, i=1. C11 is k=1, i=1.
    
    w_strassen = [
        # M1
        [ [1, 0], [0, 1] ], # C00=1, C11=1
        # M2
        [ [0, 1], [0, -1] ], # C10=1, C11=-1
        # M3
        [ [0, 0], [1, 1] ], # C01=1, C11=1
        # M4
        [ [1, 0], [0, 0] ], # C00=1
        # M5
        [ [0, -1], [0, 1] ], # C01=1, C00=-1
        # M6
        [ [0, 0], [0, 1] ], # C11=1
        # M7
        [ [0, -1], [1, 0] ]  # C00=-1, C10=1
    ]
    
    # Fill U, V, W for 5 Strassen pairs
    for pair_idx in range(5):
        k_start = pair_idx * 2
        k_end = k_start + 1
        
        for form_idx in range(7):
            r = rank_offset + form_idx
            
            # U
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j, r] = u_strassen[form_idx][i][j]
                    
            # V
            # V[j, k, r]
            # k is local 0 or 1 -> global k_start or k_end
            for j in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j * 11 + k_global, r] = v_strassen[form_idx][j][k_local]
                    
            # W
            # W[k, i, r]
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(2):
                    W[k_global * 2 + i, r] = w_strassen[form_idx][k_local][i]
                    
        rank_offset += 7
        
    # Schoolbook for last column (k=10)
    # C[i, 10] = sum_j A[i, j] B[j, 10]
    for i in range(2):
        for j in range(2):
            r = rank_offset
            U[i * 2 + j, r] = 1
            V[j * 11 + 10, r] = 1
            W[10 * 2 + i, r] = 1
            rank_offset += 1
            
    # Verify
    if _verify_matmul(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 44
        U_sb = np.zeros((n * m, 44))
        V_sb = np.zeros((m * p, 44))
        W_sb = np.zeros((p * n, 44))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
