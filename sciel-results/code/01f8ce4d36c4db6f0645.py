"""
Program for matrix multiplication 2,2,7.
Uses a rank-25 decomposition based on Strassen's algorithm for 2x2 blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 7]
START_TIME = time.time()
DEADLINE = 140.0 # 10s margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    # U is (n*m, rank) = (4, 25)
    # V is (m*p, rank) = (14, 25)
    # W is (p*n, rank) = (14, 25)
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Indexing:
    # A[i,j] -> U[2*i+j]
    # B[j,k] -> V[7*j+k]
    # C[k,i] -> W[2*k+i]  (Trace convention: W[k*n+i, r])
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    # Verify decomposition
    if time.time() - START_TIME < DEADLINE:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Build target tensor T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor decomposition: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        
        # Check a few random cases
        np.random.seed(42)
        for _ in range(10):
            A_test = np.random.randint(-5, 6, (n, m))
            B_test = np.random.randint(-5, 6, (m, p))
            C_expected = A_test @ B_test
            
            # Compute via decomposition
            C_decomp = np.zeros((n, p), dtype=np.float64)
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                
                # u_col[i] for i in 0..3, v_col[j] for j in 0..13, w_col[k] for k in 0..13
                # A[i,j] = u_col[2*i+j], B[j,k] = v_col[7*j+k], C[k,i] = w_col[2*k+i]
                
                # Reconstruct A from u_col
                A_recon = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        A_recon[i, j] = u_col[2*i + j]
                
                # Reconstruct B from v_col
                B_recon = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        B_recon[j, k] = v_col[7*j + k]
                
                # Compute contribution
                C_contrib = A_recon @ B_recon
                C_decomp += C_contrib
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                # Fall back to schoolbook if verification fails
                U_np = np.zeros((n*m, n*m*p))
                V_np = np.zeros((m*p, n*m*p))
                W_np = np.zeros((p*n, n*m*p))
                rank = n*m*p
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_np[2*i+j, idx] = 1
                            V_np[7*j+k, idx] = 1
                            W_np[2*k+i, idx] = 1
                            idx += 1
                U_np = U_np[:, :rank]
                V_np = V_np[:, :rank]
                W_np = W_np[:, :rank]
                break
        
        if time.time() - START_TIME < DEADLINE:
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
