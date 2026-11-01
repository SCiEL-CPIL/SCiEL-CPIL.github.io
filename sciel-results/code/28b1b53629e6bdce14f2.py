import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # safety margin before 150s kill

    def build_rank25():
        """
        Build a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
        
        Strategy:
        - Strassen rank-7 for columns (0,1)
        - Strassen rank-7 for columns (2,3) 
        - Strassen rank-7 for columns (4,5)
        - Schoolbook rank-4 for column 6
        Total rank: 7 + 7 + 7 + 4 = 25
        """
        rank = 25
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_strassen_block(col_start, k_start):
            """
            Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
            
            The 7 Strassen products are:
            M1 = (A00+A11)*(B0,ks + B1,ks+1)
            M2 = (A10+A11)*B0,ks
            M3 = A00*(B0,ks+1 - B1,ks+1)
            M4 = A11*(B1,ks - B0,ks)
            M5 = (A00+A01)*B1,ks+1
            M6 = (A10-A00)*(B0,ks+B0,ks+1)
            M7 = (A01-A11)*(B1,ks+B1,ks+1)
            
            Result: C0,ks = M1 - M5 + M7
                    C1,ks = M2 + M4 - M7
                    C0,ks+1 = M3 + M5 - M1
                    C1,ks+1 = M2 + M4 + M6 - M1
            """
            # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
            U[0, col_start+0] += 1.0
            U[3, col_start+0] += 1.0
            V[7*0 + k_start, col_start+0] += 1.0
            V[7*1 + (k_start+1), col_start+0] += 1.0
            W[2*k_start + 0, col_start+0] += 1.0      # C0,ks += M1
            W[2*(k_start+1) + 1, col_start+0] += 1.0  # C1,ks+1 += M1

            # M2 = (A10 + A11) * B0,ks
            U[2, col_start+1] += 1.0
            U[3, col_start+1] += 1.0
            V[7*0 + k_start, col_start+1] += 1.0
            W[2*k_start + 1, col_start+1] += 1.0      # C1,ks += M2
            W[2*(k_start+1) + 1, col_start+1] -= 1.0  # C1,ks+1 -= M2

            # M3 = A00 * (B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1.0
            V[7*0 + (k_start+1), col_start+2] += 1.0
            V[7*1 + (k_start+1), col_start+2] -= 1.0
            W[2*(k_start+1) + 0, col_start+2] += 1.0  # C0,ks+1 += M3
            W[2*(k_start+1) + 1, col_start+2] += 1.0  # C1,ks+1 += M3

            # M4 = A11 * (B1,ks - B0,ks)
            U[3, col_start+3] += 1.0
            V[7*1 + k_start, col_start+3] += 1.0
            V[7*0 + k_start, col_start+3] -= 1.0
            W[2*k_start + 0, col_start+3] += 1.0      # C0,ks += M4
            W[2*k_start + 1, col_start+3] += 1.0      # C1,ks += M4

            # M5 = (A00 + A01) * B1,ks+1
            U[0, col_start+4] += 1.0
            U[1, col_start+4] += 1.0
            V[7*1 + (k_start+1), col_start+4] += 1.0
            W[2*k_start + 0, col_start+4] -= 1.0      # C0,ks -= M5
            W[2*(k_start+1) + 0, col_start+4] += 1.0  # C0,ks+1 += M5

            # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
            U[2, col_start+5] += 1.0
            U[0, col_start+5] -= 1.0
            V[7*0 + k_start, col_start+5] += 1.0
            V[7*0 + (k_start+1), col_start+5] += 1.0
            W[2*(k_start+1) + 1, col_start+5] += 1.0  # C1,ks+1 += M6

            # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
            U[1, col_start+6] += 1.0
            U[3, col_start+6] -= 1.0
            V[7*1 + k_start, col_start+6] += 1.0
            V[7*1 + (k_start+1), col_start+6] += 1.0
            W[2*k_start + 0, col_start+6] += 1.0      # C0,ks += M7

        # Apply Strassen to pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        # Apply schoolbook to column 6
        col_start = 21
        k = 6
        
        # C0,6 = A00*B0,6 + A01*B1,6
        U[0, col_start+0] += 1.0
        V[7*0 + k, col_start+0] += 1.0
        W[2*k + 0, col_start+0] += 1.0
        
        U[1, col_start+1] += 1.0
        V[7*1 + k, col_start+1] += 1.0
        W[2*k + 0, col_start+1] += 1.0
        
        # C1,6 = A10*B0,6 + A11*B1,6
        U[2, col_start+2] += 1.0
        V[7*0 + k, col_start+2] += 1.0
        W[2*k + 1, col_start+2] += 1.0
        
        U[3, col_start+3] += 1.0
        V[7*1 + k, col_start+3] += 1.0
        W[2*k + 1, col_start+3] += 1.0

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_decomp(U, V, W, trials=50):
        """
        Verify that the decomposition exactly computes matrix multiplication.
        Checks: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all valid (i,j,k)
        """
        rank = len(U[0])
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        
        # Check all valid tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    
                    s = 0.0
                    for r in range(rank):
                        s += Ua[a_idx, r] * Va[b_idx, r] * Wa[c_idx, r]
                    
                    if abs(s - 1.0) > 1e-10:
                        return False
        
        # Additional random verification
        import random
        for _ in range(trials):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)], dtype=float)
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)], dtype=float)
            C_true = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = sum(Ua[i*m+j, r] * A[i, j] for i in range(n) for j in range(m))
                v_val = sum(Va[j*p+k, r] * B[j, k] for j in range(m) for k in range(p))
                prod = u_val * v_val
                for kk in range(p):
                    for ii in range(n):
                        C_calc[ii, kk] += Wa[kk*n+ii, r] * prod
            
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        
        return True

    # Build and verify the rank-25 decomposition
    result = build_rank25()
    
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
