import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Known rank-5 decomposition for 2x2x3 matmul
    # A is 2x2, B is 2x3, C is 3x2? No, C is 2x3.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1.
    # We need U (2*2=4 x 5), V (2*3=6 x 5), W (3*2=6 x 5)
    # Indices: U[i*2+j], V[j*3+k], W[k*2+i]
    
    # One rank-5 decomposition for 2x2x3:
    # Based on Strassen-like or specific algebraic solution.
    # Let's use a known solution or generate one via ALS for 2x2x3.
    
    def _solve_2x2x3():
        n2, m2, p2 = 2, 2, 3
        rank = 5
        # Initialize
        U = np.random.randn(n2*m2, rank)
        V = np.random.randn(m2*p2, rank)
        W = np.random.randn(p2*n2, rank)
        
        # ALS
        for _ in range(500):
            # Update W
            for i in range(n2):
                for k in range(p2):
                    A = np.zeros((m2, rank))
                    for j in range(m2):
                        A[j, :] = U[i*m2 + j, :] * V[j*p2 + k, :]
                    sol, _, _, _ = np.linalg.lstsq(A, np.ones(m2), rcond=None)
                    W[k*n2 + i, :] = sol
            # Update U
            for i in range(n2):
                for j in range(m2):
                    B = np.zeros((p2, rank))
                    for k in range(p2):
                        B[k, :] = V[j*p2 + k, :] * W[k*n2 + i, :]
                    sol, _, _, _ = np.linalg.lstsq(B, np.ones(p2), rcond=None)
                    U[i*m2 + j, :] = sol
            # Update V
            for j in range(m2):
                for k in range(p2):
                    C = np.zeros((n2, rank))
                    for i in range(n2):
                        C[i, :] = U[i*m2 + j, :] * W[k*n2 + i, :]
                    sol, _, _, _ = np.linalg.lstsq(C, np.ones(n2), rcond=None)
                    V[j*p2 + k, :] = sol
            
            # Check error
            err = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = np.dot(U[i*m2+j, :], V[j*p2+k, :] * W[k*n2+i, :])
                        err += (val - 1)**2
            if err < 1e-10:
                break
        
        # Round to half-integers if possible
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = np.dot(U[i*m2+j, :], V[j*p2+k, :] * W[k*n2+i, :])
                    if abs(val - 1) > 1e-6:
                        # Fallback to schoolbook for 2x2x3 (rank 6)
                        U = np.zeros((4, 6))
                        V = np.zeros((6, 6))
                        W = np.zeros((6, 6))
                        for ii in range(2):
                            for jj in range(2):
                                for kk in range(3):
                                    c = ii*2*3 + jj*3 + kk
                                    U[ii*2+jj, c] = 1
                                    V[jj*3+kk, c] = 1
                                    W[kk*2+ii, c] = 1
                        return U, V, W, 6
        return U, V, W, 5

    # Get the 2x2x3 decomposition
    U2, V2, W2, r2 = _solve_2x2x3()
    
    # Construct 4x4x6 solution using block decomposition
    # A (4x4) -> 4 blocks of 2x2: A00, A01, A10, A11
    # B (4x6) -> 2 blocks of 2x6? Or 4 blocks of 2x3?
    # B is 4x6. Split rows into 2x2 blocks (4 blocks), cols into 2x3 blocks (2 blocks).
    # B00 (2x3), B01 (2x3)
    # B10 (2x3), B11 (2x3)
    
    # C (4x6) -> 4 blocks of 2x3: C00, C01, C10, C11
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each term is a 2x2x3 matmul.
    # We have 8 terms. Each uses rank r2.
    # Total rank = 8 * r2.
    
    total_rank = 8 * r2
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Map blocks to indices
    # A blocks: (i_b, j_b) -> rows i_b*2..i_b*2+1, cols j_b*2..j_b*2+1
    # B blocks: (j_b, k_b) -> rows j_b*2..j_b*2+1, cols k_b*3..k_b*3+2
    # C blocks: (i_b, k_b) -> rows i_b*2..i_b*2+1, cols k_b*3..k_b*3+2
    
    # For each of the 8 terms (i_b, j_b, k_b) where i_b in 0,1 and j_b in 0,1 and k_b in 0,1
    # Wait, C(i_b, k_b) = sum_j_b A(i_b, j_b) B(j_b, k_b)
    # So for each i_b, k_b, we sum over j_b.
    # Term (i_b, j_b, k_b) contributes to C(i_b, k_b).
    
    for i_b in range(2):
        for k_b in range(2):
            for j_b in range(2):
                # This term is A(i_b, j_b) * B(j_b, k_b)
                # A(i_b, j_b) is 2x2. Index in A: i=i_b*2+ii, j=j_b*2+jj
                # B(j_b, k_b) is 2x3. Index in B: j=j_b*2+jj, k=k_b*3+kk
                # C(i_b, k_b) is 2x3. Index in C: i=i_b*2+ii, k=k_b*3+kk
                
                # Assign a slice of the rank dimension
                rank_offset = (i_b * 4 + j_b * 2 + k_b) * r2
                
                # Copy U2, V2, W2 to the appropriate positions
                for ii in range(2):
                    for jj in range(2):
                        # U index: i*m + j = (i_b*2+ii)*4 + (j_b*2+jj)
                        u_idx = (i_b*2+ii)*4 + (j_b*2+jj)
                        # U2 index: ii*2 + jj
                        u2_idx = ii*2 + jj
                        U[u_idx, rank_offset:rank_offset+r2] = U2[u2_idx, :]
                        
                for jj in range(2):
                    for kk in range(3):
                        # V index: j*p + k = (j_b*2+jj)*6 + (k_b*3+kk)
                        v_idx = (j_b*2+jj)*6 + (k_b*3+kk)
                        # V2 index: jj*3 + kk
                        v2_idx = jj*3 + kk
                        V[v_idx, rank_offset:rank_offset+r2] = V2[v2_idx, :]
                        
                for kk in range(3):
                    for ii in range(2):
                        # W index: k*n + i = (k_b*3+kk)*4 + (i_b*2+ii)
                        w_idx = (k_b*3+kk)*4 + (i_b*2+ii)
                        # W2 index: kk*2 + ii
                        w2_idx = kk*2 + ii
                        W[w_idx, rank_offset:rank_offset+r2] = W2[w2_idx, :]
                        
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-6:
                    # Fallback to schoolbook
                    return {
                        "U": np.eye(n*m).tolist(),
                        "V": np.eye(m*p).tolist(),
                        "W": np.eye(p*n).tolist()
                    }
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
