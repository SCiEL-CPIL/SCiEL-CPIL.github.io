"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    # Indices for A (2x4)
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # Indices for B (4x6) blocks
    # Block 1 (cols 0,1)
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3] # C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # Block 2 (cols 2,3)
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # Block 3 (cols 4,5)
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Convert to numpy for SVD
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build matrix M where column r is flattened u_r ⊗ v_r ⊗ w_r
    # Shape (8*24*12, 42) = (2304, 42)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    
    M = np.array(flat_terms).T  # (2304, 42)
    
    # Attempt to find and eliminate linear dependencies to reduce rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < rank:
            # Find null space vectors
            null_space = Vt_svd[effective_rank:]  # (rank-effective_rank, 42)
            
            # Try to eliminate one term at a time
            # For each null space vector, find which terms can be adjusted
            # Strategy: if term r is in the null space support, try to remove it
            # and adjust other terms
            
            best_rank = rank
            best_U, best_V, best_W = None, None, None
            
            # Try removing each term and checking if we can compensate
            for remove_idx in range(rank):
                if time.time() > deadline - 20:
                    break
                    
                # Get null space vectors that have non-zero entry at remove_idx
                null_vectors = []
                for nv in null_space:
                    if abs(nv[remove_idx]) > tol:
                        null_vectors.append(nv)
                
                if not null_vectors:
                    continue
                
                # Try to use the first null vector to eliminate this term
                nv = null_vectors[0]
                
                # Create new U, V, W with term remove_idx removed
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                
                # Apply the null space relation: sum_r nv[r] * u_r ⊗ v_r ⊗ w_r = 0
                # This means: nv[remove_idx] * u_remove ⊗ v_remove ⊗ w_remove = 
                #   -sum_{r != remove_idx} nv[r] * u_r ⊗ v_r ⊗ w_r
                # We can't directly remove a term unless the null vector has only 
                # one non-zero entry, which is unlikely.
                
                # Alternative: use SVD to project onto the effective subspace
                # and reconstruct with fewer terms
                pass
            
            # Better approach: use the SVD to find a minimal-rank representation
            # The tensor T = sum_{r=1}^{42} u_r ⊗ v_r ⊗ w_r
            # If the 42 vectors u_r ⊗ v_r ⊗ w_r are linearly dependent,
            # we can express T with fewer terms.
            
            # Use the left singular vectors to find the rank
            # T = M @ Vt_svd.T, where M has shape (2304, 42)
            # The effective rank is effective_rank
            # We can write T = sum_{i=1}^{effective_rank} s_i * u_i_svd ⊗ (v_i_svd flattened)
            # But we need to decompose each v_i_svd into u ⊗ v ⊗ w form
            
            # For now, stick with rank 42 unless we find a clear dependency
            pass
    except Exception:
        pass
    
    # Final verification
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
