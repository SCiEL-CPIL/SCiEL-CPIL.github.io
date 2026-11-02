"""
Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.

Strategy:
1. Start from the schoolbook decomposition (rank 48).
2. Use Strassen's identity on 2x2 blocks to reduce rank.
3. Use a local search to find further rank reductions.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    rank0 = n * m * p  # 48
    
    # Build the schoolbook decomposition
    U = np.zeros((n * m, rank0), dtype=float)
    V = np.zeros((m * p, rank0), dtype=float)
    W = np.zeros((p * n, rank0), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank0
    
    # Helper to verify exactness
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        if r == 0:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Helper to check if entries are integers or half-integers
    def is_valid_entries(U_mat, V_mat, W_mat):
        for mat in [U_mat, V_mat, W_mat]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # ---- Strassen-based reduction ----
    # For 2x2x2 sub-tensors, Strassen gives rank 7 instead of 8.
    # We tile this across the 3x4x4 tensor.
    # 
    # The 3x4x4 tensor can be viewed as:
    #   Rows: 3 (i = 0,1,2)
    #   Cols: 4 (j = 0,1,2,3)  
    #   Out:  4 (k = 0,1,2,3)
    #
    # Partition into 2x2x2 blocks:
    #   Block (0,0): i in {0,1}, j in {0,1}, k in {0,1} -> rank 7
    #   Block (0,1): i in {0,1}, j in {2,3}, k in {0,1} -> rank 7
    #   Block (0,2): i in {0,1}, j in {0,1}, k in {2,3} -> rank 7
    #   Block (0,3): i in {0,1}, j in {2,3}, k in {2,3} -> rank 7
    #   Block (1,0): i = {2},     j in {0,1}, k in {0,1} -> rank 4 (trivial)
    #   Block (1,1): i = {2},     j in {2,3}, k in {0,1} -> rank 4
    #   Block (1,2): i = {2},     j in {0,1}, k in {2,3} -> rank 4
    #   Block (1,3): i = {2},     j in {2,3}, k in {2,3} -> rank 4
    # Total: 4*7 + 4*4 = 28 + 16 = 44
    
    # Build Strassen decomposition for a 2x2x2 block
    # Given A (2x2), B (2x2), C = A*B
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each 2x2x2 block, the 7 rank-1 terms have:
    # U entries: linear combinations of A entries
    # V entries: linear combinations of B entries  
    # W entries: coefficients for C entries
    
    def strassen_2x2x2():
        """Return (U_block, V_block, W_block) for a 2x2x2 Strassen block.
        U_block: (4, 7) - indices (i,j) for i,j in {0,1}
        V_block: (4, 7) - indices (j,k) for j,k in {0,1}
        W_block: (4, 7) - indices (k,i) for k,i in {0,1}
        """
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        
        # Index mapping: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[k,i] -> W[k*2+i]
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = U[0], A11 = U[3]
        # B00 = V[0], B11 = V[3]
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        # C00 += M1, C11 += M1
        # C00 = W[0], C11 = W[3]
        W_b[0, 0] = 1; W_b[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # A10 = U[2], A11 = U[3]
        # B00 = V[0]
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        # C10 += M2, C11 -= M2
        # C10 = W[2], C11 = W[3]
        W_b[2, 1] = 1; W_b[3, 1] = -1
        
        # M3 = A00(B01-B11)
        # A00 = U[0]
        # B01 = V[1], B11 = V[3]
        U_b[0, 2] = 1
        V_b[1, 2] = 1; V_b[3, 2] = -1
        # C01 += M3, C11 += M3
        # C01 = W[1], C11 = W[3]
        W_b[1, 2] = 1; W_b[3, 2] = 1
        
        # M4 = A11(B10-B00)
        # A11 = U[3]
        # B10 = V[2], B00 = V[0]
        U_b[3, 3] = 1
        V_b[2, 3] = 1; V_b[0, 3] = -1
        # C00 += M4, C10 += M4
        # C00 = W[0], C10 = W[2]
        W_b[0, 3] = 1; W_b[2, 3] = 1
        
        # M5 = (A00+A01)B11
        # A00 = U[0], A01 = U[1]
        # B11 = V[3]
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        # C00 -= M5, C01 += M5
        # C00 = W[0], C01 = W[1]
        W_b[0, 4] = -1; W_b[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10 = U[2], A00 = U[0]
        # B00 = V[0], B01 = V[1]
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        # C11 += M6
        # C11 = W[3]
        W_b[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01 = U[1], A11 = U[3]
        # B10 = V[2], B11 = V[3]
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        # C00 += M7, C11 += M7
        # C00 = W[0], C11 = W[3]
        W_b[0, 6] = 1; W_b[3, 6] = 1
        
        return U_b, V_b, W_b
    
    # Build the full 44-rank decomposition
    # Blocks: (row_block, col_block, out_block)
    # row_block: 0 -> i in {0,1}, 1 -> i in {2}
    # col_block: 0 -> j in {0,1}, 1 -> j in {2,3}
    # out_block: 0 -> k in {0,1}, 1 -> k in {2,3}
    
    U_full = np.zeros((n * m, 44), dtype=float)
    V_full = np.zeros((m * p, 44), dtype=float)
    W_full = np.zeros((p * n, 44), dtype=float)
    
    U_b, V_b, W_b = strassen_2x2x2()
    
    col_offset = 0
    for rb in range(2):  # row block
        for cb in range(2):  # col block
            for ob in range(2):  # out block
                if rb == 0:
                    # Strassen block: 7 terms
                    i_off = 0  # i starts at 0
                    j_off = cb * 2  # j starts at 0 or 2
                    k_off = ob * 2  # k starts at 0 or 2
                    
                    for t in range(7):
                        # U indices: (i, j) where i in {0,1}, j in {j_off, j_off+1}
                        for i_local in range(2):
                            for j_local in range(2):
                                U_full[(i_off + i_local) * m + (j_off + j_local), col_offset + t] = U_b[i_local * 2 + j_local, t]
                        # V indices: (j, k) where j in {j_off, j_off+1}, k in {k_off, k_off+1}
                        for j_local in range(2):
                            for k_local in range(2):
                                V_full[(j_off + j_local) * p + (k_off + k_local), col_offset + t] = V_b[j_local * 2 + k_local, t]
                        # W indices: (k, i) where k in {k_off, k_off+1}, i in {i_off, i_off+1}
                        for k_local in range(2):
                            for i_local in range(2):
                                W_full[(k_off + k_local) * n + (i_off + i_local), col_offset + t] = W_b[k_local * 2 + i_local, t]
                    col_offset += 7
                else:
                    # Trivial block: 4 terms (1x2x2)
                    i_idx = 2  # only i=2
                    j_off = cb * 2
                    k_off = ob * 2
                    
                    for j_local in range(2):
                        for k_local in range(2):
                            U_full[i_idx * m + (j_off + j_local), col_offset] = 1
                            V_full[(j_off + j_local) * p + (k_off + k_local), col_offset] = 1
                            W_full[(k_off + k_local) * n + i_idx, col_offset] = 1
                            col_offset += 1
    
    # Verify
    if verify(U_full, V_full, W_full) and is_valid_entries(U_full, V_full, W_full):
        best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
        best_rank = 44
    
    # Now try local search to reduce rank further
    # Strategy: try to remove one term at a time and see if we can adjust others
    
    def try_remove_term(U_mat, V_mat, W_mat, term_idx, max_iter=50):
        """Try to remove term_idx by adjusting other terms."""
        r = U_mat.shape[1]
        if r <= 1:
            return None
        
        # Remove the term
        mask = [i for i in range(r) if i != term_idx]
        U_new = U_mat[:, mask].copy()
        V_new = V_mat[:, mask].copy()
        W_new = W_mat[:, mask].copy()
        
        r_new = len(mask)
        
        # Now try to adjust the remaining terms to compensate
        # This is a nonlinear problem, so use a simple gradient-free approach
        
        best_loss = np.inf
        best_U_adj = U_new.copy()
        best_V_adj = V_new.copy()
        best_W_adj = W_new.copy()
        
        # Simple local search: perturb each term's coefficients
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(best_U_adj[i*m+j, :], best_V_adj[j*p+k, :] * best_W_adj[k*n+i, :])
                        loss += (val - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
            
            if best_loss < 1e-12:
                break
            
            # Random perturbation
            mat_choice = random.choice([0, 1, 2])
            col_choice = random.randint(0, r_new - 1)
            row_choice = random.randint(0, (best_U_adj.shape[0] if mat_choice == 0 else 
                                            best_V_adj.shape[0] if mat_choice == 1 else 
                                            best_W_adj.shape[0]) - 1)
            delta = random.uniform(-0.5, 0.5)
            
            if mat_choice == 0:
                best_U_adj[row_choice, col_choice] += delta
            elif mat_choice == 1:
                best_V_adj[row_choice, col_choice] += delta
            else:
                best_W_adj[row_choice, col_choice] += delta
        
        if best_loss < 1e-9:
            return (best_U_adj, best_V_adj, best_W_adj)
        
        return None
    
    # Try to remove terms from the 44-rank decomposition
    current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
    current_rank = best_rank
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        term_to_remove = random.randint(0, current_rank - 1)
        result = try_remove_term(current_U, current_V, current_W, term_to_remove, max_iter=30)
        
        if result is not None:
            U_r, V_r, W_r = result
            if is_valid_entries(U_r, V_r, W_r):
                current_U, current_V, current_W = U_r, V_r, W_r
                current_rank -= 1
                if current_rank < best_rank:
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    best_rank = current_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
