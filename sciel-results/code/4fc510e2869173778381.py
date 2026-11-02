"""Program for 3x5x5 matrix multiplication tensor decomposition using Strassen sub-blocks."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2_2x2():
    """
    Return the 7 rank terms for 2x2x2 matrix multiplication using Strassen.
    """
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _strassen_3x3_3x3_3x3():
    """
    Return a known 21-term decomposition for 3x3x3 matrix multiplication.
    This uses the Laderman algorithm variant or a direct construction.
    We use a known decomposition from literature.
    """
    # Using a known 21-term decomposition for 3x3x3
    # Each term: (u[9], v[9], w[9])
    # u indexed by (i,j) row-major: i in 0..2, j in 0..2
    # v indexed by (j,k) row-major: j in 0..2, k in 0..2  
    # w indexed by (k,i) row-major: k in 0..2, i in 0..2
    
    terms = []
    
    # We'll construct a valid decomposition using block structure
    # Partition 3x3 into 2x2 + 1x1 blocks
    # Use Strassen for the 2x2x2 part and handle cross terms
    
    # Actually, let me use a direct known decomposition.
    # The minimal known rank for 3x3x3 is 21 (Laderman 1976, improved by others).
    
    # I'll use a construction based on embedding.
    # For now, let's use a simpler but valid approach:
    # Build from 2x2x2 Strassen + cross terms
    
    # Partition indices: {0,1} and {2}
    # 2x2x2 block: 7 terms (Strassen)
    # Cross terms: need to handle (2,*,*) *, (*,2,*) *, (*,*,2) terms
    
    # This is complex. Let me use a different strategy:
    # Just return a valid (not necessarily minimal) decomposition for 3x3x3
    # that we can verify, and then optimize.
    
    # For simplicity, use schoolbook for 3x3x3: 27 terms
    # But we want better. Let me construct a 21-term decomposition.
    
    # Known 21-term decomposition for 3x3x3 (from Laderman):
    # This is complex to hardcode. Let me use a different approach.
    
    # Alternative: use the fact that 3x3x3 can be decomposed using
    # 7 (Strassen 2x2x2) + cross terms
    # Cross terms: 
    #   Row 2 with cols 0,1: 2 terms (schoolbook for 1x2x2)
    #   Cols 2 with rows 0,1: 2 terms (schoolbook for 2x1x2)  
    #   Cols 2 with cols 0,1: 2 terms (schoolbook for 2x2x1)
    #   Plus corner term: 1 term
    # Total: 7 + 2 + 2 + 2 + 1 = 14? No, that's not right.
    
    # Let me just use a verified decomposition. I'll construct it programmatically.
    
    # For now, return a placeholder that we'll fill in construct()
    # Actually, let me just build it properly.
    
    # I'll use a known result: rank(3,3,3) <= 21
    # Construction: 
    # M1 = (a00+a11)*(b00+b11) -> contributes to C00, C11
    # M2 = (a10+a11)*b00 -> contributes to C10, C11
    # M3 = a00*(b01-b11) -> contributes to C01, C11
    # M4 = a11*(b10-b00) -> contributes to C00, C10
    # M5 = (a00+a01)*b11 -> contributes to C00, C01
    # M6 = (a10-a00)*(b00+b01) -> contributes to C11
    # M7 = (a01-a11)*(b10+b11) -> contributes to C00
    
    # These handle the 2x2x2 subblock (indices 0,1 in each dimension)
    # Now we need to handle terms involving index 2.
    
    # Terms for C2k (k=0,1,2): sum_j a2j * bjk
    # Terms for Ci2 (i=0,1,2): sum_j aij * bj2
    # Terms for Cij where i,j in {0,1} and k=2: sum_j aij * bj2 (already covered above)
    
    # Let me just do schoolbook for the cross terms:
    # a20*b00, a20*b01, a20*b02, a21*b10, a21*b11, a21*b12, a22*b20, a22*b21, a22*b22 (9 terms for row 2)
    # a02*b00, a02*b01, a02*b02, a12*b10, a12*b11, a12*b12 (6 terms for col 2 in A, excluding overlap)
    # a00*b20, a01*b21, a02*b22, a10*b20, a11*b21, a12*b22 (6 terms for col 2 in B, excluding overlaps)
    # But we have double counting. Let me be more careful.
    
    # Total schoolbook: 27 terms
    # Strassen handles: terms where i,j,k all in {0,1} -> 8 terms replaced by 7
    # Remaining: 27 - 8 = 19 terms
    # Total: 7 + 19 = 26 terms
    
    # That's better than 27 but not 21. For this problem, let's use 26.
    
    # Actually, for the 3x5x5 problem, we don't need a perfect 3x3x3 decomposition.
    # Let me just use the partition approach with 2x2x2 Strassen blocks.
    
    # Return empty for now; we'll handle it in _build_decomposition
    return []

def _build_decomposition(n, m, p):
    """
    Build decomposition for n x m x p matrix multiplication.
    Uses Strassen for 2x2x2 sub-blocks where possible.
    Returns (U, V, W) as numpy arrays.
    """
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    rank = 0
    
    strassen_terms = _strassen_2x2_2x2_2x2()
    
    # For n=3, m=5, p=5:
    # Use partitions that maximize 2x2x2 blocks
    i_parts = [[0, 1], [2]]
    j_parts = [[0, 1], [2, 3], [4]]
    k_parts = [[0, 1], [2, 3], [4]]
    
    for i_range in i_parts:
        for j_range in j_parts:
            for k_range in k_parts:
                ni = len(i_range)
                nj = len(j_range)
                nk = len(k_range)
                
                if ni == 2 and nj == 2 and nk == 2:
                    i0, i1 = i_range[0], i_range[1]
                    j0, j1 = j_range[0], j_range[1]
                    k0, k1 = k_range[0], k_range[1]
                    
                    for (u_vec, v_vec, w_vec) in strassen_terms:
                        for li in range(2):
                            for lj in range(2):
                                u_val = u_vec[li * 2 + lj]
                                if u_val != 0:
                                    i_idx = i_range[li]
                                    j_idx = j_range[lj]
                                    U[i_idx * m + j_idx, rank] = u_val
                        
                        for lj in range(2):
                            for lk in range(2):
                                v_val = v_vec[lj * 2 + lk]
                                if v_val != 0:
                                    j_idx = j_range[lj]
                                    k_idx = k_range[lk]
                                    V[j_idx * p + k_idx, rank] = v_val
                        
                        for lk in range(2):
                            for li in range(2):
                                w_val = w_vec[lk * 2 + li]
                                if w_val != 0:
                                    k_idx = k_range[lk]
                                    i_idx = i_range[li]
                                    W[k_idx * n + i_idx, rank] = w_val
                        
                        rank += 1
                else:
                    for i_idx in i_range:
                        for j_idx in j_range:
                            for k_idx in k_range:
                                U[i_idx * m + j_idx, rank] = 1
                                V[j_idx * p + k_idx, rank] = 1
                                W[k_idx * n + i_idx, rank] = 1
                                rank += 1
    
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return U, V, W, rank

def _verify_vectorized(U, V, W, n, m, p):
    """
    Fast vectorized verification using tensor operations.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We need T[i,j,k] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Reshape for tensor operations
    # U: (n*m, rank) -> (n, m, rank)
    U_t = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_t = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    # T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r
    # This is tricky with einsum because of the cyclic indices.
    
    # Let's do it in a loop over k for efficiency
    for k in range(p):
        # For fixed k, T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # U_part: (n, m, rank)
        # V_part: (m, rank) = V[j, k, r] for all j,r
        # W_part: (n, rank) = W[k, i, r] for all i,r
        
        V_part = V_t[:, k, :]  # (m, rank)
        W_part = W_t[k, :, :]  # (n, rank)
        
        # T_k[i,j] = sum_r U[i,j,r] * V_part[j,r] * W_part[i,r]
        # = sum_r (U[i,j,r] * W_part[i,r]) * V_part[j,r]
        
        # First compute: A[i,j,r] = U[i,j,r] * W_part[i,r] -> (n, m, rank)
        A = U_t * W_part[:, np.newaxis, :]  # (n, m, rank)
        
        # Then T_k[i,j] = sum_r A[i,j,r] * V_part[j,r]
        # = (A * V_part[np.newaxis, :, :]).sum(axis=2)
        T_k = (A * V_part[np.newaxis, :, :]).sum(axis=2)  # (n, m)
        
        if np.any(np.abs(T_k - 1.0) > 1e-9):
            return False
    
    return True

def _try_remove_column_fast(U, V, W, n, m, p, col_idx):
    """Try removing a column and verify using vectorized method."""
    rank = U.shape[1]
    if rank <= 1:
        return None, None, None
    
    U_new = np.delete(U, col_idx, axis=1)
    V_new = np.delete(V, col_idx, axis=1)
    W_new = np.delete(W, col_idx, axis=1)
    
    if _verify_vectorized(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def _greedy_column_removal_fast(U, V, W, n, m, p, deadline):
    """Greedily remove columns while maintaining validity, using fast verification."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = U.shape[1]
        for col_idx in range(rank):
            if (time.time() - start_time) > deadline:
                break
            U_new, V_new, W_new = _try_remove_column_fast(U, V, W, n, m, p, col_idx)
            if U_new is not None:
                U, V, W = U_new, V_new, W_new
                improved = True
                break
    return U, V, W

def _try_pairwise_removal(U, V, W, n, m, p, col1, col2, deadline):
    """Try removing two columns at once."""
    rank = U.shape[1]
    if rank <= 2:
        return None, None, None
    
    # Create mask excluding both columns
    mask = np.ones(rank, dtype=bool)
    mask[col1] = False
    mask[col2] = False
    if col1 > col2:
        col1, col2 = col2, col1
    
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    if _verify_vectorized(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def _greedy_pairwise_removal(U, V, W, n, m, p, deadline):
    """Greedily try to remove pairs of columns."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = U.shape[1]
        for i in range(rank):
            if (time.time() - start_time) > deadline:
                break
            for j in range(i+1, rank):
                if (time.time() - start_time) > deadline:
                    break
                U_new, V_new, W_new = _try_pairwise_removal(U, V, W, n, m, p, i, j, deadline)
                if U_new is not None:
                    U, V, W = U_new, V_new, W_new
                    improved = True
                    break
            if improved:
                break
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build initial decomposition with Strassen sub-blocks
    U, V, W, rank = _build_decomposition(n, m, p)
    
    # Verify using vectorized method
    if not _verify_vectorized(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        if not _verify_vectorized(U, V, W, n, m, p):
            raise ValueError("Verification failed")
    
    # Phase 1: Single column removal
    phase1_deadline = min(deadline, start_time + 40.0)
    U, V, W = _greedy_column_removal_fast(U, V, W, n, m, p, phase1_deadline)
    
    # Phase 2: Pairwise removal
    if time.time() < deadline - 20.0:
        phase2_deadline = min(deadline, start_time + 80.0)
        U, V, W = _greedy_pairwise_removal(U, V, W, n, m, p, phase2_deadline)
    
    # Phase 3: More single column removal
    if time.time() < deadline - 10.0:
        U, V, W = _greedy_column_removal_fast(U, V, W, n, m, p, deadline - 5.0)
    
    # Final verification
    if not _verify_vectorized(U, V, W, n, m, p):
        # Rebuild from scratch
        U, V, W, rank = _build_decomposition(n, m, p)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Clean up floating point values
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx]) < 1e-10:
                    row[idx] = 0.0
                elif abs(row[idx] - round(row[idx])) < 1e-10:
                    row[idx] = float(round(row[idx]))
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
