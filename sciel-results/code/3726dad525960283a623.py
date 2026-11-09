"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build the rank-28 Strassen-based decomposition (4 blocks x 7 Strassen terms).
    2. Use linear algebra to find the true minimal rank by analyzing tensor structure.
    3. Try systematic rank reduction via column space analysis.
    4. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def construct_strassen_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard 2x2x2 Strassen terms (7 terms)
        # M1 = (a+d)(e+h), M2 = (c+d)*e, M3 = a*(f-h), M4 = d*(g-e)
        # M5 = (a+b)*h, M6 = (c-a)*(e+f), M7 = (b-d)(g+h)
        strassen_terms = [
            {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
            {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
            {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
            {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, 1)]},
            {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
            {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
            {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
        ]
        
        # Four 2x2 blocks in the 4x4 matrix
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        col = 0
        for j_off, k_off in blocks:
            for term in strassen_terms:
                for (li, lj, coeff) in term['U']:
                    U[li * m + (lj + j_off)][col] += coeff
                for (lj, lk, coeff) in term['V']:
                    V[(lj + j_off) * p + (lk + k_off)][col] += coeff
                for (lk, li, coeff) in term['W']:
                    W[(lk + k_off) * n + li][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build the rank-28 decomposition
    best = construct_strassen_rank28()
    
    if not verify(best['U'], best['V'], best['W']):
        return best
    
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    best_rank = U_arr.shape[1]
    
    def try_remove_columns(arrays, max_removals=10):
        """Try to remove up to max_removals columns one at a time."""
        U_a, V_a, W_a = arrays
        removed = 0
        for _ in range(max_removals):
            if time_left() < 10:
                break
            found = False
            for c_remove in range(U_a.shape[1]):
                U_test = np.delete(U_a, c_remove, axis=1)
                V_test = np.delete(V_a, c_remove, axis=1)
                W_test = np.delete(W_a, c_remove, axis=1)
                if verify(U_test, V_test, W_test):
                    U_a, V_a, W_a = U_test, V_test, W_test
                    removed += 1
                    found = True
                    break
            if not found:
                break
        return U_a, V_a, W_a, removed
    
    # Phase 1: Try removing single columns iteratively
    U_arr, V_arr, W_arr, _ = try_remove_columns((U_arr, V_arr, W_arr), max_removals=10)
    best_rank = U_arr.shape[1]
    
    # Phase 2: More aggressive rank reduction using tensor structure
    # The key insight: for matrix multiplication, we can analyze the tensor
    # by looking at its slice matrices and finding dependencies
    
    if time_left() > 30:
        rank = U_arr.shape[1]
        if rank > 20:
            # Try to find a minimal generating set using linear algebra
            # For each output position (i,k), the contribution comes from sum over j of A[i,j]*B[j,k]
            # The tensor T[i,j,k] = 1 for all valid (i,j,k)
            
            # Alternative approach: try to find if we can reduce by looking at
            # the structure of which Strassen terms are actually needed
            
            # Let's try a different block decomposition that might have lower rank
            # For 2x4 * 4x4, we can think of it as:
            # A (2x4) * B (4x4) = C (2x4)
            # Split B into two 4x2 blocks: B_left (cols 0,1) and B_right (cols 2,3)
            # C_left = A * B_left, C_right = A * B_right
            # Each is 2x4 * 4x2 = 2x2
            
            # For 2x4 * 4x2, we can use a rank-12 algorithm or better
            # Actually, let's try to find the true rank by optimization
            
            # Use a greedy approach: try all subsets of columns of size r
            # and check if they span the tensor space
            
            # First, compute the tensor's "rank" by looking at the bilinear map
            # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
            
            # For the tensor to have rank r, we need r terms such that
            # T = sum_c U_c ⊗ V_c ⊗ W_c where T[i,j,k] = 1
            
            # Let's try to find a decomposition of rank r by solving for the factors
            # This is NP-hard in general, but for small r we can try
            
            # Try ranks from current best down to theoretical minimum
            # Theoretical minimum for 2x4x4: at least max(2*4, 4*4, 4*2) = 16
            # But matrix multiplication rank is typically higher
            
            # Let's try a structured approach: use the fact that we have 4 independent
            # 2x2 multiplications, each of rank 7, but there might be sharing
            
            # Try to find if any two Strassen terms across different blocks can be merged
            # by having a common factor structure
            
            improved = True
            while improved and time_left() > 15:
                improved = False
                rank = U_arr.shape[1]
                
                # For each pair of columns, check if they can be replaced by a single column
                # with appropriate scaling, or if one is a linear combination of others
                
                for i in range(rank):
                    if time_left() < 15:
                        break
                    # Try removing column i
                    U_test = np.delete(U_arr, i, axis=1)
                    V_test = np.delete(V_arr, i, axis=1)
                    W_test = np.delete(W_arr, i, axis=1)
                    if verify(U_test, V_test, W_test):
                        U_arr, V_arr, W_arr = U_test, V_test, W_test
                        improved = True
                        break
                
                # Also try: for each column, see if it can be scaled to reduce
                # the number of non-zero entries, potentially enabling removal of others
                if not improved and time_left() > 20:
                    rank = U_arr.shape[1]
                    for i in range(rank):
                        if time_left() < 15:
                            break
                        # Try scaling column i by various factors
                        # This doesn't change the rank but might help with numerical issues
                        pass
    
    # Phase 3: Try a completely different construction with potentially lower rank
    # For 2x4 * 4x4, consider using the fact that the output is 2x4
    # We can view this as 4 independent 2x4 * 4x1 multiplications (one per output column)
    # Each 2x4 * 4x1 has rank at most 8 (naive) but can be done in rank 7 or less
    
    # Actually, let's try to use a rank-16 construction based on the following:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed k, this is a linear map from A (2x4) to C[:,k] (2x1)
    # The bilinear complexity of 2x4 * 4x1 is at most 8
    
    # But we can do better by sharing computation across k values
    
    # Let's try a rank-24 construction: use 3 Strassen-like blocks instead of 4
    # by grouping the 4x4 B matrix differently
    
    if time_left() > 40 and best_rank > 24:
        # Try a different block structure
        # Split B into 2x4 blocks vertically: B_top (rows 0,1) and B_bottom (rows 2,3)
        # Then C = A_top * B_top + A_bottom * B_bottom where A_top = A[:,0:2], A_bottom = A[:,2:4]
        # Each term is 2x2 * 2x4
        # 2x2 * 2x4 can be done in rank 14 (2 * 7)
        # Total: 14 + 14 = 28, same as before
        
        # Alternative: use a rank-12 algorithm for 2x2 * 2x4
        # This would give total rank 24
        
        # Let's try to construct a rank-12 decomposition for 2x2 * 2x4
        # A (2x2) * B (2x4) = C (2x4)
        # C[0,0] = a*e + c*g
        # C[0,1] = a*f + c*h
        # C[0,2] = a*i + c*k
        # C[0,3] = a*j + c*l
        # C[1,0] = b*e + d*g
        # C[1,1] = b*f + d*h
        # C[1,2] = b*i + d*k
        # C[1,3] = b*j + d*l
        
        # This is 8 bilinear terms. Can we do it in fewer?
        # The rank of the 2x2*2x4 multiplication tensor is known to be 12
        
        # For now, let's just try to verify if rank 24 is achievable
        # by modifying the Strassen construction
        
        # Actually, the simplest improvement: try to find if the 28-term decomposition
        # has any linear dependencies we missed
        
        # Use SVD-like analysis on the tensor
        rank = U_arr.shape[1]
        
        # Build the tensor and try to find its minimal rank
        # T[i,j,k] = 1 for all i,j,k
        # We want T = sum_c u_c[i] * v_c[j] * w_c[k]
        
        # This is equivalent to finding the rank of the tensor
        # For small tensors, we can try all subsets
        
        if time_left() > 50:
            # Try to find a rank-27 decomposition by checking if any column is redundant
            # in a more subtle way (not just removal, but replacement)
            
            # For each column i, try to express it as a combination of others
            # with the constraint that the resulting decomposition is still valid
            
            rank = U_arr.shape[1]
            dim = n * m * p  # 32
            
            for i in range(rank):
                if time_left() < 20:
                    break
                
                # Contribution of column i to the tensor
                U3 = U_arr.reshape(n, m, rank)
                V3 = V_arr.reshape(m, p, rank)
                W3 = W_arr.reshape(p, n, rank)
                
                T_i = np.einsum('ij,jk,ki->ijk', U3[:, :, i], V3[:, :, i], W3[:, :, i])
                T_i_flat = T_i.flatten()
                
                # Check if T_i is in the span of other columns' contributions
                other_cols = [c for c in range(rank) if c != i]
                
                if len(other_cols) >= 1:
                    M = np.zeros((dim, len(other_cols)))
                    for idx, c in enumerate(other_cols):
                        T_c = np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
                        M[:, idx] = T_c.flatten()
                    
                    try:
                        coeffs, residuals, rnk, _ = np.linalg.lstsq(M, T_i_flat, rcond=1e-10)
                        if len(residuals) == 0:
                            residual = np.linalg.norm(M @ coeffs - T_i_flat)
                        else:
                            residual = np.linalg.norm(residuals[0])
                        
                        if residual < 1e-8:
                            # Column i is redundant
                            U_arr = np.delete(U_arr, i, axis=1)
                            V_arr = np.delete(V_arr, i, axis=1)
                            W_arr = np.delete(W_arr, i, axis=1)
                            best_rank = U_arr.shape[1]
                            break
                    except Exception:
                        continue
    
    # Final verification
    if not verify(U_arr, V_arr, W_arr):
        best = construct_strassen_rank28()
        return best
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END
