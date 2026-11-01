"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 115s as safety margin
    
    n, m, p = 2, 4, 4
    rank_28 = 28
    
    # Build the rank-28 Strassen decomposition
    U = [[0] * rank_28 for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank_28 for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank_28 for _ in range(p * n)]  # (8, 28)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    # Build the 4 Strassen blocks
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    U_28 = np.array(U, dtype=float)
    V_28 = np.array(V, dtype=float)
    W_28 = np.array(W, dtype=float)
    
    # Search for rank-27 decomposition by finding a sparse 27x28 matrix S
    # such that the tensor can be expressed with rank 27
    # Strategy: Try to find a linear combination of the 28 rank-1 terms that
    # gives the same tensor with one fewer term
    
    # The tensor T is determined by the triple (U, V, W) where:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # To reduce rank, we need to find a 27-term decomposition.
    # One approach: look for a dependency among the 28 terms.
    # If we can write one term as a linear combination of others in a way
    # that preserves the tensor structure, we can eliminate it.
    
    # Alternative: Search over sparse transformations S (28x28) with one
    # zero row, such that the transformed decomposition still equals T.
    
    # Let's try a more direct approach: enumerate possible rank-27 decompositions
    # by trying to merge or eliminate terms.
    
    # First, verify the rank-28 decomposition is correct
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # For random A (n x m) and B (m x p), check A*B = sum_r U[:,r] * V[:,r] * W[:,r]
        # More precisely: C[i,k] = sum_r U[i,r] * V[k,r] * W[r] ... 
        # Actually the tensor is: T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # where a indexes A elements, b indexes B elements, c indexes C elements
        # And the multiplication is: C[i,j] = sum_k A[i,k] B[k,j]
        # So T[A_idx(i,k), B_idx(k,j), C_idx(i,j)] = 1
        
        # Check with a few random matrices
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            # Compute C from decomposition
            # C[i,j] = sum_r (sum_k A[i,k] * U[k*m+i, r]) * (sum_l B[l,j] * V[l*p+j, r]) * W[j*n+i, r]
            # Wait, the decomposition is:
            # C[i,j] = sum_r (U_vec_r . A_flat) * (V_vec_r . B_flat) * W[j*n+i, r]
            # where A_flat[i*m+k] = A[i,k], B_flat[l*p+j] = B[l,j]
            
            A_flat = A.flatten()  # length n*m = 8
            B_flat = B.flatten()  # length m*p = 16
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for r in range(len(U_mat[0])):
                        u_val = np.dot(U_mat[:, r], A_flat)
                        v_val = np.dot(V_mat[:, r], B_flat)
                        s += u_val * v_val * W_mat[j * n + i, r]
                    C_computed[i, j] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    if not verify_decomposition(U_28, V_28, W_28, n, m, p):
        # Fallback: return the rank-28 decomposition
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-27 decomposition
    # Approach: Look for a vector s in the null space of a certain matrix
    # that allows us to express one term as a combination of others.
    
    # The tensor T has entries T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # If we have a 28-term decomposition, and we want to find a 27-term one,
    # we need to find coefficients alpha_r such that:
    # sum_r alpha_r * U[:,r] V[:,r] W[c,r] = 0 for all a,b,c
    # and exactly one alpha_r is nonzero (or we can eliminate one term).
    
    # Actually, a cleaner approach: if there exists a vector s such that
    # sum_r s_r * U[:,r] = 0 AND sum_r s_r * V[:,r] = 0 AND sum_r s_r * W[:,r] = 0
    # then we have a dependency. But that's too strong.
    
    # The correct condition: The 28 rank-1 tensors U[:,r] ⊗ V[:,r] ⊗ W[:,r] 
    # span a space. If they are linearly dependent, we can reduce the rank.
    
    # Let's check if the 28 terms are linearly independent by building the
    # 28 vectors in R^(8*16*8) = R^1024.
    
    # Actually, a more efficient approach: 
    # Try to find a 27x28 matrix S (with one zero row) such that:
    # U' = U @ S, V' = V @ S, W' = W @ S gives a valid rank-27 decomposition.
    # This requires S to be such that the tensor is preserved.
    
    # The tensor is preserved if for all a,b,c:
    # sum_r U[a,r] V[b,r] W[c,r] = sum_{r',r''} U[a,r'] S[r',r''] V[b,r''] S[r'',r''] W[c,r''] S[r'',r'']
    # This is complex. Let's try a simpler search.
    
    # Simpler approach: Try to find pairs of terms that can be merged.
    # If terms r1 and r2 have U[:,r1] proportional to U[:,r2], or similar,
    # we might be able to merge them.
    
    # Let's try a randomized search for rank-27 decompositions.
    # We'll try to find a 27-term decomposition by optimizing over the parameters.
    
    # Given the time constraint, let's try a targeted search:
    # Look for a dependency among the 28 terms by checking if any subset
    # of 27 terms spans the same tensor space.
    
    # For each term r to eliminate, check if the remaining 27 terms
    # can still represent the tensor.
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = 28
    
    # Try eliminating each term and see if we can compensate
    for elim_r in range(28):
        if time.time() > deadline - 5:
            break
        
        # Remove term elim_r
        U_sub = np.delete(U_28, elim_r, axis=1)  # (8, 27)
        V_sub = np.delete(V_28, elim_r, axis=1)  # (16, 27)
        W_sub = np.delete(W_28, elim_r, axis=1)  # (8, 27)
        
        # Check if the tensor from the 27 terms matches the full tensor
        # T_full[a,b,c] = sum_{r != elim_r} U[a,r] V[b,r] W[c,r] + U[a,elim_r] V[b,elim_r] W[c,elim_r]
        # We need: sum_{r != elim_r} U[a,r] V[b,r] W[c,r] = T_full[a,b,c] - U[a,elim_r] V[b,elim_r] W[c,elim_r]
        
        # This is a linear system in the "new" coefficients, but we're not changing
        # the U,V,W values, just removing a term. So we need to check if the
        # remaining 27 terms already span the full tensor.
        
        # Compute the tensor from the 27 terms and compare to the full tensor
        # This is expensive, so let's use a more efficient check.
        
        # The tensor T is a 3D array of size 8 x 16 x 8 = 1024 entries.
        # We can flatten it and check linear dependence.
        
        # Build the tensor from all 28 terms
        T_full = np.zeros((8, 16, 8))
        for r in range(28):
            T_full += np.einsum('i,j,k->ijk', U_28[:, r], V_28[:, r], W_28[:, r])
        
        # Build the tensor from the 27 terms (excluding elim_r)
        T_sub = np.zeros((8, 16, 8))
        for r in range(27):
            T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
        
        # Check if T_full - T_sub is zero (it should be the removed term)
        T_diff = T_full - T_sub
        # T_diff should equal U_28[:, elim_r] ⊗ V_28[:, elim_r] ⊗ W_28[:, elim_r]
        
        # If T_diff is zero, then the 27 terms already span the tensor (rank <= 27)
        if np.allclose(T_diff, 0, atol=1e-10):
            best_rank = 27
            best_result = {
                "U": U_sub.tolist(),
                "V": V_sub.tolist(),
                "W": W_sub.tolist()
            }
            break
    
    # If no rank-27 found by simple elimination, try a more sophisticated search
    # Look for a linear combination of terms that cancels out
    
    if best_rank == 28 and time.time() < deadline - 10:
        # Try to find a non-trivial linear combination
        # The 28 rank-1 tensors form a set of vectors in R^1024.
        # If they are linearly dependent, we can find coefficients that sum to zero.
        
        # Build the 28 vectors in R^1024
        vecs = []
        for r in range(28):
            vec = np.einsum('i,j,k->ijk', U_28[:, r], V_28[:, r], W_28[:, r]).flatten()
            vecs.append(vec)
        
        mat = np.array(vecs).T  # (1024, 28)
        
        # Find the null space
        try:
            # Use SVD to find null space
            U_svd, s_vals, Vt = np.linalg.svd(mat, full_matrices=False)
            # Null space vectors are the last rows of Vt where s_vals is small
            rank_mat = np.sum(s_vals > 1e-10)
            if rank_mat < 28:
                # There is a null space
                null_vecs = Vt[rank_mat:]  # (28-rank_mat, 28)
                # For each null vector, check if it corresponds to a valid rank reduction
                for nv in null_vecs:
                    if time.time() > deadline - 5:
                        break
                    # nv is a vector such that sum_r nv[r] * vec_r = 0
                    # This means sum_r nv[r] * U[:,r] ⊗ V[:,r] ⊗ W[:,r] = 0
                    # We can use this to eliminate one term if nv has a nonzero component.
                    
                    # Find the largest component
                    max_idx = np.argmax(np.abs(nv))
                    max_val = nv[max_idx]
                    
                    if abs(max_val) > 1e-10:
                        # Try to eliminate term max_idx
                        # The other terms can be adjusted by nv[r]/max_val
                        # But this changes the U, V, W values, which is not allowed
                        # in a simple rank reduction.
                        
                        # Actually, if sum_r nv[r] * U[:,r] ⊗ V[:,r] ⊗ W[:,r] = 0,
                        # then the 28 terms are linearly dependent, and the rank is at most 27.
                        # But we need to find an explicit rank-27 decomposition.
                        
                        # One way: if nv[max_idx] != 0, we can write:
                        # U[:,max_idx] ⊗ V[:,max_idx] ⊗ W[:,max_idx] = 
                        #   -sum_{r != max_idx} (nv[r]/nv[max_idx]) * U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                        
                        # This doesn't directly give a rank-27 decomposition unless
                        # the right side can be expressed as a single rank-1 tensor.
                        
                        # Let's check if the remaining 27 terms span the tensor
                        U_sub = np.delete(U_28, max_idx, axis=1)
                        V_sub = np.delete(V_28, max_idx, axis=1)
                        W_sub = np.delete(W_28, max_idx, axis=1)
                        
                        T_sub = np.zeros((8, 16, 8))
                        for r in range(27):
                            T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                        
                        if np.allclose(T_sub, T_full, atol=1e-10):
                            best_rank = 27
                            best_result = {
                                "U": U_sub.tolist(),
                                "V": V_sub.tolist(),
                                "W": W_sub.tolist()
                            }
                            break
        except Exception:
            pass
    
    if best_rank < 28:
        return best_result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
