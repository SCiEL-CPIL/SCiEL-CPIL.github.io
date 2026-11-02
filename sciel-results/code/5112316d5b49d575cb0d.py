"""Matrix multiplication 2x4 x 4x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = 2, 4, 4
    rank_28 = 28

    # Build the rank-28 Strassen decomposition using clean block structure
    U = np.zeros((n * m, rank_28), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank_28), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank_28), dtype=float)  # (8, 28)

    # Strassen 2x2 terms: each is (u_coeffs, v_coeffs, w_coeffs)
    # Local 2x2 index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (A10+A11)B00
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3 = A00(B01-B11)
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # M4 = A11(B10-B00)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5 = (A00+A01)B11
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

    # Global index mappings
    # A (2x4): row-major, index = i*m + j = i*4 + j
    u_map_L = [0, 1, 4, 5]   # A cols 0,1
    u_map_R = [2, 3, 6, 7]   # A cols 2,3

    # B (4x4): row-major, index = i*p + j = i*4 + j
    v_map_TL = [0, 1, 4, 5]   # B rows 0,1 cols 0,1
    v_map_TR = [2, 3, 6, 7]   # B rows 0,1 cols 2,3
    v_map_BL = [8, 9, 12, 13] # B rows 2,3 cols 0,1
    v_map_BR = [10, 11, 14, 15] # B rows 2,3 cols 2,3

    # C (2x4): W[k*n + i] = W[k*2 + i]
    # C_L (cols 0,1): local (i,j) -> W[j*2+i]
    w_map_L = [0, 2, 1, 3]
    # C_R (cols 2,3): local (i,j) -> W[(j+2)*2+i]
    w_map_R = [4, 6, 5, 7]

    col = 0
    # 1. A_L * B_TL -> C_L
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1

    # 2. A_R * B_BL -> C_L
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1

    # 3. A_L * B_TR -> C_R
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1

    # 4. A_R * B_BR -> C_R
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1

    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly represents matrix multiplication."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        u_val = np.dot(U_mat[:, r], A_flat)
                        v_val = np.dot(V_mat[:, r], B_flat)
                        s += u_val * v_val * W_mat[j * n + i, r]
                    C_computed[i, j] = s
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the full tensor for analysis
    T_full = np.zeros((n * m, m * p, p * n))
    for r in range(rank_28):
        T_full += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])

    best_rank = 28
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Quick check: for each term, see if removing it still gives the full tensor
    for elim_r in range(rank_28):
        if time.time() > deadline - 10:
            break
        U_sub = np.delete(U, elim_r, axis=1)
        V_sub = np.delete(V, elim_r, axis=1)
        W_sub = np.delete(W, elim_r, axis=1)

        T_sub = np.zeros((n * m, m * p, p * n))
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

    # If no simple elimination works, try SVD-based null space search
    if best_rank == 28 and time.time() < deadline - 15:
        vecs = np.zeros((n * m * m * p * p * n, rank_28))
        for r in range(rank_28):
            vecs[:, r] = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()

        try:
            _, s_vals, Vt = np.linalg.svd(vecs, full_matrices=False)
            rank_mat = np.sum(s_vals > 1e-10)
            if rank_mat < 28:
                null_vecs = Vt[rank_mat:]
                for nv in null_vecs:
                    if time.time() > deadline - 10:
                        break
                    max_idx = np.argmax(np.abs(nv))
                    if abs(nv[max_idx]) > 1e-10:
                        U_sub = np.delete(U, max_idx, axis=1)
                        V_sub = np.delete(V, max_idx, axis=1)
                        W_sub = np.delete(W, max_idx, axis=1)
                        T_sub = np.zeros((n * m, m * p, p * n))
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

    # Tensor splitting approach: partition m=4 into two subsets of size 2
    # and search for minimal rank of each 2x2x4 sub-tensor
    if best_rank == 28 and time.time() < deadline - 30:
        # The tensor T[i, j, k] = sum over m of A[i,m] * B[m,k]
        # Split m into m1 = {0,1} and m2 = {2,3}
        # T = T1 + T2 where T1 uses only m in {0,1} and T2 uses only m in {2,3}
        
        # For each sub-tensor, try to find a lower-rank decomposition
        # by checking if the 2x2x4 tensor has rank < 7
        
        def find_subtensor_rank(T_sub, n, m_sub, p, max_rank=7):
            """Try to find a rank-r decomposition of a 2 x m_sub x p tensor."""
            if time.time() > deadline - 20:
                return None, None, None, max_rank
            
            # Flatten the tensor
            T_flat = T_sub.flatten()
            dim = n * m_sub * p
            
            # Try rank from 1 to max_rank
            for r in range(1, max_rank + 1):
                if time.time() > deadline - 20:
                    return None, None, None, max_rank
                
                # We need to find U (n*m_sub, r), V (m_sub*p, r), W (p*n, r)
                # such that sum_r U[:,r] V[:,r] W[:,r] = T_sub
                #
                # This is a system of dim equations in 3*r*(n*m_sub + m_sub*p + p*n) unknowns
                # For n=2, m_sub=2, p=4: dim=16, unknowns=3*r*(4+8+8)=60r
                # For r=4: 240 unknowns, 16 equations - very underdetermined
                #
                # Use a greedy approach: fix U and V, solve for W, then optimize
                #
                # Alternative: use the fact that for small tensors, we can try
                # all possible rank-r decompositions from a basis
                
                # For 2x2x4, the tensor can be written as:
                # T[i, j, k] = sum_{m=0}^{1} A[i,m] B[m,k]
                # This is a bilinear form in A and B
                #
                # The rank of this tensor is at most 2*4=8 (naive) but could be lower
                # Let's check if it has rank <= 6 by trying to find a decomposition
                
                # Try a simple approach: assume the tensor has a specific structure
                # and see if we can reduce the rank
                
                # For now, just return the naive rank
                pass
            
            return None, None, None, max_rank
        
        # The key insight: the 2x4x4 tensor splits into two 2x2x4 tensors
        # Each 2x2x4 tensor corresponds to multiplying a 2x2 matrix by a 2x4 matrix
        # The rank of 2x2x4 multiplication is known to be 6 (not 7 as in 2x2x2)
        # So total rank could be 6+6=12 instead of 7+7=14 per half, giving 24 total
        
        # Let's verify: for C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
        # First term: 2x2 * 2x2 -> rank 7
        # Second term: 2x2 * 2x2 -> rank 7
        # But wait, the second term is A[:, 2:4] (2x2) times B[2:4, 0:2] (2x2), which is also 2x2x2
        # So each is rank 7, total 14 per output block, 28 total
        
        # The tensor splitting doesn't help here because each sub-product is still 2x2x2
        # The 2x2x4 tensor would be if we had A (2x2) * B (2x4), but that's not our case
        
        # Actually, let me reconsider. The full tensor is:
        # C[i, k] = sum_{j=0}^{3} A[i, j] B[j, k]
        # Split j into {0,1} and {2,3}:
        # C[i, k] = sum_{j=0}^{1} A[i, j] B[j, k] + sum_{j=2}^{3} A[i, j] B[j, k]
        # 
        # First term: A1 (2x2) * B1 (2x4) -> this is a 2x2x4 tensor
        # Second term: A2 (2x2) * B2 (2x4) -> this is also a 2x2x4 tensor
        #
        # The rank of the 2x2x4 multiplication tensor is what we need to determine
        # It's not 7 (that's for 2x2x2), it could be different
        
        # Let's try to find the rank of the 2x2x4 tensor
        # T[i, j, k] = A[i, j] * B[j, k] for i in {0,1}, j in {0,1}, k in {0,1,2,3}
        # This is a 2x2x4 tensor with 16 entries
        # 
        # The naive rank is 2*4=8 (each A[i,j] times each B[j,k])
        # But there might be a lower rank decomposition
        
        # For the 2x2x4 tensor, we can try rank 6, 7, 8
        # If rank is 6, total would be 12; if 7, total 14; if 8, total 16
        # But we already have 28, so this approach might not help unless we can
        # share some terms between the two sub-tensors
        
        # The key question: can we find a decomposition where the two 2x2x4 tensors
        # share some of their rank-1 terms?
        # 
        # If T1 and T2 are the two sub-tensors, and we have:
        # T1 = sum_{r=1}^{r1} u_r v_r w_r
        # T2 = sum_{r=1}^{r2} u'_r v'_r w'_r
        # Can we write T1 + T2 = sum_{r=1}^{r_total} u_r v_r w_r with r_total < r1 + r2?
        # This would require some linear dependence between the rank-1 terms
        
        # For now, let's just try to see if we can find a rank-27 decomposition
        # by checking if any two terms can be combined
        
        if time.time() < deadline - 20:
            # Try pairwise elimination: check if removing any two terms still works
            # This is more expensive but might find a rank-26 decomposition
            found_26 = False
            for elim_r1 in range(28):
                if time.time() > deadline - 15:
                    break
                for elim_r2 in range(elim_r1 + 1, 28):
                    if time.time() > deadline - 15:
                        break
                    
                    U_sub = np.delete(U, [elim_r1, elim_r2], axis=1)
                    V_sub = np.delete(V, [elim_r1, elim_r2], axis=1)
                    W_sub = np.delete(W, [elim_r1, elim_r2], axis=1)
                    
                    T_sub = np.zeros((n * m, m * p, p * n))
                    for r in range(26):
                        T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                    
                    if np.allclose(T_sub, T_full, atol=1e-10):
                        best_rank = 26
                        best_result = {
                            "U": U_sub.tolist(),
                            "V": V_sub.tolist(),
                            "W": W_sub.tolist()
                        }
                        found_26 = True
                        break
                if found_26:
                    break

    return best_result

# EVOLVE-BLOCK-END
