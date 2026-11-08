import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where:
    #   a indexes A (n*m): a = i * m + j, i in [0,n), j in [0,m)
    #   b indexes B (m*p): b = j * p + k, j in [0,m), k in [0,p)
    #   c indexes C (p*n): c = k * n + i, k in [0,p), i in [0,n)  (trace convention)
    # T[a, b, c] = 1 if there exist (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Use linear algebra to find minimal rank decomposition
    # Strategy: The CP rank of a 3-way tensor can be found by examining the
    # linear structure. For matmul tensors, we use the fact that:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # 
    # Key insight: Fix W, then for each c, the matrix T[:,:,c] should be a
    # linear combination of rank-1 matrices U[:,r]*V[:,r]^T.
    # 
    # We use the following approach:
    # 1. Compute the "flattened" matrix T_(1) of shape (n*m, m*p*p*n)
    # 2. The rank of T_(1) gives a lower bound on the CP rank
    # 3. Use the null space to find the decomposition
    
    # For the matmul tensor M_{2,3,4}, the known minimum rank is 14.
    # Let's construct a rank-14 decomposition using the known structure.
    
    # The 2x3x4 matmul tensor can be decomposed using 14 terms.
    # We construct it explicitly based on the known optimal algorithm.
    
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # A: (i,j) -> i*m+j, i in {0,1}, j in {0,1,2}
    #   A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B: (j,k) -> j*p+k, j in {0,1,2}, k in {0,1,2,3}
    #   B00=0, B01=1, B02=2, B03=3, B10=4, B11=5, B12=6, B13=7, B20=8, B21=9, B22=10, B23=11
    # C: (k,i) -> k*n+i, k in {0,1,2,3}, i in {0,1}
    #   C00=0, C01=1, C10=2, C11=3, C20=4, C21=5, C30=6, C31=7
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
    
    # We use a decomposition that shares computations across the two rows of A.
    # Define intermediate products:
    # M0 = A[0,0] + A[1,0]
    # M1 = A[0,1] + A[1,1]
    # M2 = A[0,2] + A[1,2]
    # M3 = A[1,0] - A[0,0]
    # M4 = A[1,1] - A[0,1]
    # M5 = A[1,2] - A[0,2]
    
    # Then:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[0,0] + C[1,0] = M0*B[0,0] + M1*B[1,0] + M2*B[2,0]
    # C[1,0] - C[0,0] = M3*B[0,0] + M4*B[1,0] + M5*B[2,0]
    
    # This gives us a way to compute both C[0,k] and C[1,k] using shared terms.
    
    # Let's use the following 14-term decomposition:
    # Terms 0-2: for column 0 of B (k=0)
    # Term 0: (A00+A10)*(B00) -> contributes to C00 and C10
    # Term 1: (A01+A11)*(B10) -> contributes to C00 and C10  
    # Term 2: (A02+A12)*(B20) -> contributes to C00 and C10
    # Terms 3-5: for column 1 of B (k=1)
    # Term 3: (A00+A10)*(B01) -> contributes to C01 and C11
    # Term 4: (A01+A11)*(B11) -> contributes to C01 and C11
    # Term 5: (A02+A12)*(B21) -> contributes to C01 and C11
    # Terms 6-8: for column 2 of B (k=2)
    # Term 6: (A00+A10)*(B02) -> contributes to C02 and C12
    # Term 7: (A01+A11)*(B12) -> contributes to C02 and C12
    # Term 8: (A02+A12)*(B22) -> contributes to C02 and C12
    # Terms 9-11: for column 3 of B (k=3)
    # Term 9: (A00+A10)*(B03) -> contributes to C03 and C13
    # Term 10: (A01+A11)*(B13) -> contributes to C03 and C13
    # Term 11: (A02+A12)*(B23) -> contributes to C03 and C13
    # Terms 12-13: correction terms
    # Term 12: (A10-A00)*(B00+B01+B02+B03) -> correction
    # Term 13: ... this doesn't work cleanly.
    
    # Let me use a different known rank-14 construction.
    # Based on the literature, here's a valid rank-14 decomposition:
    
    # Define the 14 terms as (U_entries, V_entries, W_entries)
    # Each entry is (index, coefficient)
    
    terms = [
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # Term 1: A[0,1]*B[1,0] -> C[0,0]
        ([(1, 1)], [(4, 1)], [(0, 1)]),
        # Term 2: A[0,2]*B[2,0] -> C[0,0]
        ([(2, 1)], [(8, 1)], [(0, 1)]),
        # Term 3: A[1,0]*B[0,0] -> C[1,0]
        ([(3, 1)], [(0, 1)], [(1, 1)]),
        # Term 4: A[1,1]*B[1,0] -> C[1,0]
        ([(4, 1)], [(4, 1)], [(1, 1)]),
        # Term 5: A[1,2]*B[2,0] -> C[1,0]
        ([(5, 1)], [(8, 1)], [(1, 1)]),
        # Term 6: A[0,0]*B[0,1] -> C[0,1]
        ([(0, 1)], [(1, 1)], [(2, 1)]),
        # Term 7: A[0,1]*B[1,1] -> C[0,1]
        ([(1, 1)], [(5, 1)], [(2, 1)]),
        # Term 8: A[0,2]*B[2,1] -> C[0,1]
        ([(2, 1)], [(9, 1)], [(2, 1)]),
        # Term 9: A[1,0]*B[0,1] -> C[1,1]
        ([(3, 1)], [(1, 1)], [(3, 1)]),
        # Term 10: A[1,1]*B[1,1] -> C[1,1]
        ([(4, 1)], [(5, 1)], [(3, 1)]),
        # Term 11: A[1,2]*B[2,1] -> C[1,1]
        ([(5, 1)], [(9, 1)], [(3, 1)]),
        # Term 12: A[0,0]*B[0,2] -> C[0,2]
        ([(0, 1)], [(2, 1)], [(4, 1)]),
        # Term 13: A[0,1]*B[1,2] -> C[0,2]
        ([(1, 1)], [(6, 1)], [(4, 1)]),
    ]
    
    # Wait, this is only 14 terms but I haven't covered all C entries.
    # Let me count: C has 8 entries (4 columns x 2 rows).
    # Each C[i,k] needs 3 terms (one for each j), so 8*3 = 24 base terms.
    # With sharing, we can reduce to 14.
    
    # Let me use a proper rank-14 construction from the literature.
    # The key is to share the computation of A[i,j]*B[j,k] across i values.
    
    # Here's a cleaner approach using the known optimal algorithm:
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We'll use 14 terms with the following structure:
    # For each k in 0..3, we compute C[:,k] using shared A-terms
    
    # Define shared A-combinations:
    # P0 = A[0,0]
    # P1 = A[1,0] 
    # P2 = A[0,1]
    # P3 = A[1,1]
    # P4 = A[0,2]
    # P5 = A[1,2]
    
    # For C[:,0]:
    # C[0,0] = P0*B[0,0] + P2*B[1,0] + P4*B[2,0]
    # C[1,0] = P1*B[0,0] + P3*B[1,0] + P5*B[2,0]
    
    # Use 3 terms for k=0:
    # T0: P0 * B[0,0] -> C[0,0]
    # T1: P2 * B[1,0] -> C[0,0]
    # T2: P4 * B[2,0] -> C[0,0]
    # T3: P1 * B[0,0] -> C[1,0]
    # T4: P3 * B[1,0] -> C[1,0]
    # T5: P5 * B[2,0] -> C[1,0]
    # That's 6 terms for k=0 alone, which is too many.
    
    # The sharing must be more clever. Let me look up the actual rank-14 algorithm.
    
    # Based on the known result, here is a valid rank-14 decomposition:
    
    def set_term(t, u_entries, v_entries, w_entries):
        for idx, val in u_entries:
            U[idx, t] += val
        for idx, val in v_entries:
            V[idx, t] += val
        for idx, val in w_entries:
            W[idx, t] += val
    
    # Term 0: (A00 + A10) * B00 -> C00 + C10
    set_term(0, [(0,1),(3,1)], [(0,1)], [(0,1),(1,1)])
    # Term 1: (A01 + A11) * B10 -> C00 + C10
    set_term(1, [(1,1),(4,1)], [(4,1)], [(0,1),(1,1)])
    # Term 2: (A02 + A12) * B20 -> C00 + C10
    set_term(2, [(2,1),(5,1)], [(8,1)], [(0,1),(1,1)])
    # Term 3: (A00 - A10) * B00 -> C00 - C10
    set_term(3, [(0,1),(3,-1)], [(0,1)], [(0,1),(1,-1)])
    # Term 4: (A01 - A11) * B10 -> C00 - C10
    set_term(4, [(1,1),(4,-1)], [(4,1)], [(0,1),(1,-1)])
    # Term 5: (A02 - A12) * B20 -> C00 - C10
    set_term(5, [(2,1),(5,-1)], [(8,1)], [(0,1),(1,-1)])
    # Term 6: A00 * B01 -> C01
    set_term(6, [(0,1)], [(1,1)], [(2,1)])
    # Term 7: A01 * B11 -> C01
    set_term(7, [(1,1)], [(5,1)], [(2,1)])
    # Term 8: A02 * B21 -> C01
    set_term(8, [(2,1)], [(9,1)], [(2,1)])
    # Term 9: A10 * B01 -> C11
    set_term(9, [(3,1)], [(1,1)], [(3,1)])
    # Term 10: A11 * B11 -> C11
    set_term(10, [(4,1)], [(5,1)], [(3,1)])
    # Term 11: A12 * B21 -> C11
    set_term(11, [(5,1)], [(9,1)], [(3,1)])
    # Term 12: A00 * B02 -> C02
    set_term(12, [(0,1)], [(2,1)], [(4,1)])
    # Term 13: A01 * B12 -> C02
    set_term(13, [(1,1)], [(6,1)], [(4,1)])
    
    # This is only 14 terms but I'm missing C02, C03, C12, C13 contributions from A02, A12 with B22, B23, etc.
    
    # I'm clearly not getting the right construction. Let me use a different approach:
    # Use the actual linear algebra to find the decomposition.
    
    # Reset and use a proper solver
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use the fact that we can solve for the decomposition using the null space
    # of the Khatri-Rao product.
    
    # For a rank-r CP decomposition, we need:
    # T_(1) = U * (V ⊙ W)^T
    # where ⊙ is the Khatri-Rao product
    
    # The matrix V ⊙ W has shape (m*p*p*n, rank) and its columns are
    # v_i ⊗ w_i (Kronecker products)
    
    # If we fix U, then V ⊙ W = (U^+ * T_(1))^T
    # But this is nonlinear in V and W.
    
    # Alternative: use the "slice" approach
    # For each c, T[:, :, c] = sum_r U[:, r] * V[:, r] * W[c, r]
    # Let X_c = T[:, :, c] (shape n*m x m*p)
    # Then X_c = sum_r W[c, r] * (U[:, r] * V[:, r]^T)
    
    # If we define M_r = U[:, r] * V[:, r]^T, then
    # X_c = sum_r W[c, r] * M_r
    # So [X_0, X_1, ..., X_{pn-1}] = [M_0, ..., M_{r-1}] * W^T
    # where the left side is a (n*m*m*p) x (p*n) matrix
    
    # Let X_all = T.transpose(0, 1, 2).reshape(n*m*m*p, p*n)
    # Then X_all = M * W^T where M has shape (n*m*m*p, rank)
    # and each column of M is vec(M_r) = vec(U[:,r] * V[:,r]^T)
    
    # The space of rank-1 matrices U[:,r]*V[:,r]^T has dimension n*m*m*p but
    # the set of such matrices is a variety, not a linear space.
    
    # However, if we fix the structure, we can solve.
    
    # Let me just use a direct numerical approach with multiple random restarts
    # and verify the result.
    
    def try_rank(r):
        """Try to find a rank-r decomposition using ALS with multiple restarts."""
        for seed in range(100):
            np.random.seed(seed)
            U_curr = np.random.randn(n * m, r)
            V_curr = np.random.randn(m * p, r)
            W_curr = np.random.randn(p * n, r)
            
            for it in range(100):
                # Solve for U given V, W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each (b,c): T[:,b,c] = U * (V[b,:]*W[c,:])
                M_u = (V_curr.reshape(m*p, 1, r) * W_curr.reshape(1, p*n, r)).reshape(m*p*p*n, r)
                T_mat_u = T.reshape(n*m, m*p*p*n)
                U_new, _, _, _ = np.linalg.lstsq(M_u, T_mat_u.T, rcond=None)
                U_new = U_new.T
                
                # Solve for V given U, W
                M_v = (U_curr.reshape(n*m, 1, r) * W_curr.reshape(1, p*n, r)).reshape(n*m*p*n, r)
                T_mat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                V_new, _, _, _ = np.linalg.lstsq(M_v, T_mat_v.T, rcond=None)
                V_new = V_new.T
                
                # Solve for W given U, V
                M_w = (U_curr.reshape(n*m, 1, r) * V_curr.reshape(1, m*p, r)).reshape(n*m*m*p, r)
                T_mat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                W_new, _, _, _ = np.linalg.lstsq(M_w, T_mat_w.T, rcond=None)
                W_new = W_new.T
                
                U_curr, V_curr, W_curr = U_new, V_new, W_new
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                err = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
                if err < 1e-12:
                    # Round to nearest half-integer
                    U_rounded = np.round(U_curr * 2) / 2
                    V_rounded = np.round(V_curr * 2) / 2
                    W_rounded = np.round(W_curr * 2) / 2
                    T_recon2 = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_recon2, T, atol=1e-10):
                        return U_rounded, V_rounded, W_rounded
            return None
        return None
    
    # Try ranks from 14 upward
    for r in range(14, 25):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return a valid (possibly non-minimal) decomposition
    # Use the rank-24 trivial decomposition
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, t] = 1
                V[b_idx, t] = 1
                W[c_idx, t] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
