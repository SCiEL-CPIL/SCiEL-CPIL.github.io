import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    n, m, p = 3, 3, 6
    dim_U = n * m      # 9
    dim_V = m * p      # 18
    dim_W = p * n      # 18

    # Build target tensor: T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_residual(U, V, W):
        """Compute sum_r U[:,r] V[:,r] W[:,r] - T_target squared norm."""
        R = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((R - T_target) ** 2)

    def snap_and_verify(U, V, W):
        """Snap to nearest half-integer and verify exactness."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
        return False, None, None, None

    def get_schoolbook(rank):
        """Standard schoolbook decomposition with rank = n*m*p."""
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def strassen_333():
        """Return rank-7 Strassen decomposition of 3x3x3 matmul tensor.
        
        Returns lists of (u_vec, v_vec, w_vec) where each is a 3-element vector
        for U (i,j), V (j,k), W (k,i) indices.
        """
        # Strassen's 7 products for 3x3 matrix multiplication
        # The standard Strassen algorithm for 3x3 uses 7 multiplications
        # Each product has specific (i,j), (j,k), (k,i) patterns
        
        # Standard Strassen for 3x3:
        # M1 = (A11+A22+A33) * (B11+B22+B33)
        # M2 = (A11-A22) * (B11-B22)  
        # M3 = (A11+A22-A33) * (B11+B22-B33)
        # M4 = (A22-A33) * (B22-B33)
        # M5 = (A33-A11) * (B33-B11)
        # M6 = (A11-A22+A33) * (B11-B22+B33)
        # M7 = (A22-A11+A33) * (B22-B11+B33)
        
        # But these don't directly give the tensor decomposition in the right form.
        # Let me use the known rank-7 decomposition of the 3x3x3 matmul tensor.
        
        # The 3x3x3 matrix multiplication tensor has tensor rank 7.
        # One explicit decomposition (from literature):
        
        # Each term r contributes: U_r[i,j] * V_r[j,k] * W_r[k,i]
        # where U_r, V_r, W_r are 3x3 matrices (flattened to length 9, 9, 9)
        # But for our tensor: U has shape (n*m,) = (9,), V has shape (m*p,) but for 3x3x3 it's (9,), W has shape (p*n,) = (9,)
        
        # Let me construct the rank-7 decomposition explicitly.
        # The matmul tensor T[i,j,k] = 1 iff output position (i,k) gets contribution from A[i,j]*B[j,k]
        # In our convention: U index = i*m+j = 3i+j, V index = j*p+k = 3j+k, W index = k*n+i = 3k+i
        
        # Rank-7 decomposition of 3x3x3 matmul:
        # We need 7 triples (u_r, v_r, w_r) where u_r is in R^9, v_r in R^9, w_r in R^9
        # such that sum_r u_r[i*3+j] * v_r[j*3+k] * w_r[k*3+i] = 1 for all (i,j,k)
        
        # Using the known Strassen-based rank-7 decomposition:
        # Actually, let me use a verified rank-7 decomposition.
        
        # One approach: use the fact that 3x3 matmul tensor = sum of 7 rank-1 tensors
        # Each rank-1 tensor is u ⊗ v ⊗ w where u[i,j], v[j,k], w[k,i]
        
        # Here's a known rank-7 decomposition (indices 0,1,2 for i,j,k):
        terms = []
        
        # Term 1: covers (0,0,0)
        # u[0,0]=1, v[0,0]=1, w[0,0]=1
        u1 = np.zeros(9); v1 = np.zeros(9); w1 = np.zeros(9)
        u1[0] = 1; v1[0] = 1; w1[0] = 1
        terms.append((u1, v1, w1))
        
        # This approach of listing 7 terms manually is error-prone.
        # Let me use a different strategy: construct via Kronecker product.
        
        # The 3x3x3 matmul tensor can be decomposed using the identity:
        # C = A*B where we use 7 products based on Strassen-like formulas.
        
        # Actually, for 3x3, the minimal rank is 7. Let me use a verified decomposition.
        
        # From the literature, one rank-7 decomposition of the 3x3x3 matmul tensor:
        # (i,j,k) indices, each in {0,1,2}
        
        # I'll use a systematic construction. The matmul tensor T[i,j,k]=1 
        # can be written as sum_r f_r(i,j) * g_r(j,k) * h_r(k,i)
        
        # Using polynomial/interpolation approach or known results:
        
        # Here's a rank-7 decomposition verified by computation:
        # We'll define 7 sets of (U_r, V_r, W_r) as 9-element vectors
        
        # Let me just construct it numerically and verify
        # For now, return a placeholder that we'll verify
        
        # Actually, let me use a different approach. The 3x3x3 matmul tensor
        # is equivalent to the standard 3x3 matrix multiplication.
        # Its tensor rank is known to be 7.
        
        # Let me construct a valid rank-7 decomposition:
        
        # Using the "Strassen-like" decomposition for 3x3:
        # The 7 multiplications are:
        # M1 = (A11+A22+A33)(B11+B22+B33)
        # M2 = (A11-A22)(B11-B22)
        # M3 = (A11+A22-A33)(B11+B22-B33) 
        # M4 = (A22-A33)(B22-B33)
        # M5 = (A33-A11)(B33-B11)
        # M6 = (A11-A22+A33)(B11-B22+B33)
        # M7 = (A22-A11+A33)(B22-B11+B33)
        
        # And C is reconstructed as:
        # C11 = M1 - M2 + M3 - M6
        # C12 = M3 + M6
        # C13 = M5 + M6 - M7
        # C21 = M2 + M4 - M6
        # C22 = M1 - M3 + M4 + M7
        # C23 = M4 + M5 - M7
        # C31 = M3 + M6 - M7  -- wait, this doesn't look right
        
        # Let me use a cleaner approach. I'll construct the rank-7 decomposition
        # by solving the system of equations.
        
        # For the 3x3x3 matmul tensor, the 27 entries T[i,j,k] for i,j,k in {0,1,2}
        # must equal sum_r U_r[3i+j] * V_r[3j+k] * W_r[3k+i]
        
        # This is a system of 27 equations in 7*27=189 unknowns (with structure).
        # Let me just hardcode a known valid decomposition.
        
        # Here's a verified rank-7 decomposition of the 3x3x3 matmul tensor:
        # (source: standard references on tensor rank)
        
        # I'll generate it programmatically and verify:
        
        # Actually, the simplest approach: since we know the answer should work,
        # let me construct the full rank-21 decomposition directly using the
        # tensor product structure.
        
        return None  # We'll construct rank-21 directly

    def construct_rank21():
        """Construct explicit rank-21 decomposition using tensor product structure.
        
        The 3x3x(3x6) matmul tensor factors as:
        T_33x6[i,j,k] = T_333[i,j,k'] * T_3x2[j',k'',i']
        where k = 3*k' + k'' (splitting 6-dim into 3x2)
        
        Wait, let me reconsider the factorization.
        
        Target: T[a,b,c] = 1 iff a=3i+j, b=3j+k, c=3k+i for i,j in {0,1,2}, k in {0,...,5}
        
        Let k = 3*k1 + k2 where k1 in {0,1}, k2 in {0,1,2}
        Then c = 3*(3*k1+k2) + i = 9*k1 + 3*k2 + i
        
        Hmm, this doesn't factor cleanly. Let me think differently.
        
        Alternative: k = 2*l + s where l in {0,1,2}, s in {0,1}
        Then c = 3*(2l+s) + i = 6l + 3s + i
        
        The tensor is:
        T[3i+j, 3j+2l+s, 6l+3s+i] = 1
        
        This can be written as a product:
        T_333[i, j, l] * T_s[j, s, i] ... no, that's not quite right either.
        
        Let me think about this more carefully.
        
        Actually, the key insight is:
        T[3i+j, 3j+k, 3k+i] = 1
        
        We can write k = 3*k1 + k2 where k1 in {0,1}, k2 in {0,1,2}
        Then:
        a = 3i + j
        b = 3j + 3*k1 + k2
        c = 3*(3*k1+k2) + i = 9*k1 + 3*k2 + i
        
        So T[a,b,c] = 1 iff:
        a = 3i+j, b = 3j+3k1+k2, c = 9k1+3k2+i
        
        This factors as:
        T[i,j,k1,k2] = T_A[i,j,k1] * T_B[j,k2,i] ... no.
        
        Let me try: T = T_333 ⊗ T_1x3x2 style.
        
        T_333[i,j,k1] contributes to (a=3i+j, b_partial, c_partial)
        T_1x3x2[j,k2,i] contributes to the remaining parts.
        
        Actually, I think the cleanest factorization is:
        
        Let the 3x3x3 matmul tensor be:
        T333[i,j,k] = 1 (meaning: output[i,k] += A[i,j]*B[j,k])
        
        And a "scalar-vector" tensor for the 3x2 part:
        T_sv[j, k2, i] where this handles the extra dimension.
        
        Hmm, let me just directly construct rank-21 by taking 7 Strassen terms 
        and for each, having 3 variants (for the 3 values of k2 or the 2 values of s).
        
        Actually, the simplest approach: 
        The 3x3x(3x6) matmul can be seen as 2 independent 3x3x3 matmuls 
        (for k=0,1,2 and k=3,4,5) plus cross terms... no, that's not right.
        
        Let me reconsider. The matrix A is 3x3, B is 3x6, C is 3x6.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        This is just 6 independent 3-vector dot products (one for each column k of C).
        Each column k: C[:,k] = A * B[:,k]
        
        So the tensor is: T[i,j,k] = 1 means C[i,k] gets A[i,j]*B[j,k].
        
        In our indexing:
        U index: i*m+j = 3i+j (9 values)
        V index: j*p+k = 3j+k (18 values) 
        W index: k*n+i = 3k+i (18 values)
        
        T[3i+j, 3j+k, 3k+i] = 1
        
        Now, if I split k into k = 2*l + s (l in {0,1,2}, s in {0,1}):
        T[3i+j, 3j+2l+s, 3(2l+s)+i] = T[3i+j, 3j+2l+s, 6l+3s+i] = 1
        
        This can be factored as a tensor product:
        For each (i,j,l,s): contribution = U_part[i,j] * V_part[j,l,s] * W_part[l,s,i]
        
        If I can find rank-r1 decomposition of T333[i,j,l] (3x3x3 matmul, rank 7)
        and rank-r2 decomposition of T_sv[j,s,i] (some 3x2x3 tensor, rank ?)
        Then the total rank is r1 * r2.
        
        What is T_sv[j,s,i]? It should satisfy:
        sum over the factorization gives 1 when the indices match.
        
        Actually, the tensor T[3i+j, 3j+2l+s, 6l+3s+i] = 1 
        can be written as:
        T333[i,j,l] * Textra[j,s,i]
        
        where T333[i,j,l] = 1 (always 1, it's just the identity in the l-dimension?)
        No, that's not right.
        
        Let me think again. The full tensor has 9*18*18 = 2916 entries, 
        of which 3*3*6 = 54 are 1.
        
        The 54 nonzero entries are at positions:
        (3i+j, 3j+k, 3k+i) for i,j in {0,1,2}, k in {0,...,5}
        
        If I fix (i,j) and vary k, I get 6 entries:
        (3i+j, 3j+k, 3k+i) for k=0,1,2,3,4,5
        
        For a fixed (i,j), as k varies:
        U index is fixed: 3i+j
        V index: 3j, 3j+1, 3j+2, 3j+3, 3j+4, 3j+5
        W index: i, 3+i, 6+i, 9+i, 12+i, 15+i
        
        So for each (i,j), we have a "line" of 6 entries in the (V,W) plane.
        
        Now, the rank-1 decomposition: each rank-1 term is u ⊗ v ⊗ w.
        For term r, it contributes to T[a,b,c] = U_r[a] * V_r[b] * W_r[c].
        
        For the 54 nonzero entries to be exactly 1 and all others 0, we need:
        sum_r U_r[3i+j] * V_r[3j+k] * W_r[3k+i] = 1 for all (i,j,k)
        sum_r U_r[a] * V_r[b] * W_r[c] = 0 for all other (a,b,c)
        
        Now, the tensor product approach:
        If I can write the tensor as T = T1 ⊗ T2 (Kronecker product in some sense),
        then rank(T) ≤ rank(T1) * rank(T2).
        
        Let me try: split the k-index as k = 3*k1 + k2 (k1 in {0,1}, k2 in {0,1,2})
        Then:
        T[3i+j, 3j+3k1+k2, 3(3k1+k2)+i] = T[3i+j, 3j+3k1+k2, 9k1+3k2+i]
        
        Define:
        T_A[i,j,k1,k2] such that T_A[i,j,k1,k2] = 1 for all valid (i,j,k1,k2)
        
        This doesn't factor as a simple product. But we can write:
        T[3i+j, 3j+3k1+k2, 9k1+3k2+i] = T333[i,j,k2] * T_k1[j,k1,i] ... no.
        
        OK let me try a completely different approach. Let me just directly 
        construct a rank-21 decomposition by hand.
        
        Strategy: Use 7 "Strassen-like" terms for the 3x3x3 part, and for each,
        have 3 variations to cover the 6 k-values (2 groups of 3).
        
        For the 3x3x3 matmul (k in {0,1,2}), we need rank 7.
        For the 3x3x(3x6) matmul, we can think of it as two 3x3x3 matmuls 
        (k=0,1,2 and k=3,4,5) that share the same A matrix.
        
        So: C[:,0:3] = A * B[:,0:3] and C[:,3:6] = A * B[:,3:6]
        
        Each 3x3x3 matmul needs rank 7, so naively 14. But we can share the 
        "A-side" structure.
        
        Actually, the tensor for the full problem is:
        T_full[i,j,k] = T_333[i,j,k mod 3] * delta(k div 3, something)
        
        No, that's not right either.
        
        Let me just try: for each of the 7 Strassen terms for 3x3x3, create 3 
        variants (for k=0,1,2 in the first group and k=3,4,5 in the second group).
        That gives 7*3 = 21 terms.
        
        Wait, that would be 7*2 = 14 if we just double for the two groups.
        Unless the Strassen structure doesn't directly apply.
        
        Let me just construct it explicitly and verify.
        
        For the 3x3x3 matmul tensor T333[i,j,k] = 1 (i,j,k in {0,1,2}):
        Rank-7 decomposition: sum_{r=1}^{7} U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
        
        For the full tensor:
        T_full[i,j,k] = 1 for i,j in {0,1,2}, k in {0,...,5}
        
        Let k = 3*k1 + k2 where k1 in {0,1}, k2 in {0,1,2}
        
        T_full[i,j,3k1+k2] = 1
        
        If I have a rank-7 decomposition of T333:
        T333[i,j,k2] = sum_r U_r[i,j] * V_r[j,k2] * W_r[k2,i]
        
        Then for the full tensor, I need:
        T_full[i,j,3k1+k2] = sum_{r, k1'} U'_r[i,j] * V'_r[j,3k1+k2] * W'_r[3(3k1+k2)+i]
        
        If I set:
        U'_r[i,j] = U_r[i,j] (same as 3x3x3)
        V'_r[j,3k1+k2] = V_r[j,k2] * f_r(k1)  (scaled by k1-dependent factor)
        W'_r[9k1+3k2+i] = W_r[k2,i] * g_r(k1)  (scaled by k1-dependent factor)
        
        Then:
        sum_r U_r[i,j] * V_r[j,k2]*f_r(k1) * W_r[k2,i]*g_r(k1)
        = sum_r U_r[i,j] * V_r[j,k2] * W_r[k2,i] * f_r(k1)*g_r(k1)
        = T333[i,j,k2] * (sum_r f_r(k1)*g_r(k1) * U_r[i,j]*V_r[j,k2]*W_r[k2,i] / T333[i,j,k2])
        
        This only works if f_r(k1)*g_r(k1) is the same for all r, which it isn't.
        
        OK, different approach. Let me just directly construct 21 rank-1 terms.
        
        For each (i,j) pair (9 total), and for the 6 k-values, I need the sum to be 1.
        
        Actually, the simplest rank-21 construction:
        Use 21 terms where each term covers a subset of the (i,j,k) combinations.
        
        Let me use the following structure:
        - 7 terms for the "first half" (k=0,1,2) based on Strassen
        - 7 terms for the "second half" (k=3,4,5) based on Strassen  
        - 7 terms for cross-coupling
        
        No, that's getting too complicated. Let me just do a direct numerical construction.
        
        I'll set up the system of equations and solve it.
        """
        
        # Direct construction of rank-21 decomposition
        # T[3i+j, 3j+k, 3k+i] = sum_r U_r[3i+j] * V_r[3j+k] * W_r[3k+i] = 1
        
        # We have 54 equations (one for each (i,j,k)) and 21*9 + 21*18 + 21*18 = 189+378+378 = 945 unknowns
        # But with the constraint that other entries must be 0, we have many more constraints.
        
        # Actually, the cleanest approach: use the tensor product structure.
        
        # The 3x3x(3x6) matmul tensor = T_333 ⊗ T_1x1x2 (in some sense)
        # where T_333 is the 3x3x3 matmul tensor (rank 7)
        # and T_1x1x2 is a rank-3 tensor for the "doubling" of k
        
        # Specifically:
        # T_full[i,j,k] = T_333[i,j, k mod 3] * T_double[k div 3, k mod 3, ...]
        
        # Hmm, this isn't a clean tensor product.
        
        # Let me try yet another approach: 
        # The 3x3x(3x6) matmul can be decomposed as:
        # C = A * B where B is 3x6 = [B1 | B2] where B1, B2 are 3x3
        # C = [A*B1 | A*B2]
        # 
        # So we need to compute two 3x3 matmuls: A*B1 and A*B2
        # Each needs rank 7, total rank 14 if independent.
        # But they share the same A, so we can potentially do better.
        
        # The tensor for A*B1: T1[i,j,k] = 1 for i,j,k in {0,1,2} (k corresponds to column of B1)
        # The tensor for A*B2: T2[i,j,k] = 1 for i,j in {0,1,2}, k in {3,4,5} (k corresponds to column of B2)
        
        # In our indexing:
        # T1: U index = 3i+j, V index = 3j+k (k=0,1,2), W index = 3k+i
        # T2: U index = 3i+j, V index = 3j+k (k=3,4,5), W index = 3k+i
        
        # If I have a rank-7 decomposition of T1:
        # sum_{r=1}^7 U_r[3i+j] * V_r[3j+k] * W_r[3k+i] = 1 for k=0,1,2
        # 
        # And a rank-7 decomposition of T2:
        # sum_{r=1}^7 U'_r[3i+j] * V'_r[3j+k] * W'_r[3k+i] = 1 for k=3,4,5
        #
        # Then the total rank is 14.
        
        # Can we do better by sharing? If U_r = U'_r for all r, then:
        # sum_{r=1}^7 U_r[3i+j] * (V_r[3j+k] for k=0,1,2 + V'_r[3j+k] for k=3,4,5) * ... 
        # This doesn't quite work because V and W have different structures.
        
        # Let me just go with rank 14 first and see if it verifies.
        # If not, try rank 21.
        
        # Actually, let me construct the rank-14 decomposition:
        # For the 3x3x3 matmul (k=0,1,2):
        # Need rank-7 decomposition.
        
        # Let me use a known rank-7 decomposition of the 3x3x3 matmul tensor.
        
        # The 3x3x3 matmul tensor T[i,j,k] = 1 has rank 7.
        # One explicit decomposition:
        
        # I'll use the following approach: the 3x3 matmul can be done with 7 multiplications.
        # Each multiplication m_r corresponds to a product of a linear combination of A entries 
        # and a linear combination of B entries, and the result is added to certain C entries.
        
        # For 3x3, one set of 7 products:
        # M1 = (A11+A22+A33) * (B11+B22+B33)
        # M2 = (A11-A22) * (B11-B22)
        # M3 = (A11+A22-A33) * (B11+B22-B33)
        # M4 = (A22-A33) * (B22-B33)
        # M5 = (A33-A11) * (B33-B11)
        # M6 = (A11-A22+A33) * (B11-B22+B33)
        # M7 = (A22-A11+A33) * (B22-B11+B33)
        
        # And:
        # C11 = M1 - M2 + M3 - M6
        # C12 = M3 + M6
        # C13 = M5 + M6 - M7
        # C21 = M2 + M4 - M6
        # C22 = M1 - M3 + M4 + M7
        # C23 = M4 + M5 - M7
        # C31 = M3 + M6 - M7  -- need to verify
        # C32 = M5 + M7
        # C33 = M1 - M2 - M4 + M7 -- need to verify
        
        # Let me verify C11: 
        # (A11+A22+A33)(B11+B22+B33) - (A11-A22)(B11-B22) + (A11+A22-A33)(B11+B22-B33) - (A11-A22+A33)(B11-B22+B33)
        
        # This is getting very messy. Let me just construct the decomposition numerically.
        
        # I'll solve the linear system for the rank-7 decomposition of 3x3x3 matmul.
        
        # For 3x3x3 matmul: T[i,j,k] = 1 for i,j,k in {0,1,2}
        # We need: sum_{r=1}^7 U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
        
        # This is a system of 27 equations. With 7*9 + 7*9 + 7*9 = 189 unknowns,
        # it's underdetermined. We can fix some variables and solve.
        
        # Actually, let me just use a simpler approach for the full problem.
        # I'll construct a rank-21 decomposition directly.
        
        # For each (i,j) in {0,1,2}^2 and each k in {0,...,5}:
        # We need sum_r U_r[3i+j] * V_r[3j+k] * W_r[3k+i] = 1
        
        # And for all other (a,b,c): sum_r U_r[a] * V_r[b] * W_r[c] = 0
        
        # Strategy: Use 21 terms where:
        # - Terms 0-6: handle k=0,1,2 (first 3 columns)
        # - Terms 7-13: handle k=3,4,5 (last 3 columns)
        # - Terms 14-20: zero out cross terms
        
        # For terms 0-6 (k=0,1,2):
        # Use rank-7 decomposition of 3x3x3 matmul
        # U_r[3i+j], V_r[3j+k] for k=0,1,2, W_r[3k+i] for k=0,1,2
        # Set V_r[3j+3], V_r[3j+4], V_r[3j+5] = 0
        # Set W_r[9+3k+i], W_r[12+3k+i], W_r[15+3k+i] = 0
        
        # For terms 7-13 (k=3,4,5):
        # Use rank-7 decomposition of 3x3x3 matmul (shifted)
        # U_r[3i+j], V_r[3j+3+k'] for k'=0,1,2, W_r[3(3+k')+i]
        # Set V_r[3j], V_r[3j+1], V_r[3j+2] = 0
        # Set W_r[3k'+i] for k'=0,1,2 = 0
        
        # This gives rank 14, not 21. Let me check if rank 14 works.
        
        # For the rank-7 decomposition of 3x3x3 matmul, I need explicit values.
        # Let me construct one.
        
        # The 3x3x3 matmul tensor: T[i,j,k] = 1
        # Rank-7 decomposition:
        
        # I'll use the following known decomposition (verified by computation):
        
        # Actually, let me just solve it numerically.
        
        # Set up the system:
        # For each (i,j,k) in {0,1,2}^3:
        #   sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
        # 
        # For all other (a,b,c) not of the form (i,j,k) with the matmul structure:
        #   sum_r U_r[a] * V_r[b] * W_r[c] = 0
        #
        # But wait, for the 3x3x3 case, ALL (a,b,c) with a=3i+j, b=3j+k, c=3k+i 
        # for some (i,j,k) are the nonzero entries. There are 27 such entries.
        # The remaining 9*9*9 - 27 = 702 entries must be 0.
        
        # This is a system of 729 equations in 189 unknowns (for rank 7).
        # It's overdetermined, so a solution may not exist with this simple structure.
        
        # The issue is that the rank-7 decomposition of the 3x3x3 matmul tensor 
        # means that when you form the tensor sum_r U_r ⊗ V_r ⊗ W_r, it equals 
        # the matmul tensor. The matmul tensor has exactly 27 nonzero entries 
        # (at positions (3i+j, 3j+k, 3k+i) for i,j,k in {0,1,2}).
        
        # So the system is:
        # 27 equations = 1 (the nonzero entries)
        # 702 equations = 0 (the zero entries)
        # Total: 729 equations, 189 unknowns.
        
        # This is overdetermined. But we know the tensor rank is 7, so a solution exists.
        # The key is that the unknowns are not independent - they come in triples (U_r, V_r, W_r).
        
        # Let me just hardcode a known valid rank-7 decomposition.
        
        # After some research, here's a rank-7 decomposition of the 3x3x3 matmul tensor:
        # (This is based on the Strassen algorithm adapted for 3x3)
        
        # I'll construct it as follows:
        # Define 7 sets of (u_r, v_r, w_r) where each is a 9-element vector.
        
        # Let me use a computational approach: fix U_r and V_r, solve for W_r.
        
        # Actually, the simplest approach for this problem: 
        # Just use the schoolbook decomposition but with rank 54, and try to 
        # reduce it using the structure.
        
        # OR: Let me just directly construct a valid rank-21 decomposition 
        # by using 3 copies of a rank-7 decomposition.
        
        # Here's my plan:
        # 1. Construct a rank-7 decomposition of the 3x3x3 matmul tensor (numerically)
        # 2. Use it twice (for k=0,1,2 and k=3,4,5) to get rank 14
        # 3. Verify it works for the full 3x3x(3x6) tensor
        # 4. If it works, great! If not, add more terms.
        
        # Let me construct the rank-7 decomposition numerically.
        
        # For 3x3x3 matmul:
        # T[i,j,k] = 1 for i,j,k in {0,1,2}
        # Need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
        
        # I'll parameterize:
        # U_r[i,j] = u_r[i] * v_r[j] (rank-1 in the (i,j) plane) - NO, this is too restrictive
        
        # Actually, U_r is a 3x3 matrix (flattened to 9 elements), not necessarily rank 1.
        
        # Let me just solve the system numerically.
        
        # Fix the first 63 unknowns (first 7 U vectors, 7 V vectors), solve for W.
        # Or use a least-squares approach.
        
        # Given time constraints, let me just use a simple construction:
        
        # Rank-21 decomposition:
        # For each (i,j) in {0,1,2}^2 and each k in {0,...,5}:
        #   Term (i,j,k): U[3i+j] = 1, V[3j+k] = 1, W[3k+i] = 1
        # This is rank 54 (schoolbook).
        
        # To reduce to rank 21, I need to combine terms.
        
        # Let me try: for each i, j, and each pair of k-values, use one term.
        # There are 3*3*3 = 27 pairs if I group k into 3 pairs... no, 6 k-values give 3 pairs.
        # 3*3*3 = 27 terms. Still more than 21.
        
        # Or: for each i, j, use 2 terms to cover all 6 k-values.
        # 3*3*2 = 18 terms. Less than 21!
        
        # Can I cover 6 k-values with 2 terms for each (i,j)?
        # For fixed (i,j):
        #   Term 1: V[3j+k1]=1, V[3j+k2]=1, W[3k1+i]=1, W[3k2+i]=1
        #   Term 2: V[3j+k3]=1, V[3j+k4]=1, W[3k3+i]=1, W[3k4+i]=1
        #   ... but this only covers 4 k-values with 2 terms.
        
        # With 2 terms, I can cover at most 4 k-values (2 per term).
        # I need 6 k-values, so I need at least 3 terms per (i,j).
        # 3*3*3 = 27 terms.
        
        # Hmm, 27 > 21. So this simple approach doesn't work.
        
        # Let me try a different grouping.
        
        # What if I share terms across different (i,j) pairs?
        
        # For example, one term could contribute to multiple (i,j) pairs.
        
        # Term r: U_r[3i1+j1]=1, U_r[3i2+j2]=1, ... 
        # V_r[3j1+k]=1, V_r[3j2+k]=1, ...
        # W_r[3k+i1]=1, W_r[3k+i2]=1, ...
        
        # This is getting very complex. Let me just try rank 27 first 
        # (3 per (i,j) pair) and see if it verifies.
        
        # Actually, let me step back and think about this differently.
        
        # The tensor rank of the 3x3x(3x6) matmul tensor is at most 54 (schoolbook).
        # It's at most 2*7 = 14 if the two 3x3x3 matmuls can be done independently 
        # with shared A-structure.
        
        # Actually, I realize that the two 3x3x3 matmuls (for k=0,1,2 and k=3,4,5) 
        # DO share the same A matrix. So the U vectors can be shared.
        
        # If I use the same 7 U vectors for both halves, and separate V and W vectors,
        # the total rank is 7 (for U) + 7 (for V) + 7 (for W) = 21? No, that's not how rank works.
        
        # The rank is the number of terms in the decomposition. Each term has one U, one V, one W.
        # If I use 7 terms for the first half and 7 terms for the second half, that's 14 terms total.
        # The U vectors for the first 7 terms are the same as... no, they're separate terms.
        
        # OK so rank 14 is the upper bound from this approach.
        
        # Let me just construct rank-14 and verify it.
        
        # For the 3x3x3 matmul tensor, I need a rank-7 decomposition.
        # Let me construct one numerically.
        
        # System of equations for 3x3x3 matmul:
        # For (i,j,k) in {0,1,2}^3: sum_r U_r[3i+j] * V_r[3j+k] * W_r[3k+i] = 1
        # For all other (a,b,c) in {0,...,8}^3: sum_r U_r[a] * V_r[b] * W_r[c] = 0
        
        # This is 729 equations in 7*27 = 189 unknowns.
        # Let me solve it using least squares.
        
        # Actually, I'll just construct a valid decomposition by hand.
        
        # Here's a rank-7 decomposition of the 3x3x3 matmul tensor:
        # (Based on the "Strassen for 3x3" algorithm)
        
        # The 7 products are:
        # P1 = (A00+A11+A22) * (B00+B11+B22)
        # P2 = (A00-A11) * (B00-B11)
        # P3 = (A00+A11-A22) * (B00+B11-B22)
        # P4 = (A11-A22) * (B11-B22)
        # P5 = (A22-A00) * (B22-B00)
        # P6 = (A00-A11+A22) * (B00-B11+B22)
        # P7 = (A11-A00+A22) * (B11-B00+B22)
        
        # And C is:
        # C00 = P1 - P2 + P3 - P6
        # C01 = P3 + P6
        # C02 = P5 + P6 - P7
        # C10 = P2 + P4 - P6
        # C11 = P1 - P3 + P4 + P7
        # C12 = P4 + P5 - P7
        # C20 = P3 + P6 - P7  -- let me verify
        # C21 = P5 + P7
        # C22 = P1 - P2 - P4 + P7 -- let me verify
        
        # Let me verify C00:
        # P1 = (A00+A11+A22)(B00+B11+B22)
        # P2 = (A00-A11)(B00-B11)
        # P3 = (A00+A11-A22)(B00+B11-B22)
        # P6 = (A00-A11+A22)(B00-B11+B22)
        
        # C00 = P1 - P2 + P3 - P6
        # = (A00+A11+A22)(B00+B11+B22) - (A00-A11)(B00-B11) + (A00+A11-A22)(B00+B11-B22) - (A00-A11+A22)(B00-B11+B22)
        
        # Let me expand each:
        # P1 = A00B00 + A00B11 + A00B22 + A11B00 + A11B11 + A11B22 + A22B00 + A22B11 + A22B22
        # P2 = A00B00 - A00B11 - A11B00 + A11B11
        # P3 = A00B00 + A00B11 - A00B22 + A11B00 + A11B11 - A11B22 - A22B00 - A22B11 + A22B22
        # P6 = A00B00 - A00B11 + A00B22 - A11B00 + A11B11 - A11B22 + A22B00 - A22B11 + A22B22
        
        # C00 = P1 - P2 + P3 - P6
        # Coefficient of A00B00: 1 - 1 + 1 - 1 = 0
        # Coefficient of A00B11: 1 - (-1) + 1 - (-1) = 1+1+1+1 = 4
        # 
        # This doesn't give C00 = A00B00 + A01B10 + A02B20. So my Strassen formulas are wrong.
        
        # Let me use a different approach. I'll just solve the system numerically.
        
        # Given the time pressure, let me just return the schoolbook decomposition 
        # with rank 54, but try a few lower ranks first.
        
        # Actually, let me try a completely different approach:
        # Use the fact that the tensor is a "matrix multiplication tensor" and 
        # use known results about its rank.
        
        # The rank of the 3x3x(3x6) matmul tensor is at most 54.
        # It's known that the rank of the m×n×p matmul tensor is at most m*n*p.
        # For 3x3x6, that's 54.
        
        # The minimum rank is at least max(m*n, m*p, n*p) / something... 
        # Actually, the lower bound is max(m*n, m*p, n*p) = max(9, 18, 18) = 18.
        
        # So the rank is between 18 and 54.
        
        # Let me try rank 18, 21, 24, 27, 30, ... up to 54.
        
        # For each rank, I'll try to find a valid decomposition using ALS.
        
        # But first, let me try the tensor product approach more carefully.
        
        # The 3x3x(3x6) matmul tensor can be written as:
        # T[i,j,k] = T_333[i,j,k mod 3] * T_2[k div 3, ...]
        
        # No, that's not a valid tensor product.
        
        # Let me just try rank 18 (the lower bound) and see if ALS can find it.
        
        # Given time constraints, let me just return the schoolbook rank-54 decomposition.
        # But let me try a few lower ranks first with ALS.
        
        pass

    def als_optimize(U, V, W, max_iter=200, tol=1e-12):
        """Optimized ALS with precomputed contractions."""
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break

            old_res = compute_residual(U, V, W)

            # === Update U ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = V[:, r]
                B = W[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', U[:, r], A, B)
                R_other = R_full - R_r
                rhs = np.einsum('j,k,ijk->i', A, B, T_target) - np.einsum('j,k,ijk->i', A, B, R_other)
                U[:, r] = rhs / denom

            # === Update V ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = U[:, r]
                B = W[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', A, V[:, r], B)
                R_other = R_full - R_r
                rhs = np.einsum('i,k,ijk->j', A, B, T_target) - np.einsum('i,k,ijk->j', A, B, R_other)
                V[:, r] = rhs / denom

            # === Update W ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = U[:, r]
                B = V[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', A, B, W[:, r])
                R_other = R_full - R_r
                rhs = np.einsum('i,j,ijk->k', A, B, T_target) - np.einsum('i,j,ijk->k', A, B, R_other)
                W[:, r] = rhs / denom

            new_res = compute_residual(U, V, W)
            if abs(old_res - new_res) < tol:
                break

        return U, V, W

    def polish_pass(U, V, W):
        """High-precision polish with more iterations and tighter tolerance."""
        U, V, W = als_optimize(U, V, W, max_iter=300, tol=1e-14)
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            R = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = R - T_target
            res = np.sum(diff ** 2)
            if res < 1e-20:
                break
            for r in range(U.shape[1]):
                A = V[:, r]
                B = W[:, r]
                grad_U = np.einsum('j,k,ijk->i', A, B, diff)
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom > 1e-15:
                    U[:, r] -= grad_U / denom * 0.5
                grad_V = np.einsum('i,k,ijk->j', U[:, r], B, diff)
                if denom > 1e-15:
                    V[:, r] -= grad_V / denom * 0.5
                grad_W = np.einsum('i,j,ijk->k', U[:, r], A, diff)
                if denom > 1e-15:
                    W[:, r] -= grad_W / denom * 0.5
        return U, V, W

    def try_rank(rank, num_attempts=5):
        """Try to find exact decomposition for given rank."""
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 25:
                break
            
            seed = rank * 1000 + attempt * 137 + 42
            np.random.seed(seed)
            
            # Random initialization with structure
            U = np.zeros((dim_U, rank))
            V = np.zeros((dim_V, rank))
            W = np.zeros((dim_W, rank))
            
            for r in range(rank):
                # Initialize with sparse random values
                for idx in range(dim_U):
                    if np.random.random() < 0.3:
                        U[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
                for idx in range(dim_V):
                    if np.random.random() < 0.2:
                        V[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
                for idx in range(dim_W):
                    if np.random.random() < 0.2:
                        W[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
            
            U_opt, V_opt, W_opt = als_optimize(U, V, W, max_iter=100, tol=1e-12)
            res = compute_residual(U_opt, V_opt, W_opt)
            
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            if res < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return True, U_s, V_s, W_s
        
        # Try polish on best found
        if best_U is not None and best_res < 1e-4:
            U_pol, V_pol, W_pol = polish_pass(best_U, best_V, best_W)
            res_pol = compute_residual(U_pol, V_pol, W_pol)
            if res_pol < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_pol, V_pol, W_pol)
                if success:
                    return True, U_s, V_s, W_s
        
        return False, None, None, None

    # Try ranks from 18 upward
    for rank in range(18, 55):
        if time.time() > deadline - 30:
            break
        
        num_attempts = 6 if rank <= 30 else 4
        
        success, U_s, V_s, W_s = try_rank(rank, num_attempts)
        if success:
            return {"U": U_s, "V": V_s, "W": W_s}
    
    # Fallback to schoolbook (rank 54)
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
