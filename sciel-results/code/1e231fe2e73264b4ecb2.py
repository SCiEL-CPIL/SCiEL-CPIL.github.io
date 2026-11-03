"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen, then applies linear-algebra-based
rank reduction by finding the null space of the constraint system.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Builds a structured decomposition, then reduces rank via null-space analysis.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        # Schoolbook for k = k0+6
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Schoolbook = rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 25 (Strassen x3 + schoolbook)
          M_{1,2,7}: 14 (schoolbook)
        Total: 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split m=4 as 2+2:
          M_{3,2,7}: 39
          M_{3,2,7}: 39
        Total: 78
        """
        m327_decomp(i0, j0, k0)
        m327_decomp(i0, j0 + 2, k0)

    # Build the full decomposition
    m347_decomp(0, 0, 0)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness with a few random tests
    def verify_decomp(U_mat, V_mat, W_mat, rank_val, num_tests=3):
        """Verify the decomposition is exact."""
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for r in range(rank_val):
                            val += A[i, j] * B[j, k] * U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    C_computed[i, k] = val
            if np.max(np.abs(C_computed - C_expected)) > 1e-9:
                return False
        return True

    if not verify_decomp(U, V, W, rank):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sb = [[0] * rank_school for _ in range(n * m)]
        V_sb = [[0] * rank_school for _ in range(m * p)]
        W_sb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Linear-algebra-based rank reduction
    # The decomposition is exact iff for all (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We build a system: for each triple (i,j,k), the sum over r must equal 1.
    # If we can find a null-space vector c such that the perturbed decomposition
    # still satisfies all constraints, we can reduce rank.
    
    # Better approach: construct the constraint matrix directly.
    # For each output element (i,k) and input pair (j), the tensor entry T[i,j,k] = 1.
    # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Key insight: if two rank-1 terms are "parallel" in some sense, they can be merged.
    # More generally, we look for linear dependencies in the space of decompositions.
    
    # Approach: Build a matrix M where each row corresponds to a constraint (i,j,k)
    # and each column corresponds to a rank term r. But the constraint is nonlinear
    # (product of three vectors), so we need a different approach.
    
    # Alternative: Use the fact that the decomposition must satisfy:
    # For each (i,k): sum_j A[i,j] * (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) * B[j,k] = (A@B)[i,k]
    # This must hold for ALL A, B, so we need:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    
    # To find reducible terms, we check: can we write term r as a linear combination
    # of other terms? This is hard in general, but we can try a heuristic:
    # If removing term r and adjusting others still works, we can remove it.
    
    # More efficient: Build the system of equations. For each (i,j,k), we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p = 84 equations in rank unknowns (the entries of U,V,W).
    # The solution space has dimension >= rank - 84 if rank > 84, but our rank is 78 < 84.
    
    # Since rank (78) < number of constraints (84), the system is overdetermined.
    # The decomposition is exact, so all 84 constraints are satisfied.
    # To reduce rank, we need to find a subset of terms that still satisfies all constraints.
    
    # Practical approach: iteratively try removing each term and solving for corrections.
    # For term r to be removable, there must exist coefficients alpha_s (s != r) such that:
    # sum_{s != r} alpha_s * (U_s * V_s * W_s) = U_r * V_r * W_r
    # This is a system of 84 equations in (rank-1) unknowns.
    
    # Let's build this system for each candidate removal.
    
    def try_remove_term(U_mat, V_mat, W_mat, rank_val, term_idx):
        """
        Try to remove term term_idx by expressing it as a combination of others.
        Returns (success, new_U, new_V, new_W, new_rank) or (False, None, None, None, None).
        """
        if time.time() - start_time > deadline - 5.0:
            return False, None, None, None, None
        
        # Build the system: find alpha_s (s != term_idx) such that
        # sum_{s != term_idx} alpha_s * U[s] * V[s] * W[s] = U[term_idx] * V[term_idx] * W[term_idx]
        # where the product is elementwise in the 3D tensor space.
        
        # The "product" U[s]*V[s]*W[s] is a 3D tensor of size n x m x p.
        # We flatten it to a vector of length n*m*p = 84.
        
        nmp = n * m * p
        target = np.zeros(nmp)
        A_mat = np.zeros((nmp, rank_val - 1))
        
        col = 0
        for s in range(rank_val):
            if s == term_idx:
                continue
            # Compute the 3D tensor for term s
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A_mat[i*m*p + j*p + k, col] = U_mat[i*m+j, s] * V_mat[j*p+k, s] * W_mat[k*n+i, s]
            col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m*p + j*p + k] = U_mat[i*m+j, term_idx] * V_mat[j*p+k, term_idx] * W_mat[k*n+i, term_idx]
        
        # Solve A_mat * alpha = target
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, target, rcond=1e-10)
        except np.linalg.LinAlgError:
            return False, None, None, None, None
        
        # Check if the solution is good
        residual = np.linalg.norm(A_mat @ alpha - target)
        if residual > 1e-8:
            return False, None, None, None, None
        
        # Build new U, V, W by absorbing alpha into the remaining terms
        new_U = U_mat[:, [s for s in range(rank_val) if s != term_idx]].copy()
        new_V = V_mat[:, [s for s in range(rank_val) if s != term_idx]].copy()
        new_W = W_mat[:, [s for s in range(rank_val) if s != term_idx]].copy()
        
        # Scale each remaining term by its alpha
        for col_idx, s in enumerate([s for s in range(rank_val) if s != term_idx]):
            new_U[:, col_idx] *= alpha[col_idx]
            # We can't just scale U, V, W independently because the product must be preserved.
            # Actually, we need to distribute alpha among U, V, W.
            # The simplest: put all alpha in U.
            # But this changes the decomposition. Let me reconsider.
            
            # Actually, the constraint is:
            # sum_s alpha_s * (U_s * V_s * W_s) = U_r * V_r * W_r
            # If we set new_U_s = alpha_s * U_s, new_V_s = V_s, new_W_s = W_s,
            # then new_U_s * new_V_s * new_W_s = alpha_s * U_s * V_s * W_s, which is correct.
        
        # Verify the new decomposition
        if not verify_decomp(new_U, new_V, new_W, rank_val - 1, num_tests=2):
            return False, None, None, None, None
        
        return True, new_U, new_V, new_W, rank_val - 1

    # Iteratively try to remove terms
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    rank_cur = rank
    
    removed = True
    while removed:
        if time.time() - start_time > deadline - 5.0:
            break
        removed = False
        for r in range(rank_cur - 1, -1, -1):
            if time.time() - start_time > deadline - 5.0:
                break
            success, new_U, new_V, new_W, new_rank = try_remove_term(U_cur, V_cur, W_cur, rank_cur, r)
            if success:
                U_cur, V_cur, W_cur = new_U, new_V, new_W
                rank_cur = new_rank
                removed = True
                break
    
    # Final verification
    if not verify_decomp(U_cur, V_cur, W_cur, rank_cur):
        # Fallback to original
        U_cur, V_cur, W_cur = U, V, W
        rank_cur = rank
    
    return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
# EVOLVE-BLOCK-END
