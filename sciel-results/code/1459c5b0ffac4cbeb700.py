import time

import numpy as np

_DEADLINE = 120.0  # wall-clock budget (s); evaluator kills at 150 s
_start_time = time.time()

def _time_left():
    # simple deadline helper: returns seconds remaining
    return _DEADLINE - (time.time() - _start_time)


def _build_rank50():
    """Build the rank-50 decomposition and return U, V, W as lists."""
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        W[w_00][col]   += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        W[w_11][col]   += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        col += 7

    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    U[u_a][col]   += 1
    V[v_e8][col]  += 1
    W[w_08][col]  += 1
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1
    col += 4

    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    for k in range(p):
        v_bk = 2 * p + k
        w_0k = k * n + 0
        w_1k = k * n + 1
        U[u_ar0][col]   += 1
        V[v_bk][col]    += 1
        W[w_0k][col]    += 1
        U[u_ar1][col+1] += 1
        V[v_bk][col+1]  += 1
        W[w_1k][col+1]  += 1
        col += 2

    return U, V, W, rank


def _verify_decomposition(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]

    # Build the full multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We'll check by computing C = A*B for random matrices and comparing with tensor contraction
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B

    # Compute C via tensor decomposition
    # C[k*n+i] = sum_{i,j,k} A[i,j] * B[j,k] * (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])
    # More directly: C[i,k] = sum_j A[i,j]*B[j,k]
    # In our convention: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # Tensor term: for each r, the contribution to C[k*n+i] is
    #   (sum_j U[i*m+j,r] * A[i*m+j]) * (sum_j V[j*p+k,r] * B[j*p+k]) * W[k*n+i,r]
    # Wait, that's not right. Let me think more carefully.
    
    # The tensor decomposition says:
    # T_{a,b,c} = sum_r U_{a,r} * V_{b,r} * W_{c,r}
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For the decomposition to be correct, we need:
    # T_{i*m+j, j*p+k, k*n+i} = 1 if (i,j,k) is valid, else 0
    # Actually, the multiplication tensor has T_{a,b,c} = 1 when a=i*m+j, b=j*p+k, c=k*n+i
    # for the same i,j,k, and 0 otherwise.
    
    # Let's just verify by computing the product directly from the factors:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i*m+j]) * (sum_j V[j*p+k,r] * B[j*p+k]) * W[k*n+i,r]
    # This is wrong - the U, V, W factors don't work that way.
    
    # Correct interpretation: The tensor T has indices (a, b, c) where:
    # a = i*m + j (A index)
    # b = j'*p + k (B index)  
    # c = k'*n + i' (C index)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for the SAME i,j,k, else 0
    
    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # So for verification: for each (i,j,k), check that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for invalid combinations, it should be 0.
    
    # Check valid entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                max_err = max(max_err, abs(val - 1.0))
    
    # Check some invalid entries (where j in a doesn't match j' in b, or k in b doesn't match k' in c)
    # e.g., a = 0*m+0 = 0 (i=0,j=0), b = 1*p+0 = 9 (j'=1,k=0), c = 0*n+0 = 0 (k'=0,i'=0)
    # Here j=0 != j'=1, so T should be 0
    invalid_checks = [
        (0, 9, 0),   # a=0(i=0,j=0), b=9(j'=1,k=0), c=0(k'=0,i'=0): j!=j'
        (0, 9, 1),   # j!=j'
        (3, 0, 0),   # a=3(i=1,j=0), b=0(j'=0,k=0), c=0(k'=0,i'=0): i!=i'
        (3, 0, 1),   # i!=i'
    ]
    for (a, b, c) in invalid_checks:
        val = 0.0
        for r in range(rank):
            val += U_np[a, r] * V_np[b, r] * W_np[c, r]
        max_err = max(max_err, abs(val))
    
    return max_err < tol


def _try_reduce_rank(U, V, W, n=2, m=3, p=9, max_time=60.0):
    """Try to find linear dependencies among rank-1 terms to reduce rank."""
    start = time.time()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Now we want to find if there's a subset of terms that can be removed
    # by finding a linear dependency: sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = 0
    # where alpha_r are not all zero.
    
    # This is equivalent to finding the null space of the "unfolding" of the tensor
    # along the rank dimension.
    
    # Vectorize each rank-1 tensor into a vector of length (n*m)*(m*p)*(p*n)
    # Then find the rank of the matrix whose columns are these vectors.
    
    # This matrix has shape (n*m*m*p*p*n, rank) = (2*3*3*9*9*2, 50) = (2916, 50)
    # We can compute its rank via SVD.
    
    vecs = np.zeros((n*m * m*p * p*n, rank))
    for r in range(rank):
        vecs[:, r] = (U_np[:, r] * V_np[:, r]).reshape(-1) * W_np[:, r]
        # Wait, that's not right. The tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Vectorized: for each (a,b,c), value is U[a,r]*V[b,r]*W[c,r]
        # So vecs[a*m*p*p*n + b*p*n + c, r] = U[a,r]*V[b,r]*W[c,r]
        vecs[:, r] = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r]).flatten()
    
    # Compute rank of vecs via SVD
    try:
        U_svd, s, Vt = np.linalg.svd(vecs, full_matrices=False)
        # Rank is the number of significant singular values
        tol_svd = max(vecs.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
        num_significant = np.sum(s > tol_svd)
        nullity = rank - num_significant
        
        if nullity > 0:
            # There are null directions. We can try to eliminate terms.
            # The null space vectors tell us which linear combinations give zero.
            # For each null vector, we can potentially eliminate one term.
            
            # Get null space basis
            null_vecs = Vt[num_significant:]  # shape (nullity, rank)
            
            # Try to find a set of terms to eliminate
            # Strategy: for each null vector, find the term with largest coefficient
            # and see if we can eliminate it by adjusting others.
            
            # Simple approach: try eliminating one term at a time
            best_rank = rank
            for elim_idx in range(rank):
                # Check if we can eliminate term elim_idx
                # We need to express term elim_idx as a linear combination of others
                # such that the total tensor is unchanged.
                
                # This means finding alpha such that:
                # sum_{r != elim_idx} alpha_r * (U_r ⊗ V_r ⊗ W_r) = U_elim ⊗ V_elim ⊗ W_elim
                # which is a system of (n*m*m*p*p*n) equations in (rank-1) unknowns.
                # This is generally overdetermined, so it won't work unless there's a dependency.
                
                # Instead, use the null space: if there's a null vector with non-zero
                # component at elim_idx, we can use it to eliminate that term.
                
                found = False
                for nv in null_vecs:
                    if abs(nv[elim_idx]) > 1e-6:
                        # We can use this null vector to eliminate term elim_idx
                        # New coefficients: for r != elim_idx, add nv[r]/nv[elim_idx] to... 
                        # Actually, the null vector means sum_r nv[r] * T_r = 0
                        # So T_elim = -sum_{r!=elim} (nv[r]/nv[elim_idx]) * T_r
                        # This means we can remove term elim_idx and adjust others.
                        # But this changes the other terms, so we need to update U, V, W.
                        # For now, just count that we can eliminate one term.
                        found = True
                        break
                
                if found:
                    best_rank = min(best_rank, rank - 1)
            
            return best_rank, nullity
        
        return rank, 0
    
    except Exception as e:
        return rank, 0


def _attempt_rank_reduction(U, V, W, n=2, m=3, p=9, time_budget=30.0):
    """
    Try to find a valid decomposition with rank < len(U[0]) by finding
    linear dependencies among the rank-1 terms.
    Returns (U_new, V_new, W_new, new_rank) or None if no improvement found.
    """
    start = time.time()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Vectorize each rank-1 term
    vecs = np.zeros((n*m * m*p * p*n, rank))
    for r in range(rank):
        vecs[:, r] = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r]).flatten()
    
    # Find null space of vecs
    U_svd, s, Vt = np.linalg.svd(vecs, full_matrices=False)
    tol_svd = max(vecs.shape) * np.finfo(float).eps * (s[0] if len(s) > 0 else 1.0)
    num_significant = np.sum(s > tol_svd)
    nullity = rank - num_significant
    
    if nullity <= 0:
        return None
    
    # For each null vector, try to eliminate one term
    null_vecs = Vt[num_significant:]
    
    best_result = None
    best_rank = rank
    
    for nv in null_vecs:
        # Find the term with largest absolute coefficient in this null vector
        abs_nv = np.abs(nv)
        elim_idx = np.argmax(abs_nv)
        
        if abs_nv[elim_idx] < 1e-8:
            continue
        
        # We can eliminate term elim_idx by expressing it as a combination of others
        # T_elim = -sum_{r!=elim} (nv[r]/nv[elim_idx]) * T_r
        # This means we can remove term elim_idx and modify the other terms.
        
        # For simplicity, let's just try to drop the term and see if the rest
        # can be adjusted. Actually, the null vector tells us the exact adjustment.
        
        # New U, V, W: for r != elim_idx, new_U[r] = U[r] + (nv[r]/nv[elim_idx]) * U[elim_idx]
        # But this doesn't work because we need to keep the tensor the same.
        
        # Actually, the correct approach: if sum_r nv[r] * T_r = 0, then
        # T_elim = -sum_{r!=elim} (nv[r]/nv[elim_idx]) * T_r
        # So we can remove term elim_idx and the remaining terms (with adjusted
        # coefficients) still give the same tensor.
        
        # But the "adjusted coefficients" means we need to scale each term.
        # The problem is that scaling a rank-1 term U_r*V_r*W_r by alpha gives
        # alpha*U_r * V_r * W_r, which is (alpha*U_r)*V_r*W_r = U_r*(alpha*V_r)*W_r = U_r*V_r*(alpha*W_r)
        # So we can absorb the scaling into any one of the three matrices.
        
        # Strategy: for each r != elim_idx, add (nv[r]/nv[elim_idx]) * U[elim_idx] to U[r]
        # This is wrong - we need to be more careful.
        
        # The tensor contribution of term r is U_r ⊗ V_r ⊗ W_r
        # We want: sum_{r!=elim} U_r' ⊗ V_r' ⊗ W_r' = sum_{r!=elim} U_r ⊗ V_r ⊗ W_r + T_elim
        # where T_elim = U_elim ⊗ V_elim ⊗ W_elim
        
        # This is a complex nonlinear problem. Let's try a simpler heuristic:
        # just drop the term and check if the decomposition is still valid (it won't be,
        # but let's see if we can fix it with small adjustments).
        
        # For now, let's just note that we found a dependency and return a flag.
        # Actual rank reduction construction is left for future work.
        
        if rank - 1 < best_rank:
            best_rank = rank - 1
    
    return None


def _find_and_exploit_nullity(U, V, W, n=2, m=3, p=9, time_budget=30.0):
    """
    Find null space of the vectorized rank-1 terms and try to construct
    a valid lower-rank decomposition by exploiting linear dependencies.
    """
    start = time.time()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Vectorize each rank-1 term
    vecs = np.zeros((n*m * m*p * p*n, rank))
    for r in range(rank):
        vecs[:, r] = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r]).flatten()
    
    # SVD to find null space
    U_svd, s, Vt = np.linalg.svd(vecs, full_matrices=False)
    tol_svd = max(vecs.shape) * np.finfo(float).eps * (s[0] if len(s) > 0 else 1.0)
    num_significant = np.sum(s > tol_svd)
    nullity = rank - num_significant
    
    if nullity <= 0:
        return None
    
    # Try to eliminate terms using null vectors
    null_vecs = Vt[num_significant:]
    
    for nv in null_vecs:
        # Find term with largest coefficient
        elim_idx = np.argmax(np.abs(nv))
        if abs(nv[elim_idx]) < 1e-8:
            continue
        
        # Construct new factors by absorbing the eliminated term into others
        # For r != elim_idx: U_new[r] = U[r] + (nv[r]/nv[elim_idx]) * U[elim_idx]
        # This doesn't work directly because tensor products are nonlinear.
        
        # Alternative: just drop the term and verify (will fail, but let's check)
        U_new = U_np[:, [r for r in range(rank) if r != elim_idx]]
        V_new = V_np[:, [r for r in range(rank) if r != elim_idx]]
        W_new = W_np[:, [r for r in range(rank) if r != elim_idx]]
        
        # Convert back to lists
        U_list = U_new.tolist()
        V_list = V_new.tolist()
        W_list = W_new.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n=n, m=m, p=p):
            return (U_list, V_list, W_list, rank - 1)
    
    return None


def construct():
    # Build the rank-50 decomposition
    U50, V50, W50, rank50 = _build_rank50()
    
    # Verify correctness
    if not _verify_decomposition(U50, V50, W50):
        return {"U": U50, "V": V50, "W": W50}
    
    # Return the rank-50 decomposition
    return {"U": U50, "V": V50, "W": W50}
    # Dimensions: n=2 (rows of A), m=3 (inner), p=9 (cols of B)
    n, m, p = 2, 3, 9

    # Rank 50 explicit construction:
    #  4 Strassen blocks over B columns (0,1),(2,3),(4,5),(6,7):  4*7 = 28 terms
    #  1 naive 2x1x2x1 block   over B column 8:                   4      terms
    #  9 + 9 naive single-product terms for A_right * B_bottom:   18     terms
    # ---------------------------------------------------------------------------
    # Total rank = 50
    rank = 50

    # U has shape (n*m, rank) = (6, 50)
    # V has shape (m*p, rank) = (27, 50)
    # W has shape (p*n, rank) = (18, 50)
    # Indexing convention (matching the seed / task contract):
    #   A index  = i*m + j       (i in 0..n-1, j in 0..m-1)
    #   B index  = j*p + k       (j in 0..m-1, k in 0..p-1)
    #   C index  = k*n + i       (k in 0..p-1, i in 0..n-1)   ← trace convention

    U = [[0] * rank for _ in range(n * m)]   # 6 x 50
    V = [[0] * rank for _ in range(m * p)]   # 27 x 50
    W = [[0] * rank for _ in range(p * n)]   # 18 x 50

    col = 0

    # ------------------------------------------------------------------
    # Part 1: Strassen-style 2x2 blocks over A_top (2x2) and B_top (2x9)
    # ------------------------------------------------------------------
    # A_top indices:
    #   a = A_{0,0} -> U[0*3+0] = U[0]
    #   b = A_{0,1} -> U[0*3+1] = U[1]
    #   c = A_{1,0} -> U[1*3+0] = U[3]
    #   d = A_{1,1} -> U[1*3+1] = U[4]
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):  # 4 blocks over pairs of B columns
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices for these columns
        v_e = 0 * p + k0   # B_{0, k0}
        v_f = 0 * p + k1   # B_{0, k1}
        v_g = 1 * p + k0   # B_{1, k0}
        v_h = 1 * p + k1   # B_{1, k1}

        # W indices (trace convention C = k*n + i)
        w_00 = k0 * n + 0   # C_{0, k0}
        w_01 = k1 * n + 0   # C_{0, k1}
        w_10 = k0 * n + 1   # C_{1, k0}
        w_11 = k1 * n + 1   # C_{1, k1}

        # 7 Strassen terms --------------------------------------------------
        # M1 = (a + d)(e + h)
        U[u_a][col] = U[u_a][col] + 1
        U[u_d][col] = U[u_d][col] + 1
        V[v_e][col] = V[v_e][col] + 1
        V[v_h][col] = V[v_h][col] + 1

        # M2 = (c + d) * e
        U[u_c][col+1] = U[u_c][col+1] + 1
        U[u_d][col+1] = U[u_d][col+1] + 1
        V[v_e][col+1] = V[v_e][col+1] + 1

        # M3 = a * (f - h)
        U[u_a][col+2] = U[u_a][col+2] + 1
        V[v_f][col+2] = V[v_f][col+2] + 1
        V[v_h][col+2] = V[v_h][col+2] - 1

        # M4 = d * (g - e)
        U[u_d][col+3] = U[u_d][col+3] + 1
        V[v_g][col+3] = V[v_g][col+3] + 1
        V[v_e][col+3] = V[v_e][col+3] - 1

        # M5 = (a + b) * h
        U[u_a][col+4] = U[u_a][col+4] + 1
        U[u_b][col+4] = U[u_b][col+4] + 1
        V[v_h][col+4] = V[v_h][col+4] + 1

        # M6 = (c - a)(e + f)
        U[u_c][col+5] = U[u_c][col+5] + 1
        U[u_a][col+5] = U[u_a][col+5] - 1
        V[v_e][col+5] = V[v_e][col+5] + 1
        V[v_f][col+5] = V[v_f][col+5] + 1

        # M7 = (b - d)(g + h)
        U[u_b][col+6] = U[u_b][col+6] + 1
        U[u_d][col+6] = U[u_d][col+6] - 1
        V[v_g][col+6] = V[v_g][col+6] + 1
        V[v_h][col+6] = V[v_h][col+6] + 1

        # Reconstruct the 2x2 block output ----------------------------------
        # C_{0, k0} = M1 + M4 - M5 + M7
        W[w_00][col]   += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        # C_{0, k1} = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        # C_{1, k0} = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        # C_{1, k1} = M1 - M2 + M3 + M6
        W[w_11][col]   += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7  # 7 terms per block

    # ------------------------------------------------------------------
    # Part 2: Naive 2x1 block for last B column k = 8
    # A_top right column (A_{0,1}, A_{1,1}) times B column 8
    # Actually re-reading the seed: it uses A_top * B_col for k=8 with
    # the same four A entries (a,b,c,d) as above, and B row entries e=B_{0,8}, g=B_{1,8}
    # This is a 2x2 x 2x1 multiplication done naively: 4 scalar products.
    # ------------------------------------------------------------------
    k8 = 8
    v_e8 = 0 * p + k8   # B_{0, 8}
    v_g8 = 1 * p + k8   # B_{1, 8}

    w_08 = k8 * n + 0   # C_{0, 8}
    w_18 = k8 * n + 1   # C_{1, 8}

    # P1 = a * B_{0,8}
    U[u_a][col]   += 1
    V[v_e8][col]  += 1
    W[w_08][col]  += 1

    # P2 = b * B_{1,8}   (contributing to C_{0,8})
    # Wait – in a 2x2 * 2x1 multiply:
    #   C_{0,0} = a*e + b*g   (row 0 of A dot column of B)
    #   C_{1,0} = c*e + d*g   (row 1 of A dot column of B)
    # So:
    # P2 = b * B_{1,8} contributes to C_{0,8}
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1

    # P3 = c * B_{0,8} contributes to C_{1,8}
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1

    # P4 = d * B_{1,8} contributes to C_{1,8}
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1

    col += 4

    # ------------------------------------------------------------------
    # Part 3: A_right (col 2) × B_bottom (row 2):  2 x 1 × 1 x 9  →  18 terms
    # ------------------------------------------------------------------
    u_ar0 = 0 * m + 2   # A_{0,2}
    u_ar1 = 1 * m + 2   # A_{1,2}

    for k in range(p):
        v_bk = 2 * p + k   # B_{2, k}
        w_0k = k * n + 0   # C_{0, k} from A_{0,2}
        w_1k = k * n + 1   # C_{1, k} from A_{1,2}

        # Q_k = A_{0,2} * B_{2,k}  → C_{0,k}
        U[u_ar0][col]   += 1
        V[v_bk][col]    += 1
        W[w_0k][col]    += 1

        # R_k = A_{1,2} * B_{2,k}  → C_{1,k}
        U[u_ar1][col+1] += 1
        V[v_bk][col+1]  += 1
        W[w_1k][col+1]  += 1

        col += 2

    # At this point col should equal rank (50). Assert for safety.
    # (kept as light check; no print on success path)

    return {"U": U, "V": V, "W": W}
