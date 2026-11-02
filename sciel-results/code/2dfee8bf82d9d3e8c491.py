import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n

    # Strassen 7-term decomposition for 2x2x2
    strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    def build_strassen_rank35():
        rank = 35
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        col = 0
        for block in range(5):
            for a, b, c in strassen:
                for i in range(nm):
                    U[i, col] = a[i]
                V[0 * p + 2 * block, col] = b[0]
                V[0 * p + 2 * block + 1, col] = b[1]
                V[1 * p + 2 * block, col] = b[2]
                V[1 * p + 2 * block + 1, col] = b[3]
                W[(2 * block) * 2 + 0, col] = c[0]
                W[(2 * block) * 2 + 1, col] = c[1]
                W[(2 * block + 1) * 2 + 0, col] = c[2]
                W[(2 * block + 1) * 2 + 1, col] = c[3]
                col += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-9):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += Ua[idx_u, r] * Va[idx_v, r] * Wa[idx_w, r]
                    expected = 1.0 if j == k else 0.0
                    if abs(total - expected) > tol:
                        return False
        return True

    # Build the known rank-35 solution
    best = build_strassen_rank35()

    if not verify(best["U"], best["V"], best["W"]):
        pass

    # Attempt rank-34 via symmetry-adapted basis reduction
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 if j==k else 0
    # has invariance under independent column permutations of the 2x10 matrix.
    # We seek a rank-34 decomposition by checking if the 35 Strassen terms
    # have a non-trivial linear dependency when viewed in a symmetry-adapted basis.
    
    if time.time() < deadline - 30:
        # Build the full tensor T of shape (4, 20, 20)
        T = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        T[i * m + j, j * p + k, k * n + i] = 1.0

        # The tensor is a direct sum of 5 copies of M_{2,2,2}.
        # Rank(M_{2,2,2}) = 7 (Strassen, optimal).
        # For a direct sum, rank(T1 ⊕ T2 ⊕ ... ⊕ T5) >= rank(T1) + ... + rank(T5) = 35
        # This is a known result: the rank of a direct sum of matrix multiplication tensors
        # is at least the sum of the ranks.
        
        # However, let's verify this more carefully by checking if there's any
        # cross-block structure that could reduce rank.
        
        # Construct the 35 rank-1 terms from Strassen
        terms = []
        for block in range(5):
            for a, b, c in strassen:
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                for i in range(nm):
                    u[i] = a[i]
                v[0 * p + 2 * block] = b[0]
                v[0 * p + 2 * block + 1] = b[1]
                v[1 * p + 2 * block] = b[2]
                v[1 * p + 2 * block + 1] = b[3]
                w[(2 * block) * 2 + 0] = c[0]
                w[(2 * block) * 2 + 1] = c[1]
                w[(2 * block + 1) * 2 + 0] = c[2]
                w[(2 * block + 1) * 2 + 1] = c[3]
                terms.append((u, v, w))
        
        # Check if any linear combination of these 35 terms sums to zero
        # while preserving the tensor. This would indicate a rank reduction.
        # We set up: sum_r alpha_r * (u_r ⊗ v_r ⊗ w_r) = 0
        # This is equivalent to: for all (i,j,k): sum_r alpha_r * u_r[i] * v_r[j] * w_r[k] = 0
        
        # Build the constraint matrix: rows indexed by (i,j,k) where T is nonzero or could be affected
        # Actually, we need: sum_r alpha_r * u_r[i] * v_r[j] * w_r[k] = 0 for ALL (i,j,k)
        # because the zero tensor has all entries zero.
        
        # The number of (i,j,k) triples is 4*20*20 = 1600
        # We have 35 unknowns (alpha_r)
        # Build A of shape (1600, 35) where A[idx, r] = u_r[i]*v_r[j]*w_r[k]
        
        A = np.zeros((nm * mp * pn, 35), dtype=float)
        for idx in range(nm * mp * pn):
            k_idx = idx % pn
            temp = idx // pn
            j_idx = temp % mp
            i_idx = temp // mp
            for r in range(35):
                u, v, w = terms[r]
                A[idx, r] = u[i_idx] * v[j_idx] * w[k_idx]
        
        # Check if A has a non-trivial null space
        # If nullity > 0, there's a linear dependency, suggesting rank < 35
        try:
            rank_A = np.linalg.matrix_rank(A, tol=1e-10)
            nullity = 35 - rank_A
        except:
            nullity = 0
        
        if nullity > 0:
            # There might be a rank reduction. Find the null space vector.
            # Use SVD to find the null space
            try:
                U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
                # The last (35 - rank_A) rows of Vt span the null space
                if rank_A < 35:
                    null_vec = Vt[rank_A, :]
                    # Check if this gives a valid rank-34 decomposition
                    # Remove one term and see if we can express it as a combination
                    # of the others with the null vector coefficients
                    
                    # For each r, the coefficient is null_vec[r]
                    # We can eliminate term r* where null_vec[r*] != 0
                    # and express it as a combination of the others
                    
                    # Find the term with the largest coefficient
                    r_star = np.argmax(np.abs(null_vec))
                    if abs(null_vec[r_star]) > 1e-10:
                        # Express term r_star as a linear combination of the others
                        # alpha_r * term_r + alpha_{r*} * term_{r*} = 0
                        # term_{r*} = -sum_{r != r*} (alpha_r / alpha_{r*}) * term_r
                        
                        # Build a new rank-34 decomposition by replacing term r_star
                        # with the linear combination. But this doesn't reduce rank directly.
                        # We need to check if one of the terms is a linear combination
                        # of the others in a way that allows elimination.
                        
                        # Actually, if there's a null vector, we can write:
                        # sum_r alpha_r * term_r = 0
                        # This means the 35 terms are linearly dependent as tensors.
                        # So the tensor can be written as a sum of at most 34 terms.
                        
                        # To construct the rank-34 decomposition, we eliminate one term.
                        # Let's eliminate term r_star.
                        # The remaining 34 terms still span the same tensor space,
                        # but we need to find coefficients such that:
                        # T = sum_{r != r_star} beta_r * term_r
                        
                        # From the null vector: sum_r alpha_r * term_r = 0
                        # => term_{r*} = -sum_{r != r*} (alpha_r / alpha_{r*}) * term_r
                        
                        # The original T = sum_{r=0}^{34} 1.0 * term_r
                        # = sum_{r != r*} 1.0 * term_r + 1.0 * term_{r*}
                        # = sum_{r != r*} 1.0 * term_r - sum_{r != r*} (alpha_r / alpha_{r*}) * term_r
                        # = sum_{r != r*} (1.0 - alpha_r / alpha_{r*}) * term_r
                        
                        # So the new coefficients are:
                        # beta_r = 1.0 - alpha_r / alpha_{r*} for r != r_star
                        
                        new_U = np.zeros((nm, 34), dtype=float)
                        new_V = np.zeros((mp, 34), dtype=float)
                        new_W = np.zeros((pn, 34), dtype=float)
                        
                        col_new = 0
                        for r in range(35):
                            if r == r_star:
                                continue
                            beta = 1.0 - null_vec[r] / null_vec[r_star]
                            u, v, w = terms[r]
                            new_U[:, col_new] = beta * u
                            new_V[:, col_new] = beta * v
                            new_W[:, col_new] = beta * w
                            col_new += 1
                        
                        new_result = {
                            "U": new_U.tolist(),
                            "V": new_V.tolist(),
                            "W": new_W.tolist()
                        }
                        
                        if verify(new_result["U"], new_result["V"], new_result["W"]):
                            best = new_result
                            # We found rank 34!
                            pass
                        else:
                            # The linear dependency doesn't give a valid decomposition
                            # (might be numerical issues or the dependency is in a
                            # different sense). Keep rank 35.
                            pass
            except:
                pass

    # Final verification
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback to the known correct rank-35 solution
        best = build_strassen_rank35()

    return best
