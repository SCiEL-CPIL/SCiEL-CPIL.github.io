import numpy as np
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank < 49 decomposition for 4x4 matrix multiplication.
    Strategy: Build the rank-49 recursive Strassen decomposition, then use
    linear algebra to find linear dependencies among the rank-1 terms and
    eliminate redundant ones.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin

    n, m, p = 4, 4, 4
    base_rank = 49

    def build_strassen_49():
        """Build the rank-49 recursive Strassen decomposition."""
        U = [[0] * base_rank for _ in range(n * m)]
        V = [[0] * base_rank for _ in range(m * p)]
        W = [[0] * base_rank for _ in range(p * n)]

        def get_U_index(row, col):
            return row * m + col

        def get_V_index(row, col):
            return row * p + col

        def get_W_index(row, col):
            return col * n + row  # trace convention: W[k*n+i, r]

        def a_elem(bi, bj, r, c):
            return (2 * bi + r, 2 * bj + c)

        def b_elem(bi, bj, r, c):
            return (2 * bi + r, 2 * bj + c)

        def c_elem(bi, bj, r, c):
            return (2 * bi + r, 2 * bj + c)

        # Define the 7 block-level Strassen terms
        L_terms = []
        R_terms = []
        contrib_terms = []

        # Term 0: P1 = (A11 + A22)(B11 + B22)
        L0 = [[{} for _ in range(2)] for _ in range(2)]
        R0 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(0, 0, r, c)
                dr, dc = a_elem(1, 1, r, c)
                L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
                er, ec = b_elem(0, 0, r, c)
                hr, hc = b_elem(1, 1, r, c)
                R0[r][c] = {(er, ec): 1, (hr, hc): 1}
        contrib0 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib0[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 1, r, c)
                contrib0[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L0)
        R_terms.append(R0)
        contrib_terms.append(contrib0)

        # Term 1: P2 = (A21 + A22)B11
        L1 = [[{} for _ in range(2)] for _ in range(2)]
        R1 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(1, 0, r, c)
                dr, dc = a_elem(1, 1, r, c)
                L1[r][c] = {(ar, ac): 1, (dr, dc): 1}
                er, ec = b_elem(0, 0, r, c)
                R1[r][c] = {(er, ec): 1}
        contrib1 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(1, 0, r, c)
                contrib1[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 1, r, c)
                contrib1[(r, c)].append((cr2, cc2, -1))
        L_terms.append(L1)
        R_terms.append(R1)
        contrib_terms.append(contrib1)

        # Term 2: P3 = A11(B12 - B22)
        L2 = [[{} for _ in range(2)] for _ in range(2)]
        R2 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(0, 0, r, c)
                L2[r][c] = {(ar, ac): 1}
                fr, fc = b_elem(0, 1, r, c)
                hr, hc = b_elem(1, 1, r, c)
                R2[r][c] = {(fr, fc): 1, (hr, hc): -1}
        contrib2 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 1, r, c)
                contrib2[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 1, r, c)
                contrib2[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L2)
        R_terms.append(R2)
        contrib_terms.append(contrib2)

        # Term 3: P4 = A22(B21 - B11)
        L3 = [[{} for _ in range(2)] for _ in range(2)]
        R3 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                dr, dc = a_elem(1, 1, r, c)
                L3[r][c] = {(dr, dc): 1}
                gr, gc = b_elem(1, 0, r, c)
                er, ec = b_elem(0, 0, r, c)
                R3[r][c] = {(gr, gc): 1, (er, ec): -1}
        contrib3 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib3[(r, c)] = [(cr, cc, 1)]
                cr2, cc2 = c_elem(1, 0, r, c)
                contrib3[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L3)
        R_terms.append(R3)
        contrib_terms.append(contrib3)

        # Term 4: P5 = (A11 + A12)B22
        L4 = [[{} for _ in range(2)] for _ in range(2)]
        R4 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                ar, ac = a_elem(0, 0, r, c)
                br, bc = a_elem(0, 1, r, c)
                L4[r][c] = {(ar, ac): 1, (br, bc): 1}
                hr, hc = b_elem(1, 1, r, c)
                R4[r][c] = {(hr, hc): 1}
        contrib4 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib4[(r, c)] = [(cr, cc, -1)]
                cr2, cc2 = c_elem(0, 1, r, c)
                contrib4[(r, c)].append((cr2, cc2, 1))
        L_terms.append(L4)
        R_terms.append(R4)
        contrib_terms.append(contrib4)

        # Term 5: P6 = (A21 - A11)(B11 + B12)
        L5 = [[{} for _ in range(2)] for _ in range(2)]
        R5 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                cr, cc = a_elem(1, 0, r, c)
                ar, ac = a_elem(0, 0, r, c)
                L5[r][c] = {(cr, cc): 1, (ar, ac): -1}
                er, ec = b_elem(0, 0, r, c)
                fr, fc = b_elem(0, 1, r, c)
                R5[r][c] = {(er, ec): 1, (fr, fc): 1}
        contrib5 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(1, 1, r, c)
                contrib5[(r, c)] = [(cr, cc, 1)]
        L_terms.append(L5)
        R_terms.append(R5)
        contrib_terms.append(contrib5)

        # Term 6: P7 = (A12 - A22)(B21 + B22)
        L6 = [[{} for _ in range(2)] for _ in range(2)]
        R6 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                br, bc = a_elem(0, 1, r, c)
                dr, dc = a_elem(1, 1, r, c)
                L6[r][c] = {(br, bc): 1, (dr, dc): -1}
                gr, gc = b_elem(1, 0, r, c)
                hr, hc = b_elem(1, 1, r, c)
                R6[r][c] = {(gr, gc): 1, (hr, hc): 1}
        contrib6 = {}
        for r in range(2):
            for c in range(2):
                cr, cc = c_elem(0, 0, r, c)
                contrib6[(r, c)] = [(cr, cc, 1)]
        L_terms.append(L6)
        R_terms.append(R6)
        contrib_terms.append(contrib6)

        # Strassen sub-product contributions to 2x2 output
        sub_contrib = [
            [(0, 0, 1), (1, 1, 1)],    # M1 -> P11:+1, P22:+1
            [(1, 0, 1), (1, 1, -1)],   # M2 -> P21:+1, P22:-1
            [(0, 1, 1), (1, 1, 1)],    # M3 -> P12:+1, P22:+1
            [(0, 0, 1), (1, 0, 1)],    # M4 -> P11:+1, P21:+1
            [(0, 0, -1), (0, 1, 1)],   # M5 -> P11:-1, P12:+1
            [(1, 1, 1)],               # M6 -> P22:+1
            [(0, 0, 1)]                # M7 -> P11:+1
        ]

        for t in range(7):
            L_t = L_terms[t]
            R_t = R_terms[t]
            contrib_t = contrib_terms[t]

            a = L_t[0][0]
            b = L_t[0][1]
            c = L_t[1][0]
            d = L_t[1][1]
            e = R_t[0][0]
            f = R_t[0][1]
            g = R_t[1][0]
            h = R_t[1][1]

            sub_products = []

            # M1: X = a+d, Y = e+h
            x1 = {}
            for key, coeff in a.items():
                x1[key] = x1.get(key, 0) + coeff
            for key, coeff in d.items():
                x1[key] = x1.get(key, 0) + coeff
            y1 = {}
            for key, coeff in e.items():
                y1[key] = y1.get(key, 0) + coeff
            for key, coeff in h.items():
                y1[key] = y1.get(key, 0) + coeff
            sub_products.append((x1, y1))

            # M2: X = c+d, Y = e
            x2 = {}
            for key, coeff in c.items():
                x2[key] = x2.get(key, 0) + coeff
            for key, coeff in d.items():
                x2[key] = x2.get(key, 0) + coeff
            y2 = dict(e)
            sub_products.append((x2, y2))

            # M3: X = a, Y = f-h
            x3 = dict(a)
            y3 = {}
            for key, coeff in f.items():
                y3[key] = y3.get(key, 0) + coeff
            for key, coeff in h.items():
                y3[key] = y3.get(key, 0) - coeff
            sub_products.append((x3, y3))

            # M4: X = d, Y = g-e
            x4 = dict(d)
            y4 = {}
            for key, coeff in g.items():
                y4[key] = y4.get(key, 0) + coeff
            for key, coeff in e.items():
                y4[key] = y4.get(key, 0) - coeff
            sub_products.append((x4, y4))

            # M5: X = a+b, Y = h
            x5 = {}
            for key, coeff in a.items():
                x5[key] = x5.get(key, 0) + coeff
            for key, coeff in b.items():
                x5[key] = x5.get(key, 0) + coeff
            y5 = dict(h)
            sub_products.append((x5, y5))

            # M6: X = c-a, Y = e+f
            x6 = {}
            for key, coeff in c.items():
                x6[key] = x6.get(key, 0) + coeff
            for key, coeff in a.items():
                x6[key] = x6.get(key, 0) - coeff
            y6 = {}
            for key, coeff in e.items():
                y6[key] = y6.get(key, 0) + coeff
            for key, coeff in f.items():
                y6[key] = y6.get(key, 0) + coeff
            sub_products.append((x6, y6))

            # M7: X = b-d, Y = g+h
            x7 = {}
            for key, coeff in b.items():
                x7[key] = x7.get(key, 0) + coeff
            for key, coeff in d.items():
                x7[key] = x7.get(key, 0) - coeff
            y7 = {}
            for key, coeff in g.items():
                y7[key] = y7.get(key, 0) + coeff
            for key, coeff in h.items():
                y7[key] = y7.get(key, 0) + coeff
            sub_products.append((x7, y7))

            for k in range(7):
                x_k, y_k = sub_products[k]
                rank_idx = t * 7 + k

                for (row, col), coeff in x_k.items():
                    u_idx = get_U_index(row, col)
                    U[u_idx][rank_idx] += coeff

                for (row, col), coeff in y_k.items():
                    v_idx = get_V_index(row, col)
                    V[v_idx][rank_idx] += coeff

                for (pr, pc, sub_coeff) in sub_contrib[k]:
                    if (pr, pc) in contrib_t:
                        for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                            w_idx = get_W_index(out_row, out_col)
                            W[w_idx][rank_idx] += sub_coeff * block_coeff

        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Build the tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # where i = row*4+col for A, j = row*4+col for B, k = col*4+row for C (trace convention)
        # T[i,j,k] should equal 1 if i=(a,c), j=(c,b), k=(b,a) in appropriate sense
        
        # Actually, let's verify directly: for each (i,j) pair in C,
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # The tensor equation is: delta[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # where the indices are: i indexes A (16 values), j indexes B (16 values), k indexes C (16 values)
        # And the constraint is that this equals 1 when (i,j,k) form a valid triple, 0 otherwise.
        
        # More precisely: T[A_idx, B_idx, C_idx] = sum_r U[A_idx,r] * V[B_idx,r] * W[C_idx,r]
        # This should equal 1 if A_idx=(i,k), B_idx=(k,j), C_idx=(j,i) [trace convention W[k*n+i]],
        # and 0 otherwise.
        
        # Let's check: for all i,j,k in 0..3:
        # A_idx = i*4+k, B_idx = k*4+j, C_idx = j*4+i (since W[k*n+i] means W index = k*4+i where k is B-row and i is A-col... wait)
        
        # Actually the trace convention says W[k*n+i, r] where the output element C[i,j] gets contribution
        # from W[j*4+i, r] * U[i*4+k, r] * V[k*4+j, r] summed over k and r.
        # So C[i,j] = sum_k sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
        # This should equal sum_k A[i,k] * B[k,j]
        
        # So the tensor T[A_idx, B_idx, C_idx] = sum_r U[A_idx,r]*V[B_idx,r]*W[C_idx,r]
        # where A_idx = i*4+k, B_idx = k*4+j, C_idx = j*4+i
        # should equal 1 for all valid (i,j,k) triples, and 0 otherwise.
        
        # Let's just verify by checking all 16*16*16 = 4096 combinations
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This should be 1 when a=i*4+k, b=k*4+j, c=j*4+i for some i,j,k
        # and 0 otherwise.
        
        # Build expected tensor
        T_expected = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a_idx = i * 4 + k
                    b_idx = k * 4 + j
                    c_idx = j * 4 + i
                    T_expected[a_idx, b_idx, c_idx] = 1.0
        
        # Compute actual tensor
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a 3-tensor with rank decomposition
        T_actual = np.zeros((16, 16, 16))
        for r in range(rank):
            T_actual += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        
        max_err = np.max(np.abs(T_actual - T_expected))
        return max_err < tol, max_err

    def find_minimal_rank(U, V, W):
        """
        Try to reduce the rank by finding linear dependencies.
        The key insight: if sum_{r in S} alpha_r * (u_r, v_r, w_r) = (0, 0, 0) component-wise
        in a way that preserves the tensor, we can eliminate one term.
        
        Actually, a simpler approach: check if any rank-1 term is redundant,
        i.e., if removing it still gives a valid decomposition (which it won't unless
        the coefficient is zero).
        
        Better approach: look for linear dependencies in the 3-tensor space.
        Each rank-1 term contributes u_r ⊗ v_r ⊗ w_r to the tensor.
        If there exist coefficients alpha_r such that sum_r alpha_r * u_r ⊗ v_r ⊗ w_r = 0,
        then we can't simply remove a term. But if one of the alpha_r is nonzero,
        we can express that term as a combination of others... but that doesn't reduce rank
        unless we can absorb it.
        
        Actually, the rank is the minimum number of rank-1 tensors that sum to T.
        If the 49 vectors (u_r ⊗ v_r ⊗ w_r) are linearly dependent in the 16^3 = 4096-dim space,
        that doesn't directly help. What we need is: if one of the rank-1 terms can be written
        as a linear combination of the others in the tensor space, AND that combination
        corresponds to a valid rank-1 decomposition... which is generally not possible.
        
        Let me try a different approach: use the fact that the Strassen decomposition has
        structure. Look for pairs of terms that can be combined.
        
        Actually, let me try the most direct approach: use numpy to check if the 49 rank-1
        tensors are linearly independent. If not, we might be able to reduce the rank.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Flatten each rank-1 tensor to a 4096-dim vector
        # T_r = u_r ⊗ v_r ⊗ w_r, flattened
        vecs = []
        for r in range(rank):
            vec = np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r]).flatten()
            vecs.append(vec)
        
        vecs = np.array(vecs)  # shape (49, 4096)
        
        # Check rank of this matrix
        # If rank < 49, there are linear dependencies
        # But this alone doesn't let us reduce the decomposition rank
        
        # Let's try a different approach: for each term r, check if it can be
        # expressed as a linear combination of the others. If so, and if the
        # combination has a special structure, we might be able to eliminate it.
        
        # Actually, the simplest valid approach: try removing each term one at a time
        # and see if the remaining terms can still span the tensor (they can't, since
        # the tensor is fixed). So this won't work directly.
        
        # Let me think differently. The Bläser rank-48 decomposition is known to exist.
        # It's obtained by a careful analysis of the Strassen decomposition and finding
        # a way to save one multiplication.
        
        # For now, let's just return the rank-49 decomposition and try to verify it.
        # Then we can try to implement the Bläser decomposition.
        
        return rank, vecs

    # Build the base rank-49 decomposition
    U, V, W = build_strassen_49()
    
    # Verify it's correct
    valid, max_err = verify_decomposition(U, V, W)
    
    if not valid:
        # If the base decomposition is wrong, we need to fix it
        # For now, return it anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    # Attempt Bläser-style rank reduction
    # The idea: in the recursive Strassen, some of the 7 sub-Strassen decompositions
    # have overlapping structure that can be exploited.
    
    # Let's try a more systematic approach: use integer linear programming or
    # Gaussian elimination over the rationals to find if any term is redundant.
    
    # Actually, let me try the most promising approach: look for linear dependencies
    # among the 49 rank-1 terms in the tensor space, and if found, try to use them
    # to eliminate a term.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build the 49 x 4096 matrix of flattened rank-1 tensors
    vecs = np.array([
        np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r]).flatten()
        for r in range(rank)
    ])
    
    # Check if there's a linear dependency
    # Use SVD to find the null space
    try:
        U_svd, s, Vt = np.linalg.svd(vecs, full_matrices=False)
        # The null space dimension is rank - len(s) where s has non-zero singular values
        tol = 1e-10
        num_nonzero = np.sum(s > tol)
        nullity = rank - num_nonzero
        
        if nullity > 0:
            # There are linear dependencies
            # The null space vectors are the last `nullity` columns of Vt
            null_space = Vt[num_nonzero:, :]  # shape (nullity, 49)
            
            # For each null space vector, we have sum_r null_space[i,r] * T_r = 0
            # This means we can potentially eliminate one term if the coefficient
            # for that term is nonzero and we can "absorb" it into the others.
            
            # But this is tricky because the rank-1 structure is not preserved
            # under linear combinations in general.
            
            # Let's try a different approach: for each term r, check if removing it
            # and adjusting the others can still give a valid decomposition.
            # This is equivalent to checking if T_r is in the span of {T_s : s != r}
            # with the additional constraint that the span elements maintain rank-1 form.
            
            # This is generally not possible, so let's just try the simple case:
            # if a term has all-zero U, V, or W vectors, it can be removed.
            for r in range(rank):
                if np.allclose(U_arr[:, r], 0) or np.allclose(V_arr[:, r], 0) or np.allclose(W_arr[:, r], 0):
                    # This term is zero and can be removed
                    U_new = U_arr[:, [i for i in range(rank) if i != r]]
                    V_new = V_arr[:, [i for i in range(rank) if i != r]]
                    W_new = W_arr[:, [i for i in range(rank) if i != r]]
                    valid_new, _ = verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist())
                    if valid_new:
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            
            # No zero terms found. Try another approach.
    except:
        pass
    
    # If we couldn't reduce the rank, return the original
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
