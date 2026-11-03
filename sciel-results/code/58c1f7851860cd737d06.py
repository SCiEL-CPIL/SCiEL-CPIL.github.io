import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140
    
    # Fallback: Current Rank 189 (Block-Strassen with Schoolbook 3x3)
    def construct_rank_189():
        bs = 3
        rank = 7 * 27
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        strassen_defs = [
            {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
            {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
            {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
            {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
            {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
            {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
            {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
        ]
        recon_coeffs = [
            [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
            [(2, 1.0), (4, 1.0)],
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
        ]
        
        r_idx = 0
        for s in range(7):
            s_def = strassen_defs[s]
            for u in range(bs):
                for v in range(bs):
                    for j in range(bs):
                        for block_idx, coeff_a in s_def['A']:
                            br = block_idx // 2
                            bc = block_idx % 2
                            i_glob = br * bs + u
                            k_glob = bc * bs + v
                            u_row = i_glob * m + k_glob
                            U[u_row][r_idx] += coeff_a
                        for block_idx, coeff_b in s_def['B']:
                            br = block_idx // 2
                            bc = block_idx % 2
                            k_glob = br * bs + v
                            j_glob = bc * bs + j
                            v_row = k_glob * p + j_glob
                            V[v_row][r_idx] += coeff_b
                        for c_block_idx in range(4):
                            coeffs = recon_coeffs[c_block_idx]
                            gamma = 0.0
                            for (s_idx, g) in coeffs:
                                if s_idx == s:
                                    gamma = g
                                    break
                            if gamma != 0.0:
                                br = c_block_idx // 2
                                bc = c_block_idx % 2
                                i_glob = br * bs + u
                                j_glob = bc * bs + j
                                w_row = j_glob * n + i_glob
                                W[w_row][r_idx] += gamma
                        r_idx += 1
        return {"U": U, "V": V, "W": W}

    # Attempt Rank 161 (Strassen 2x2 outer, Rank 23 BCLT 3x3 inner)
    def construct_rank_161():
        bs = 3
        rank3 = 23
        rank = 7 * rank3
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        strassen_defs = [
            {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
            {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
            {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
            {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
            {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
            {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
            {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
        ]
        recon_coeffs = [
            [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
            [(2, 1.0), (4, 1.0)],
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
        ]
        
        # BCLT rank-23 decomposition for 3x3 matrix multiplication
        # Each term r has U3[r][i*3+k], V3[r][k*3+j], W3[r][j*3+i]
        # We define 23 terms where each is a sparse representation
        
        # BCLT algorithm terms (u, v, w) for each of 23 multiplications
        # u: which A element (i*3+k), v: which B element (k*3+j), w: which C element (j*3+i)
        bclt_terms = [
            # Term 0: A[0,0]*B[0,0] -> C[0,0]
            (0, 0, 0),
            # Term 1: A[0,1]*B[1,0] -> C[0,0]
            (1, 3, 0),
            # Term 2: A[0,2]*B[2,0] -> C[0,0]
            (2, 6, 0),
            # Term 3: A[1,0]*B[0,1] -> C[0,1]
            (3, 1, 3),
            # Term 4: A[1,1]*B[1,1] -> C[0,1]
            (4, 4, 3),
            # Term 5: A[1,2]*B[2,1] -> C[0,1]
            (5, 7, 3),
            # Term 6: A[2,0]*B[0,2] -> C[0,2]
            (6, 2, 6),
            # Term 7: A[2,1]*B[1,2] -> C[0,2]
            (7, 5, 6),
            # Term 8: A[2,2]*B[2,2] -> C[0,2]
            (8, 8, 6),
            # Term 9: A[0,0]*B[0,1] -> C[1,0]
            (0, 1, 9),
            # Term 10: A[0,1]*B[1,1] -> C[1,0]
            (1, 4, 9),
            # Term 11: A[0,2]*B[2,1] -> C[1,0]
            (2, 7, 9),
            # Term 12: A[1,0]*B[0,2] -> C[1,1]
            (3, 2, 12),
            # Term 13: A[1,1]*B[1,2] -> C[1,1]
            (4, 5, 12),
            # Term 14: A[1,2]*B[2,2] -> C[1,1]
            (5, 8, 12),
            # Term 15: A[2,0]*B[0,0] -> C[2,0]
            (6, 0, 15),
            # Term 16: A[2,1]*B[1,0] -> C[2,0]
            (7, 3, 15),
            # Term 17: A[2,2]*B[2,0] -> C[2,0]
            (8, 6, 15),
            # Term 18: A[0,0]*B[1,0] -> C[0,0] (cross term)
            (0, 3, 0),
            # Term 19: A[1,0]*B[0,0] -> C[0,0] (cross term)
            (3, 0, 0),
            # Term 20: A[0,1]*B[0,0] -> C[1,0] (cross term)
            (1, 0, 9),
            # Term 21: A[1,1]*B[0,1] -> C[0,1] (cross term)
            (4, 1, 3),
            # Term 22: A[0,0]*B[0,2] -> C[2,0] (cross term)
            (0, 2, 15),
        ]
        
        r_idx = 0
        for s in range(7):
            s_def = strassen_defs[s]
            for u in range(bs):
                for v in range(bs):
                    for j in range(bs):
                        for term_idx in range(rank3):
                            a_elem, b_elem, c_elem = bclt_terms[term_idx]
                            a_i, a_k = a_elem // 3, a_elem % 3
                            b_k, b_j = b_elem // 3, b_elem % 3
                            c_j, c_i = c_elem // 3, c_elem % 3
                            
                            # Only contribute if this term uses the (u,v,j) indices
                            # In the inner 3x3 multiplication, we need to map
                            # the BCLT terms to the current (u,v,j) position
                            
                            # The BCLT decomposition is for the full 3x3 product
                            # For the Strassen outer loop, we're computing M_s[u,j]
                            # which is the (u,j) element of A_s * B_s
                            # A_s and B_s are 3x3 matrices formed from blocks
                            
                            # For each BCLT term that contributes to C[u,j] in the 3x3 product:
                            if a_i == u and b_j == j and a_k == v:  # A[u,v] * B[v,j] -> C[u,j]
                                # This BCLT term contributes to our (u,v,j) position
                                pass  # This approach is flawed; BCLT is a full 3x3 decomposition
        
        # The above approach is incorrect. Let me use a different strategy.
        # Instead of trying to map BCLT terms directly, I'll use the known fact that
        # rank-23 exists for 3x3 and construct it properly.
        
        # Actually, the safest approach is to just return rank 189 since I cannot
        # guarantee the BCLT coefficients are correct without a reference.
        return None

    # Try rank 161, fall back to rank 189
    result_161 = construct_rank_161()
    if result_161 is not None:
        # Verify rank 161
        U = np.array(result_161['U'])
        V = np.array(result_161['V'])
        W = np.array(result_161['W'])
        
        # Verify with random matrices
        for _ in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using decomposition
            # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            # But this is not how tensor decomposition works for matrix multiplication.
            # The correct formula is:
            # C[i,j] = sum_k A[i,k] * B[k,j]
            # And the decomposition says:
            # A[i,k] * B[k,j] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
            # Wait, that's not right either.
            
            # For matrix multiplication tensor:
            # T[i,k,j] = delta (the multiplication tensor)
            # T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
            
            # Let me verify properly
            C_computed = np.zeros((n, p))
            for r in range(len(U[0])):
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            C_computed[i, j] += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                break
        else:
            return result_161
    
    return construct_rank_189()
# EVOLVE-BLOCK-END
