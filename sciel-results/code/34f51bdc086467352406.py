"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a structural approach: build a candidate decomposition using
Strassen blocks and direct terms, then aggressively reduce rank via
efficient column elimination using matrix rank checks.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def build_strassen_direct():
        """Build initial decomposition using Strassen for 2x2 blocks + direct terms."""
        U_t = []
        V_t = []
        W_t = []

        def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
            a = (row_off_a, col_off_a)
            b = (row_off_a, col_off_a + 1)
            c = (row_off_a + 1, col_off_a)
            d = (row_off_a + 1, col_off_a + 1)
            e = (row_off_b, col_off_b)
            f = (row_off_b, col_off_b + 1)
            g = (row_off_b + 1, col_off_b)
            h = (row_off_b + 1, col_off_b + 1)

            C00 = (row_off_a, col_off_b)
            C01 = (row_off_a, col_off_b + 1)
            C10 = (row_off_a + 1, col_off_b)
            C11 = (row_off_a + 1, col_off_b + 1)

            U_t.append([(a, 1), (d, 1)])
            V_t.append([(e, 1), (h, 1)])
            W_t.append([(C00, 1), (C11, 1)])

            U_t.append([(c, 1), (d, 1)])
            V_t.append([(e, 1)])
            W_t.append([(C10, 1), (C11, -1)])

            U_t.append([(a, 1)])
            V_t.append([(f, 1), (h, -1)])
            W_t.append([(C01, 1), (C11, 1)])

            U_t.append([(d, 1)])
            V_t.append([(g, 1), (e, -1)])
            W_t.append([(C00, 1), (C10, 1)])

            U_t.append([(a, 1), (b, 1)])
            V_t.append([(h, 1)])
            W_t.append([(C00, -1), (C01, 1)])

            U_t.append([(c, 1), (a, -1)])
            V_t.append([(e, 1), (f, 1)])
            W_t.append([(C11, 1)])

            U_t.append([(b, 1), (d, -1)])
            V_t.append([(g, 1), (h, 1)])
            W_t.append([(C00, 1)])

        def direct_term(i, j, k):
            U_t.append([((i, j), 1)])
            V_t.append([((j, k), 1)])
            W_t.append([((i, k), 1)])

        # A[0:2, 0:2] * B[0:2, 0:8]: 4 Strassen blocks
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off)

        # A[0:2, 2] * B[2, 0:8]: direct
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)

        # A[2, 0:2] * B[0:2, 0:8]: direct
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)

        # A[2, 2] * B[2, 0:8]: direct
        for k in range(8):
            direct_term(2, 2, k)

        return U_t, V_t, W_t

    def build_matrices(U_t, V_t, W_t):
        rank = len(W_t)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff

        return U, V, W

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(float(val))
            result.append(nice_row)
        return result

    def fast_column_elimination(Ua, Va, Wa, max_iter=50):
        """
        Fast column elimination using matrix rank checks.
        A column r is redundant if the tensor column T_r = u_r ⊗ v_r ⊗ w_r
        is in the span of other tensor columns.
        
        Key insight: T_r is in span of {T_s} iff for all i,j,k:
        U[i,r]*V[j,r]*W[k,r] = sum_s c_s * U[i,s]*V[j,s]*W[k,s]
        
        We can check this by building the matrix M where M[i,j,k] = T_r[i,j,k]
        and checking if rank([others | T_r]) == rank([others]).
        
        More efficient: use the fact that we can check via multiple matrix products.
        """
        U = Ua.copy()
        V = Va.copy()
        W = Wa.copy()
        
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            rank = U.shape[1]
            if rank <= 1:
                break
            
            removed = False
            # Try removing columns from the end first (often more redundant)
            for r in range(rank - 1, -1, -1):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Check if column r is in span of others
                # Method: For each pair of indices (i,j), check if 
                # U[i,r]*V[j,r] * W[:,r] is in span of {U[i,s]*V[j,s]*W[:,s]}
                # Actually, simpler: build the full tensor and check rank
                
                # Efficient check: use the structure
                # T_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]
                # T_r is in span of others iff for the matrix 
                # A = U[:,r] * V[:,r]^T (outer product), we have
                # A ⊙ (something with W) in span...
                
                # Simplest reliable method: build tensor vectors but do it smart
                # For each (i,j) pair, we get a vector in R^p: v_{ij} = U[i,r]*V[j,r]*W[:,r]
                # And for others: v_{ij,s} = U[i,s]*V[j,s]*W[:,s]
                # T_r in span iff for ALL (i,j), v_{ij} is in span of {v_{ij,s}}
                
                # Even more efficient: check if the matrix 
                # M = U[:,r] ⊗ V[:,r] ⊗ W[:,r] (flattened) is in span
                # by checking rank of [others | T_r] vs rank of [others]
                
                # Let's use a different approach: 
                # Check if there exist coefficients c_s such that
                # U[i,r]*V[j,r]*W[k,r] = sum_s c_s*U[i,s]*V[j,s]*W[k,s] for all i,j,k
                # This is equivalent to: the matrix (U[:,r] * V[:,r]^T) ⊙ W[:,r] (in some sense)
                
                # Practical approach: use lstsq on the flattened tensor
                # But make it faster by only checking a subset of (i,j,k) tuples
                # Actually for our small sizes, let's just do it directly
                
                others_idx = [s for s in range(rank) if s != r]
                if not others_idx:
                    continue
                
                # Build the tensor for column r
                # Shape: (n*m, m*p, p*n) but we'll flatten
                # T_r = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                # Flatten: T_r_flat = T_r.flatten()
                
                # Build others matrix
                # This is expensive, so let's use a smarter check
                
                # Smart check: For the decomposition to hold, we need
                # For each i: U[i,r] * (V[:,r] ⊗ W[:,r]) is in span of {U[i,s] * (V[:,s] ⊗ W[:,s])}
                # And for each j: V[j,r] * (U[:,r] ⊗ W[:,r]) is in span of {V[j,s] * (U[:,s] ⊗ W[:,s])}
                # etc.
                
                # Let's just do the full check but with early termination
                # Build flattened tensor vectors
                dim = n * m * m * p * p * n
                T_r = np.zeros(dim)
                for i in range(n * m):
                    for j in range(m * p):
                        for k in range(p * n):
                            T_r[i * (m * p * p * n) + j * (p * n) + k] = U[i, r] * V[j, r] * W[k, r]
                
                # Build others matrix
                others_mat = np.zeros((dim, len(others_idx)))
                for idx, s in enumerate(others_idx):
                    for i in range(n * m):
                        for j in range(m * p):
                            for k in range(p * n):
                                others_mat[i * (m * p * p * n) + j * (p * n) + k, idx] = U[i, s] * V[j, s] * W[k, s]
                
                # Check if T_r is in column space of others_mat
                try:
                    # Use lstsq
                    coeffs, residuals, rk, sv = np.linalg.lstsq(others_mat, T_r, rcond=1e-10)
                    recon = others_mat @ coeffs
                    norm_T = np.linalg.norm(T_r)
                    if norm_T > 1e-12:
                        if np.linalg.norm(recon - T_r) < 1e-8 * norm_T:
                            # Remove column r
                            U = np.delete(U, r, axis=1)
                            V = np.delete(V, r, axis=1)
                            W = np.delete(W, r, axis=1)
                            removed = True
                            break
                except:
                    pass
            
            if not removed:
                break
        
        return U, V, W

    # Build initial decomposition
    U_t, V_t, W_t = build_strassen_direct()
    U, V, W = build_matrices(U_t, V_t, W_t)
    rank = len(W_t)

    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Run fast column elimination
    if time.time() - start_time < DEADLINE - 10:
        U, V, W = fast_column_elimination(U, V, W, max_iter=30)
        if verify_decomp(U, V, W):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]

    # Try algebraic simplification: look for duplicate or nearly duplicate columns
    if time.time() - start_time < DEADLINE - 5:
        U2, V2, W2 = best_U.copy(), best_V.copy(), best_W.copy()
        current_rank = U2.shape[1]
        
        # Check for exact duplicates or proportional columns
        for r1 in range(current_rank):
            if time.time() - start_time > DEADLINE - 5:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                u1, u2 = U2[:, r1], U2[:, r2]
                v1, v2 = V2[:, r1], V2[:, r2]
                w1, w2 = W2[:, r1], W2[:, r2]
                
                # Check if columns are proportional
                # Find ratio from U
                mask_u = abs(u2) > 1e-10
                if not np.any(mask_u):
                    continue
                idx_u = np.argmax(mask_u)
                ratio_u = u1[idx_u] / u2[idx_u]
                
                if not np.allclose(u1, ratio_u * u2, atol=1e-10, rtol=1e-10):
                    continue
                
                mask_v = abs(v2) > 1e-10
                if not np.any(mask_v):
                    continue
                idx_v = np.argmax(mask_v)
                ratio_v = v1[idx_v] / v2[idx_v]
                
                if not np.allclose(v1, ratio_v * v2, atol=1e-10, rtol=1e-10):
                    continue
                
                mask_w = abs(w2) > 1e-10
                if not np.any(mask_w):
                    continue
                idx_w = np.argmax(mask_w)
                ratio_w = w1[idx_w] / w2[idx_w]
                
                if not np.allclose(w1, ratio_w * w2, atol=1e-10, rtol=1e-10):
                    continue
                
                # All three are proportional - check if ratios are consistent
                # If ratio_u ≈ ratio_v ≈ ratio_w, then the terms are identical
                # and we can remove one
                if abs(ratio_u - ratio_v) < 1e-8 and abs(ratio_v - ratio_w) < 1e-8:
                    # Remove column r2
                    U2 = np.delete(U2, r2, axis=1)
                    V2 = np.delete(V2, r2, axis=1)
                    W2 = np.delete(W2, r2, axis=1)
                    current_rank -= 1
                    # Adjust r2 index for subsequent checks
                    break
        
        if verify_decomp(U2, V2, W2) and U2.shape[1] < best_rank:
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
            best_rank = U2.shape[1]

    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Rebuild from scratch
        U_t2, V_t2, W_t2 = build_strassen_direct()
        U2, V2, W2 = build_matrices(U_t2, V_t2, W_t2)
        return {"U": to_nice_list(U2), "V": to_nice_list(V2), "W": to_nice_list(W2)}

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
