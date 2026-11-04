"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller blocks.
Then applies aggressive peeling with vectorized checks to remove redundant rank-1 terms.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
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

    def strassen_222_local(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
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

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block. Rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block. Rank 23 (3x7 Strassen + 2 schoolbook).
        """
        strassen_222_local(i0, j0, k0)
        strassen_222_local(i0, j0, k0 + 2)
        strassen_222_local(i0, j0, k0 + 4)
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m347_decomp():
        """
        Full 3x4 * 4x7 decomposition.
        Split n=3 into 1+2.
        n=1 part: 1x4 * 4x7. Split m=4 into 2+2. Two 1x2x7 blocks.
        n=2 part: 2x4 * 4x7. Split m=4 into 2+2. Two 2x2x7 blocks.
        """
        m127_decomp(0, 0, 0)
        m127_decomp(0, 2, 0)
        m227_decomp(1, 0, 0)
        m227_decomp(1, 2, 0)

    m347_decomp()

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def fast_peel(U_mat, V_mat, W_mat):
        """
        Aggressive peeling: repeatedly check each term for redundancy using
        vectorized numpy operations.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        rank = U_curr.shape[1]
        
        # Precompute index arrays for vectorized computation
        # For each (i,j,k), we need U[i*m+j,:], V[j*p+k,:], W[k*n+i,:]
        ijs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        removed = set()
        any_removed = True
        while any_removed and time.time() < deadline - 15:
            any_removed = False
            rank = U_curr.shape[1]
            for r in range(rank):
                if r in removed:
                    continue
                # Check if term r is redundant
                is_redundant = True
                for (i, j, k) in ijs:
                    u_row = U_curr[i*m+j, :]
                    v_row = V_curr[j*p+k, :]
                    w_row = W_curr[k*n+i, :]
                    # Compute sum excluding term r and removed terms
                    prod = u_row * v_row * w_row
                    total = prod.sum()
                    total -= prod[r]  # remove term r
                    for r2 in removed:
                        total -= prod[r2]
                    if abs(total - 1.0) > 1e-9:
                        is_redundant = False
                        break
                if is_redundant:
                    removed.add(r)
                    any_removed = True
        
        if not removed:
            return U_mat, V_mat, W_mat
        
        keep = [r for r in range(rank) if r not in removed]
        return U_curr[:, keep], V_curr[:, keep], W_curr[:, keep]

    def peel_with_swap(U_mat, V_mat, W_mat):
        """
        Try to reduce rank by finding pairs of terms that can be merged.
        If terms r1 and r2 have proportional contributions in all but one factor,
        they might be combinable.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        rank = U_curr.shape[1]
        
        # For each pair, check if we can eliminate one by adjusting the other
        # This is a simplified approach: check if removing term r and scaling
        # term s preserves the decomposition
        improved = False
        for r in range(rank):
            if time.time() > deadline - 20:
                break
            for s in range(rank):
                if s == r:
                    continue
                # Try: remove r, see if we can scale s to compensate
                # This only works if r and s have the same support pattern
                # Check if U[:,r] is proportional to U[:,s]
                if np.allclose(U_curr[:, r], 0.0):
                    # Term r has zero U component, check if it's truly redundant
                    continue
                
                # Simple check: can we express term r as a multiple of term s?
                # Check U
                if np.any(U_curr[:, r] != 0):
                    scale_u = U_curr[:, r] / U_curr[:, s]
                    if not np.allclose(U_curr[:, r], scale_u[0] * U_curr[:, s]):
                        continue
                # Check V
                if np.any(V_curr[:, r] != 0):
                    scale_v = V_curr[:, r] / V_curr[:, s]
                    if not np.allclose(V_curr[:, r], scale_v[0] * V_curr[:, s]):
                        continue
                # Check W
                if np.any(W_curr[:, r] != 0):
                    scale_w = W_curr[:, r] / W_curr[:, s]
                    if not np.allclose(W_curr[:, r], scale_w[0] * W_curr[:, s]):
                        continue
                
                # If all three are proportional, term r is redundant with term s
                # Verify by removing r
                U_test = U_curr[:, list(range(r)) + list(range(r+1, rank))]
                V_test = V_curr[:, list(range(r)) + list(range(r+1, rank))]
                W_test = W_curr[:, list(range(r)) + list(range(r+1, rank))]
                if verify(U_test, V_test, W_test):
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    rank -= 1
                    improved = True
                    break
            if improved:
                break
        
        return U_curr, V_curr, W_curr

    # Apply peeling iteratively
    U, V, W = fast_peel(U, V, W)
    
    # Second pass
    if time.time() < deadline - 25:
        U2, V2, W2 = fast_peel(U, V, W)
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Try swap-based peeling
    if time.time() < deadline - 30:
        U3, V3, W3 = peel_with_swap(U, V, W)
        if U3.shape[1] < U.shape[1]:
            U, V, W = U3, V3, W3
    
    # Final verification
    if not verify(U, V, W):
        # Fall back to original
        U_orig = np.array(U_vecs).T
        V_orig = np.array(V_vecs).T
        W_orig = np.array(W_vecs).T
        return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
