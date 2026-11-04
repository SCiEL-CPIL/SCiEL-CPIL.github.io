import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # S1 = (A00+A11)(B00+B11)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # S2 = (A10+A11)B00
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        # S3 = A00(B10-B00)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # S4 = A11(B01-B10)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        # S5 = (A00+A01)B11
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        # S6 = (A10-A00)(B00+B01)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # S7 = (A01-A11)(B10+B11)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        for jj in range(2):
            for kk in range(2):
                U[(rA+0)*m + (cA+jj), col] = 1
                V[(rB+jj)*p + (cB+kk), col] = 1
                W[(cB+kk)*n + (rA+0), col] = 1
                col += 1

    # 4 Strassen blocks (2x2 each) for rows 0-1
    add_strassen(0, 0, 0, 0)  # C[0:2, 0:2]
    add_strassen(0, 2, 0, 2)  # C[0:2, 2:4]
    add_strassen(0, 0, 2, 0)  # C[0:2, 0:2] from B rows 2-3
    add_strassen(0, 2, 2, 2)  # C[0:2, 2:4] from B rows 2-3
    # 4 schoolbook 1x2x2 blocks for row 2
    add_schoolbook_1x2x2(2, 0, 0, 0)  # C[2, 0:2] from B rows 0-1, cols 0-1
    add_schoolbook_1x2x2(2, 2, 0, 2)  # C[2, 2:4] from B rows 0-1, cols 2-3
    add_schoolbook_1x2x2(2, 0, 2, 0)  # C[2, 0:2] from B rows 2-3, cols 0-1
    add_schoolbook_1x2x2(2, 2, 2, 2)  # C[2, 2:4] from B rows 2-3, cols 2-3
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, time_budget=90):
    """Try to reduce rank by finding linear dependencies among terms."""
    start = time.time()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute the outer product tensor for each term
    def compute_outer(Ur, Vr, Wr):
        """Compute the 3D outer product u⊗v⊗w flattened."""
        u = Ur.reshape(n, m)
        v = Vr.reshape(m, p)
        w = Wr.reshape(p, n)
        return np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    
    # Build the tensor equation: sum_r outer_r = T (all ones)
    # If we can express one term as a linear combination of others, we can eliminate it.
    
    # Strategy: For each pair of terms (r1, r2), try to eliminate r2 by expressing
    # outer_r2 = alpha * outer_r1 + ... (using other terms)
    # Simpler: try to remove each term and see if the residual can be absorbed.
    
    for r_remove in range(rank):
        if time.time() - start > time_budget:
            break
        if rank - 1 < best_rank:
            break
            
        # Remove term r_remove
        remaining = [r for r in range(rank) if r != r_remove]
        n_rem = len(remaining)
        
        # The deficit is the contribution of the removed term
        deficit = compute_outer(U[:, r_remove], V[:, r_remove], W[:, r_remove])
        
        # Build the matrix of remaining terms' outer products
        A_mat = np.zeros((n * m * p, n_rem))
        for new_r, orig_r in enumerate(remaining):
            A_mat[:, new_r] = compute_outer(U[:, orig_r], V[:, orig_r], W[:, orig_r])
        
        # Solve A_mat @ delta = deficit
        try:
            delta, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, deficit, rcond=None)
            # Check if the solution is exact
            if np.allclose(A_mat @ delta, deficit, atol=1e-8):
                # Check if delta has nice values (integers or half-integers)
                delta_2x = delta * 2
                if np.allclose(delta_2x, np.round(delta_2x), atol=1e-6):
                    # Update the remaining terms
                    U_new = np.delete(U, r_remove, axis=1)
                    V_new = np.delete(V, r_remove, axis=1)
                    W_new = np.delete(W, r_remove, axis=1)
                    
                    for new_r in range(n_rem):
                        U_new[:, new_r] += delta[new_r] * U[:, remaining[new_r]]
                        V_new[:, new_r] += delta[new_r] * V[:, remaining[new_r]]
                        W_new[:, new_r] += delta[new_r] * W[:, remaining[new_r]]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                        best_rank = n_rem
                        best_U, best_V, best_W = U_new, V_new, W_new
                        U, V, W = U_new, V_new, W_new
                        rank = n_rem
                        # Restart the loop since indices changed
                        r_remove = 0
                        continue
        except:
            pass
    
    return best_U, best_V, best_W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Start with the structured rank-44 decomposition
    U, V, W = _construct_structured_decomposition()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Fallback: return schoolbook rank 48
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, time_budget=80)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
