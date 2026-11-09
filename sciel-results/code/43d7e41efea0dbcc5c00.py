"""Bootstrap MODEL-AND-OPTIMIZE rewrite: parameterize the tensor decomposition
and verify it exactly, then attempt a small search to reduce rank.

Dimensions: A is (n x m) = (2 x 3), B is (m x p) = (3 x 4), C = A B is (n x p) = (2 x 4).
Convention: U index = i*m+j (A entry), V index = j*p+k (B entry), W index = k*n+i (C entry).
Decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r]  (outer product along r).
Score = rank (number of terms). Lower is better.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 140.0  # evaluator kills at 150s; keep 10s margin


def _tensor_index(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)


def _matricization_rank_lower_bound(T):
    """Compute lower bound on tensor rank from matricization ranks."""
    n, m, p = T.shape
    # Matricize along each mode
    # Mode-1 matricization: (n*m) x (m*p * p*n)
    M1 = T.reshape(n*m, -1)
    r1 = np.linalg.matrix_rank(M1, tol=1e-10)
    # Mode-2 matricization: (m*p) x (n*m * p*n)
    M2 = T.transpose(1, 0, 2).reshape(m*p, -1)
    r2 = np.linalg.matrix_rank(M2, tol=1e-10)
    # Mode-3 matricization: (p*n) x (n*m * m*p)
    M3 = T.transpose(2, 0, 1).reshape(p*n, -1)
    r3 = np.linalg.matrix_rank(M3, tol=1e-10)
    return max(r1, r2, r3)


def _als_optimize(T, r, n_init=5, max_iter=50, tol=1e-10):
    """Alternating Least Squares to find a rank-r decomposition of T.
    
    T has shape (n*m, m*p, p*n) = (6, 12, 8).
    We seek U (6xr), V (12xr), W (8xr) such that
    sum_s U[:,s] * V[:,s] * W[:,s] == T.
    
    For fixed V and W, U is determined by:
    For each a: U[a,:] satisfies sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c] for all (b,c) with T[a,b,c]!=0.
    This is linear in U[a,:].
    """
    dim_u, dim_v, dim_w = T.shape
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    for init in range(n_init):
        if time.time() - START_TIME > TIME_BUDGET - 10:
            break
        
        # Random initialization
        np.random.seed(init * 12345 + 7)
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        W = np.random.randn(dim_w, r)
        
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_BUDGET - 10:
                break
            
            # Update U: for each a, solve linear system
            for a in range(dim_u):
                # Find all (b,c) where T[a,b,c] != 0
                mask = T[a, :, :] != 0
                bc_indices = np.where(mask)
                if len(bc_indices[0]) == 0:
                    continue
                # For each (b,c) pair: sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
                # Build matrix A of shape (num_eq, r): A[j,s] = V[b_j,s]*W[c_j,s]
                b_idx = bc_indices[0]
                c_idx = bc_indices[1]
                A = np.outer(V[b_idx, :], W[c_idx, :]).reshape(-1, r)
                # Wait, that's not right. Let me think again.
                # For each equation j with (b_j, c_j): sum_s U[a,s]*V[b_j,s]*W[c_j,s] = T[a,b_j,c_j]
                # So A[j,s] = V[b_j,s]*W[c_j,s], and we solve A * U[a,:] = T[a,b_idx,c_idx]
                A = np.zeros((len(b_idx), r))
                for j in range(len(b_idx)):
                    A[j, :] = V[b_idx[j], :] * W[c_idx[j], :]
                b_vec = T[a, b_idx, c_idx]
                sol, residuals, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                U[a, :] = sol
            
            # Update V: for each b, solve linear system
            for b in range(dim_v):
                mask = T[:, b, :] != 0
                ac_indices = np.where(mask)
                if len(ac_indices[0]) == 0:
                    continue
                a_idx = ac_indices[0]
                c_idx = ac_indices[1]
                A = np.zeros((len(a_idx), r))
                for j in range(len(a_idx)):
                    A[j, :] = U[a_idx[j], :] * W[c_idx[j], :]
                b_vec = T[a_idx, b, c_idx]
                sol, residuals, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                V[b, :] = sol
            
            # Update W: for each c, solve linear system
            for c in range(dim_w):
                mask = T[:, :, c] != 0
                ab_indices = np.where(mask)
                if len(ab_indices[0]) == 0:
                    continue
                a_idx = ab_indices[0]
                b_idx = ab_indices[1]
                A = np.zeros((len(a_idx), r))
                for j in range(len(a_idx)):
                    A[j, :] = U[a_idx[j], :] * V[b_idx[j], :]
                b_vec = T[a_idx, b_idx, c]
                sol, residuals, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(recon - T) / np.linalg.norm(T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
        
        if best_err < tol:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def _try_reduce(U, V, W, T, max_attempts=200):
    """Attempt to find a lower-rank decomposition using ALS optimization
    and term removal strategies."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # First, try to find the matricization lower bound
    lb = _matricization_rank_lower_bound(T)
    
    # Try ALS optimization for ranks from lb to current rank - 1
    for r_try in range(max(lb, 1), rank):
        if time.time() - START_TIME > TIME_BUDGET - 15:
            break
        
        # Use the existing decomposition as initialization
        U_new, V_new, W_new, err = _als_optimize(T, r_try, n_init=10, max_iter=100,
                                                  init_from=(U, V, W))
        if U_new is not None and err < 1e-8:
            # Verify exactly
            if _verify(U_new, V_new, W_new, T):
                return U_new, V_new, W_new
    
    # Also try the original term-removal approach as fallback
    n, m, p = T.shape
    # Try removing each term and solving for W adjustments
    for r_remove in range(rank):
        if time.time() - START_TIME > TIME_BUDGET - 15:
            break
        U_rem = np.delete(U, r_remove, axis=1)
        V_rem = np.delete(V, r_remove, axis=1)
        W_rem = np.delete(W, r_remove, axis=1).copy()
        r_new = rank - 1
        
        valid = True
        for c in range(p * n):
            i_c = c % n
            k_c = c // n
            if k_c >= p:
                continue
            A_sys = np.zeros((m, r_new))
            b_sys = np.ones(m)
            for j in range(m):
                a_idx = i_c * m + j
                b_idx = j * p + k_c
                A_sys[j, :] = U_rem[a_idx, :] * V_rem[b_idx, :]
            
            sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            if not np.allclose(A_sys @ sol, b_sys, atol=1e-8):
                valid = False
                break
            W_rem[c, :] = sol
        
        if valid:
            if _verify(U_rem, V_rem, W_rem, T):
                U = U_rem
                V = V_rem
                W = W_rem
                rank = r_new
                break
    
    return U, V, W


def _build_rank22_decomposition(n, m, p):
    """Build the rank-22 Strassen-based decomposition for 2x3 x 3x4.

    Structure:
      C = A_L * B_T1 + A_L * B_T2 + A_R * B_B
    where:
      A_L = A[:, 0:2] (2x2), A_R = A[:, 2:3] (2x1)
      B_T1 = B[0:2, 0:2] (2x2), B_T2 = B[0:2, 2:4] (2x2), B_B = B[2:3, :] (1x4)

    Strassen (rank 7) for each 2x2 block, plus rank-8 schoolbook for 2x1 x 1x4.
    Total: 7 + 7 + 8 = 22.
    """
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        """Add a Strassen rank-7 decomposition for a 2x2x2 block.

        a_idx: dict with keys '00','01','10','11' -> U row indices
        b_idx: dict with keys '00','01','10','11' -> V row indices
        c_idx: dict with keys '00','01','10','11' -> W row indices
        col_start: starting column in U,V,W
        """
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        col = col_start
        # M1
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    # Block 1: A_L (2x2) * B_T1 (2x2) -> C[:, 0:2]
    # A indices: A[i,j] -> i*m+j. m=3.
    # A_L[0,0]=A[0,0]->idx 0, A_L[0,1]=A[0,1]->idx 1
    # A_L[1,0]=A[1,0]->idx 3, A_L[1,1]=A[1,1]->idx 4
    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    # B_T1[j,k] -> j*p+k. p=4.
    # B_T1[0,0]=B[0,0]->0, B_T1[0,1]=B[0,1]->1
    # B_T1[1,0]=B[1,0]->4, B_T1[1,1]=B[1,1]->5
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    # C[i,k] -> k*n+i. n=2.
    # C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    # Block 2: A_L (2x2) * B_T2 (2x2) -> C[:, 2:4]
    # B_T2[j,k] where k in {2,3}: B_T2[0,2]=B[0,2]->2, B_T2[0,3]=B[0,3]->3
    # B_T2[1,2]=B[1,2]->6, B_T2[1,3]=B[1,3]->7
    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    # C[0,2]->4, C[0,3]->6, C[1,2]->5, C[1,3]->7
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    # Block 3: A_R (2x1) * B_B (1x4) -> C (2x4), rank-8 schoolbook
    # A_R[0]=A[0,2]->idx 2, A_R[1]=A[1,2]->idx 5
    # B_B[k]=B[2,k]->idx 8+k for k=0..3
    # C[i,k] -> k*2+i
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0  # A[i,2]
            V[2 * p + k][col] = 1.0  # B[2,k]
            W[k * n + i][col] = 1.0  # C[i,k]
            col += 1

    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)

    # Start with rank-22 Strassen decomposition
    U_list, V_list, W_list = _build_rank22_decomposition(n, m, p)
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)

    if not _verify(U, V, W, T):
        # Fallback: schoolbook rank n*m*p = 2*3*4 = 24
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Save the verified rank-22 as fallback
    U_fallback = U.copy()
    V_fallback = V.copy()
    W_fallback = W.copy()

    # Attempt rank reduction
    if time.time() - START_TIME < TIME_BUDGET - 20:
        U, V, W = _try_reduce(U, V, W, T)
        
        # If we got a lower rank, try to go even lower
        current_rank = U.shape[1]
        if current_rank < 22 and time.time() - START_TIME < TIME_BUDGET - 15:
            for r_try in range(max(1, current_rank - 1), 0, -1):
                if time.time() - START_TIME > TIME_BUDGET - 15:
                    break
                U_new, V_new, W_new, err = _als_optimize(T, r_try, n_init=10, max_iter=100,
                                                          init_from=(U, V, W))
                if U_new is not None and err < 1e-8:
                    if _verify(U_new, V_new, W_new, T):
                        U, V, W = U_new, V_new, W_new
                        break

    # Verify after reduction
    if not _verify(U, V, W, T):
        # Revert to rank 22
        U = U_fallback
        V = V_fallback
        W = W_fallback

    # Convert to list format with clean values
    # Round to handle floating point noise
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)

    # Final safety check
    if not _verify(U, V, W, T):
        U = np.round(U_fallback, 10)
        V = np.round(V_fallback, 10)
        W = np.round(W_fallback, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
    T = _tensor_index(2, 3, 4)
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify(U, V, W, T)}")
