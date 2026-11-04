import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    Uses a 2+3 block structure with Strassen on the 2x2x2 block,
    plus a rank-reducing merge across the full 5x5x5 tensor.
    """
    start = time.time()
    deadline = start + 110

    n, m, p = 5, 5, 5

    try:
        result = _construct_merged(n, m, p, deadline)
        if result is not None and _verify_decomposition(result, n, m, p):
            return result
    except Exception:
        pass

    try:
        result = _construct_block_strassen(n, m, p)
        if result is not None and _verify_decomposition(result, n, m, p):
            return result
    except Exception:
        pass

    return _construct_schoolbook(n, m, p)


def _verify_decomposition(result, n, m, p):
    """Vectorized verification."""
    U = np.asarray(result['U'], dtype=np.float64)
    V = np.asarray(result['V'], dtype=np.float64)
    W = np.asarray(result['W'], dtype=np.float64)
    rank = U.shape[1]

    # Build the tensor: for each (i,j,k), check sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    # Vectorize over (i,j,k):
    # U_flat: (n*m, rank) -> U[i*m+j, r]
    # V_flat: (m*p, rank) -> V[j*p+k, r]
    # W_flat: (p*n, rank) -> W[k*n+i, r]

    # For each r, form outer products appropriately
    # Sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Reshape approach: 
    # Create arrays indexed by (i,j,k)
    # U contributes: for fixed r, U[:,r] has shape (n*m,) = (i,j) with i*m+j
    # V[:,r] has shape (m*p,) = (j,k) with j*p+k
    # W[:,r] has shape (p*n,) = (k,i) with k*n+i

    total = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)       # u_r[i,j]
        v_r = V[:, r].reshape(m, p)       # v_r[j,k]
        w_r = W[:, r].reshape(p, n)       # w_r[k,i]
        # For each (i,j,k): u_r[i,j] * v_r[j,k] * w_r[k,i]
        # Build tensor: (i,j,k) = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # Use einsum: u_r[i,j] * v_r[j,k] gives (i,k), then * w_r[k,i] -> need (i,j,k)
        # Actually: sum over j is not right. We want the product for each (i,j,k).
        # u_r[i,j] * v_r[j,k] * w_r[k,i]
        # = for each (i,j,k): u_r[i,j] * v_r[j,k] * w_r[k,i]
        # Reshape: u_r[i,j] -> (i,j,1), v_r[j,k] -> (1,j,k) broadcast, w_r[k,i] -> (i,1,k)
        total += u_r[:, :, None] * v_r[None, :, :] * w_r[None, :, :]
        # Wait: u_r is (n,m) so u_r[:,:,None] is (n,m,1)
        # v_r is (m,p) so v_r[None,:,:] is (1,m,p)
        # w_r is (p,n) so w_r.T is (n,p) -> w_r.T[:, :, None]? No.
        # w_r[k,i] where k is first index, i is second.
        # We need w_r indexed as [k,i] for each (i,j,k).
        # So we need an array of shape (n, m, p) where element [i,j,k] = w_r[k,i].
        # w_r has shape (p,n): w_r[k,i]. To get shape (n,m,p): 
        #   w_r[k,i] for fixed i,k -> need to broadcast to (n, m, p)
        #   reshape w_r to (1, 1, p, n)? No.
        #   We want result[i,j,k] = w_r[k,i]
        #   Create tmp: shape (n, p) where tmp[i,k] = w_r[k,i] = w_r.T[i,k]
        #   Then tmp[:, None, :] has shape (n, 1, p), broadcast with (n,m,1) and (1,m,p)
        
    # Let me redo this more carefully
    total = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)       # shape (n, m), u_r[i, j]
        v_r = V[:, r].reshape(m, p)       # shape (m, p), v_r[j, k]
        w_r = W[:, r].reshape(p, n)       # shape (p, n), w_r[k, i]
        
        # Need product[i, j, k] = u_r[i, j] * v_r[j, k] * w_r[k, i]
        # u_r[i,j]: shape (n, m) -> expand to (n, m, 1)
        # v_r[j,k]: shape (m, p) -> expand to (1, m, p)
        # w_r[k,i]: shape (p, n) -> need as (n, 1, p) where element [i, 0, k] = w_r[k, i]
        #   w_r.T has shape (n, p), w_r.T[i, k] = w_r[k, i]. Expand to (n, 1, p).
        
        product = u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]
        total += product

    if np.max(np.abs(total - 1.0)) > 1e-10:
        return False
    return True


def _construct_block_strassen(n, m, p):
    """
    Block decomposition: split into {0,1}, {2,3}, {4}.
    Apply Strassen (rank 7) to 2x2x2 diagonal blocks; direct elsewhere.
    """
    B = [[0, 1], [2, 3], [4]]
    sizes = [2, 2, 1]
    
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[u_idx] += u_val
        v[v_idx] += v_val
        w[w_idx] += w_val
        terms.append((u, v, w))
    
    def add_strassen_block(idx_i, idx_j, idx_k):
        """Apply Strassen to 2x2x2 block with given global indices."""
        M = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        for (u_dict, v_dict, w_dict) in M:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            for (li, lj), val in u_dict.items():
                u[idx_i[li] * m + idx_j[lj]] = val
            for (lj, lk), val in v_dict.items():
                v[idx_j[lj] * p + idx_k[lk]] = val
            for (lk, li), val in w_dict.items():
                w[idx_k[lk] * n + idx_i[li]] = val
            terms.append((u, v, w))
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                if bi == bj == bk:
                    if sizes[bi] == 2:
                        add_strassen_block(B[bi], B[bj], B[bk])
                    else:
                        # Direct for 1x1x1
                        i = B[bi][0]; j = B[bj][0]; k = B[bk][0]
                        add_term(i*m+j, 1.0, j*p+k, 1.0, k*n+i, 1.0)
                else:
                    # Direct multiplication
                    for ii in B[bi]:
                        for jj in B[bj]:
                            for kk in B[bk]:
                                add_term(ii*m+jj, 1.0, jj*p+kk, 1.0, kk*n+ii, 1.0)
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _construct_merged(n, m, p, deadline):
    """
    Build the block Strassen decomposition, then attempt to merge linearly
    dependent rank terms by solving for redundancy.
    """
    start = time.time()
    
    # First build the base decomposition
    base = _construct_block_strassen(n, m, p)
    U_base = np.asarray(base['U'], dtype=np.float64)
    V_base = np.asarray(base['V'], dtype=np.float64)
    W_base = np.asarray(base['W'], dtype=np.float64)
    rank_base = U_base.shape[1]
    
    # Check if any columns of U are zero (which would be trivially removable)
    new_terms = []
    for r in range(rank_base):
        if np.any(U_base[:, r] != 0):
            new_terms.append(r)
    
    U_new = U_base[:, new_terms]
    V_new = V_base[:, new_terms]
    W_new = W_base[:, new_terms]
    rank_new = U_new.shape[1]
    
    if rank_new < rank_base:
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    elapsed = time.time() - start
    if elapsed > 30:
        return base
    
    # Check numerical ranks
    U_rank = np.linalg.matrix_rank(U_base, tol=1e-10)
    V_rank = np.linalg.matrix_rank(V_base, tol=1e-10)
    W_rank = np.linalg.matrix_rank(W_base, tol=1e-10)
    
    min_rank = min(U_rank, V_rank, W_rank)
    
    if min_rank < rank_base:
        if U_rank == min_rank:
            U_pivots = _find_pivot_cols(U_base)
        elif V_rank == min_rank:
            U_pivots = list(range(rank_base))
        else:
            U_pivots = list(range(rank_base))
        
        if len(U_pivots) < rank_base:
            U_sel = U_base[:, U_pivots]
            V_sel = V_base[:, U_pivots]
            W_sel = W_base[:, U_pivots]
            result = {"U": U_sel.tolist(), "V": V_sel.tolist(), "W": W_sel.tolist()}
            if _verify_decomposition(result, n, m, p):
                return result
    
    # Attempt 3: Try to improve the 3x3x3 block with a lower-rank decomposition
    if time.time() - start < 90:
        result = _try_improve_3x3x3(n, m, p, U_base, V_base, W_base, deadline)
        if result is not None and _verify_decomposition(result, n, m, p):
            return result
    
    return base


def _find_pivot_cols(matrix, tol=1e-10):
    """Find pivot columns of a matrix using QR decomposition."""
    try:
        Q, R = np.linalg.qr(matrix)
        diag = np.abs(np.diag(R))
        pivots = []
        for j in range(matrix.shape[1]):
            if diag[j] > tol:
                pivots.append(j)
        return pivots
    except:
        return list(range(matrix.shape[1]))


def _try_random_merge(n, m, p, U, V, W, deadline):
    """
    Try to reduce rank by finding linear dependencies.
    For each column r, check if it can be expressed as a linear combination
    of other columns in all three matrices simultaneously.
    """
    rank = U.shape[1]
    best_result = None
    best_rank = rank
    
    # For each column, try to eliminate it
    for r_elim in range(rank):
        if time.time() > deadline - 10:
            break
        
        # Try to express column r_elim as combination of others
        # Solve: U[:,r_elim] = sum_{r != r_elim} alpha_r * U[:,r]
        #         V[:,r_elim] = sum_{r != r_elim} alpha_r * V[:,r]  
        #         W[:,r_elim] = sum_{r != r_elim} alpha_r * W[:,r]
        
        other_cols = [r for r in range(rank) if r != r_elim]
        
        # Set up linear system from U
        A_u = U[:, other_cols]
        b_u = U[:, r_elim]
        
        # Check if b_u is in column space of A_u
        try:
            # Least squares
            sol_u, residuals_u, rank_u, _ = np.linalg.lstsq(A_u, b_u, rcond=1e-10)
            # Check residual
            if len(residuals_u) > 0 and residuals_u[0] > 1e-8:
                continue
            # Verify with V
            pred_v = V[:, other_cols] @ sol_u
            if np.max(np.abs(pred_v - V[:, r_elim])) > 1e-8:
                continue
            # Verify with W
            pred_w = W[:, other_cols] @ sol_u
            if np.max(np.abs(pred_w - W[:, r_elim])) > 1e-8:
                continue
            
            # Column r_elim is redundant!
            new_rank = rank - 1
            if new_rank < best_rank:
                keep = other_cols
                U_new = U[:, keep]
                V_new = V[:, keep]
                W_new = W[:, keep]
                best_rank = new_rank
                best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        except:
            continue
    
    return best_result


def _try_improve_3x3x3(n, m, p, U, V, W, deadline):
    """
    Try to improve the 3x3x3 block in the decomposition.
    The 3x3x3 block uses direct multiplication (27 terms).
    Try to find a lower-rank decomposition for this block.
    """
    # For now, try the random merge approach on the full decomposition
    return _try_random_merge(n, m, p, U, V, W, deadline)


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
