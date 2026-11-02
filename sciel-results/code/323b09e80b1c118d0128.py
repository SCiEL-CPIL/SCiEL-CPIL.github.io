import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen patterns for 2x2 matrix multiplication
    strassen_blocks = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]

    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    block_configs = []
    for u in range(2):
        for v in range(4):
            a_ids = [0 * m + 2 * u, 0 * m + 2 * u + 1, 1 * m + 2 * u, 1 * m + 2 * u + 1]
            b_ids = [(2 * u) * p + 2 * v, (2 * u) * p + 2 * v + 1, (2 * u + 1) * p + 2 * v, (2 * u + 1) * p + 2 * v + 1]
            c_ids = [(2 * v) * n + 0, (2 * v + 1) * n + 0, (2 * v) * n + 1, (2 * v + 1) * n + 1]
            base_col = 7 * (4 * u + v)
            block_configs.append((a_ids, b_ids, c_ids, base_col))

    for a_ids, b_ids, c_ids, base_col in block_configs:
        for s_idx, pattern in enumerate(strassen_blocks):
            col = base_col + s_idx
            for idx, coeff in pattern['u']:
                U[a_ids[idx], col] += coeff
            for idx, coeff in pattern['v']:
                V[b_ids[idx], col] += coeff
            for idx, coeff in pattern['w']:
                W[c_ids[idx], col] += coeff

    def verify_fast(U, V, W):
        """Fast verification using tensor unfolding via matrix multiplication."""
        # Tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Unfold along (i,j): for each (i,j), T_ij = U_ij @ diag(V_jk * W_ki) ... 
        # Better: for each k, T[:,:,k] = (U[:, :, k] outer V[:, k]) * W[k, :, :]... 
        # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed k: T[:,:,k][i,j] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
        # = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This is like: for each k, form matrix M_k where M_k[i,j] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We can compute: for each k, A_k = U.reshape(n,m,-1) * V_k * W_k
        # where V_k[j,r] = V[j*p+k,r], W_k[i,r] = W[k*n+i,r]
        # T[:,:,k] = sum over r of (U[:,:,r] * V_k[:,r][:,None] * W_k[:,r][None,:])
        # = U_unfold @ diag stuff... 
        # Simplest fast method: use einsum
        # T = np.einsum('ir,jr,kr->ijk', U.reshape(n,m,-1), V.reshape(m,p,-1), W.reshape(p,n,-1))
        # But that's 64 elements * rank ops. Let's just do it with einsum.
        T = np.einsum('ir,jr,kr->ijk', U.reshape(n, m, -1), V.reshape(m, p, -1), W.reshape(p, n, -1))
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_fast_vals(U, V, W):
        """Return the actual tensor values for debugging."""
        return np.einsum('ir,jr,kr->ijk', U.reshape(n, m, -1), V.reshape(m, p, -1), W.reshape(p, n, -1))

    # Verify initial decomposition
    if not verify_fast(U, V, W):
        U, V, W, rank = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by removing linearly dependent terms
    R = rank
    keep = list(range(R))
    
    # Precompute for faster checks
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    for col in range(R):
        if time.time() > deadline - 15:
            break
        if col not in keep:
            continue
        # Temporarily remove this column
        mask = np.ones(R, dtype=bool)
        mask[col] = False
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        if verify_fast(U_test, V_test, W_test):
            keep.remove(col)
            U = U_test
            V = V_test
            W = W_test
            R -= 1
            # Update reshaped views
            U_r = U.reshape(n, m, -1)
            V_r = V.reshape(m, p, -1)
            W_r = W.reshape(p, n, -1)

    if R < rank:
        rank = R

    # Try adding one column and removing two (net -1)
    if time.time() < deadline - 30:
        R2 = R
        # Try each pair of columns to remove, and see if we can add one column to compensate
        # This is expensive, so only try a limited number
        for remove1 in range(R2):
            if time.time() > deadline - 20:
                break
            for remove2 in range(remove1 + 1, R2):
                if time.time() > deadline - 15:
                    break
                # Remove both columns
                mask = np.ones(R2, dtype=bool)
                mask[remove1] = False
                mask[remove2] = False
                U_base = U[:, mask]
                V_base = V[:, mask]
                W_base = W[:, mask]
                
                # Compute the error tensor
                T_err = 1.0 - np.einsum('ir,jr,kr->ijk', 
                                        U_base.reshape(n, m, -1), 
                                        V_base.reshape(m, p, -1), 
                                        W_base.reshape(p, n, -1))
                
                # Check if T_err has rank 1 (i.e., can be expressed as u*v*w)
                # T_err[i,j,k] = a[i]*b[j]*c[k] for some vectors a, b, c
                # Try: fix i=0, j=0, then T_err[0,0,k] = a[0]*b[0]*c[k]
                # T_err[i,0,k] = a[i]*b[0]*c[k]
                # T_err[i,j,k] = a[i]*b[j]*c[k]
                # So T_err[i,j,k] / (T_err[i,0,k] * T_err[0,j,k]) should equal T_err[0,0,k] / (T_err[0,0,k])^2 * ... 
                # Simpler: check if T_err[i,j,k] * T_err[0,0,0] == T_err[i,0,k] * T_err[0,j,k] for all i,j,k
                # This is the rank-1 condition for a 3D tensor.
                
                T000 = T_err[0, 0, 0]
                if abs(T000) < 1e-12:
                    # Try a different pivot
                    found_pivot = False
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                if abs(T_err[ii, jj, kk]) > 1e-12:
                                    T000 = T_err[ii, jj, kk]
                                    found_pivot = True
                                    break
                            if found_pivot:
                                break
                        if found_pivot:
                            break
                    if not found_pivot:
                        # T_err is zero, we can remove both columns!
                        U = U_base
                        V = V_base
                        W = W_base
                        R2 -= 2
                        R = R2
                        break
                    # Re-check with new pivot... this gets complicated, skip for now
                    continue
                
                # Check rank-1: T[i,j,k] * T000 == T[i,0,k] * T[0,j,k]
                is_rank1 = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if abs(T_err[i, j, k] * T000 - T_err[i, 0, k] * T_err[0, j, k]) > 1e-9:
                                is_rank1 = False
                                break
                        if not is_rank1:
                            break
                    if not is_rank1:
                        break
                
                if is_rank1:
                    # Extract the rank-1 components
                    # T[i,j,k] = a[i]*b[j]*c[k] where a[0]*b[0]*c[0] = T000
                    # a[i] = T[i,0,0] / b[0] / c[0]... 
                    # Let's set c[0] = 1, then a[i]*b[j] = T[i,j,0] and a[i]*c[k] = T[i,0,k]
                    # b[j] = T[0,j,0], c[k] = T[0,0,k] / a[0]
                    # a[0] = T[0,0,0] / (b[0]*c[0])... circular
                    # Set: a[i] = T[i,0,0], b[j] = T[0,j,0] / a[0], c[k] = T[0,0,k] / a[0]
                    a0 = T_err[0, 0, 0]
                    if abs(a0) > 1e-12:
                        a_vec = T_err[:, 0, 0].copy()
                        b_vec = T_err[0, :, 0] / a0
                        c_vec = T_err[0, 0, :] / a0
                        # Verify: a[i]*b[j]*c[k] should equal T[i,j,k]
                        test_val = np.einsum('i,j,k->ijk', a_vec, b_vec, c_vec)
                        if np.allclose(test_val, T_err, atol=1e-9):
                            # Add this column
                            U_new = np.column_stack([U_base, a_vec.flatten()])
                            V_new = np.column_stack([V_base, b_vec.flatten()])
                            W_new = np.column_stack([W_base, c_vec.flatten()])
                            if verify_fast(U_new, V_new, W_new):
                                U = U_new
                                V = V_new
                                W = W_new
                                R2 -= 1
                                R = R2
                                break
                        else:
                            is_rank1 = False
                if is_rank1:
                    break
            if R2 < R:
                break

    if R < rank:
        rank = R

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
