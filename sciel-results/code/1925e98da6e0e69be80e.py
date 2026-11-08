import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def _als_tensor_decomposition(n, m, p, max_rank, deadline, seed=42):
    """
    Use Alternating Least Squares to find a low-rank decomposition of the
    matrix multiplication tensor T[i,j,k] = delta_{sum} for C = A*B.
    
    The tensor is: T_{(i,j),(j,k),(k,i)} = 1 if indices match, 0 otherwise.
    We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = T.
    
    This is equivalent to finding rank-r factor matrices such that:
    sum_r U[:,r] V[:,r]^T (elementwise) ... actually let's think in terms of
    the 3-way tensor unfolding.
    
    The multiplication tensor has entries:
    T[a, b, c] where a = (i,j) in [0,nm), b = (j,k) in [0,mp), c = (k,i) in [0,pn)
    T[a,b,c] = 1 if the three indices are consistent (j matches between a and b,
    k matches between b and c, i matches between c and a), else 0.
    
    For a rank-r decomposition:
    T[a,b,c] = sum_{r=0}^{r-1} U[a,r] * V[b,r] * W[c,r]
    
    ALS: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W.
    Each solve is a linear system.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    best_result = None
    best_rank = max_rank + 1
    
    # Try ranks from low to high
    for rank in range(7, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 15 if rank <= 40 else 8
        max_iters = 200 if rank <= 30 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random small values
            U = rng.randn(nm, rank) * 0.5
            V = rng.randn(mp, rank) * 0.5
            W = rng.randn(pn, rank) * 0.5
            
            prev_err = np.inf
            converged = False
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Solve for U: for each a, sum_r U[a,r] * (V[:,r] * W[:,r])[b,c] = T[a,b,c]
                # This is: U[a,:] @ M_U[a] = T[a,:,:] where M_U[a] depends on V,W
                # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # For fixed V,W: define Z[b,c] outer product structure
                # U[a,r] appears multiplied by (V[:,r] ⊗ W[:,r])[b,c]
                # So: U[a,:] @ (V * W.T) reshaped... 
                # Let's define: for each r, the contribution to T from term r is
                # U[:,r] * V[:,r] * W[:,r] (outer product)
                # So T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # For U: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # = sum_r U[a,r] * (V[:,r] * W[:,r].T)  [as a matrix indexed by (b,c)]
                # So: T[a] = U[a,:] @ M where M[r, (b,c)] = V[b,r] * W[c,r]
                # M has shape (rank, mp*pn)
                
                # Build M_U: shape (rank, mp*pn)
                # M_U[r, b*pn+c] = V[b,r] * W[c,r]
                M_U = (V[:, :, None] * W[None, :, :]).T.reshape(rank, mp * pn)
                # M_U[r, idx] where idx = b*pn + c, value = V[b,r]*W[c,r]
                
                # T_flat[a] = T[a,:,:].reshape(-1), shape (mp*pn,)
                T_flat = T.reshape(nm, mp * pn)
                
                # Solve: U @ M_U.T = T_flat  =>  U[a,:] @ M_U.T = T_flat[a,:]
                # This is: for each a, solve (M_U @ M_U.T) U[a,:]^T = M_U @ T_flat[a,:]^T
                # Actually: U[a,:] @ M_U.T = T_flat[a,:]
                # => M_U @ U[a,:]^T = T_flat[a,:]^T
                # => (M_U.T @ M_U) U[a,:]^T = M_U.T @ T_flat[a,:]^T
                
                try:
                    gram_U = M_U @ M_U.T  # (rank, rank)
                    rhs_U = M_U @ T_flat.T  # (rank, nm)
                    # Solve gram_U @ U.T = rhs_U
                    U_new = np.linalg.solve(gram_U + 1e-10 * np.eye(rank), rhs_U).T
                except np.linalg.LinAlgError:
                    try:
                        U_new = np.linalg.lstsq(M_U, T_flat.T, rcond=None)[0].T
                    except:
                        break
                
                # Similarly for V:
                # T[a,b,:] = sum_r V[b,r] * (U[a,r] * W[c,r])
                # M_V[r, a*pn+c] = U[a,r] * W[c,r]
                M_V = (U[:, :, None] * W[None, :, :]).T.reshape(rank, nm * pn)
                T_flat_V = T.reshape(mp, nm * pn)
                # T_flat_V[b, a*pn+c] = T[a,b,c]
                # Need: T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # For fixed b: sum_r V[b,r] * (U[:,r] * W[:,r])[a,c] = T[:,b,:].reshape(-1)
                # M_V[r, a*pn+c] = U[a,r]*W[c,r]
                # V[b,:] @ M_V.T = T[:,b,:].reshape(-1)
                # M_V @ V[b,:]^T = T[:,b,:].reshape(-1)^T
                
                try:
                    gram_V = M_V @ M_V.T
                    rhs_V = M_V @ T_flat_V.T
                    V_new = np.linalg.solve(gram_V + 1e-10 * np.eye(rank), rhs_V).T
                except np.linalg.LinAlgError:
                    try:
                        V_new = np.linalg.lstsq(M_V, T_flat_V.T, rcond=None)[0].T
                    except:
                        break
                
                # Similarly for W:
                # T[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
                # M_W[r, a*mp+b] = U[a,r] * V[b,r]
                M_W = (U[:, :, None] * V[None, :, :]).T.reshape(rank, nm * mp)
                T_flat_W = T.reshape(pn, nm * mp)
                # T_flat_W[c, a*mp+b] = T[a,b,c]
                # W[c,:] @ M_W.T = T_flat_W[c,:]
                
                try:
                    gram_W = M_W @ M_W.T
                    rhs_W = M_W @ T_flat_W.T
                    W_new = np.linalg.solve(gram_W + 1e-10 * np.eye(rank), rhs_W).T
                except np.linalg.LinAlgError:
                    try:
                        W_new = np.linalg.lstsq(M_W, T_flat_W.T, rcond=None)[0].T
                    except:
                        break
                
                U, V, W = U_new, V_new, W_new
                
                # Compute error
                # err = ||T - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||_F
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(recon - T)
                
                if err < prev_err * (1 - 1e-6) and it > 5:
                    # Converged
                    if err < 1e-6:
                        converged = True
                        break
                    if err < 1e-4:
                        # Keep going a bit more
                        pass
                if err < 1e-8:
                    converged = True
                    break
                
                prev_err = err
            
            if converged or (err < 1e-6 if 'err' in dir() else False):
                # Verify exactly
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                max_err = np.max(np.abs(recon - T))
                if max_err < 1e-9:
                    # Round to half-integers if close
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    recon_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                    if np.max(np.abs(recon_r - T)) < 1e-9:
                        U, V, W = U_round, V_round, W_round
                    best_result = (U.copy(), V.copy(), W.copy())
                    best_rank = rank
                    break  # Found a valid decomposition at this rank
            elif err < 1e-4 and best_rank > rank:
                # Near convergence, keep as candidate
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                max_err = np.max(np.abs(recon - T))
                if max_err < 1e-6:
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    recon_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                    if np.max(np.abs(recon_r - T)) < 1e-9:
                        best_result = (U_round.copy(), V_round.copy(), W_round.copy())
                        best_rank = rank
                        break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    u1 = {(0,0): 1, (1,1): 1}; v1 = {(0,0): 1, (1,1): 1}; w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    u2 = {(1,0): 1, (1,1): 1}; v2 = {(0,0): 1}; w2 = {(0,1): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    u3 = {(0,0): 1}; v3 = {(0,1): 1, (1,1): -1}; w3 = {(1,0): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    u4 = {(1,1): 1}; v4 = {(1,0): 1, (0,0): -1}; w4 = {(0,0): 1, (0,1): 1}
    terms.append((u4, v4, w4))
    u5 = {(0,0): 1, (0,1): 1}; v5 = {(1,1): 1}; w5 = {(0,0): -1, (1,0): 1}
    terms.append((u5, v5, w5))
    u6 = {(1,0): 1, (0,0): -1}; v6 = {(0,0): 1, (0,1): 1}; w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    u7 = {(0,1): 1, (1,1): -1}; v7 = {(1,0): 1, (1,1): 1}; w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    return terms

def _build_block_decomp(n, m, p, i_groups, j_groups, k_groups):
    """Build block decomposition with Strassen for 2x2x2 blocks."""
    strassen_terms = _strassen_2x2x2_terms()
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    terms_list = []
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                if si == 2 and sj == 2 and sk == 2:
                    for (u_pat, v_pat, w_pat) in strassen_terms:
                        u_term = {(ig[li], jg[lj]): val for (li, lj), val in u_pat.items()}
                        v_term = {(jg[lj], kg[lk]): val for (lj, lk), val in v_pat.items()}
                        w_term = {(kg[lk], ig[li]): val for (lk, li), val in w_pat.items()}
                        terms_list.append((u_term, v_term, w_term))
                else:
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                gi, gj, gk = ig[li], jg[lj], kg[lk]
                                terms_list.append(
                                    ({(gi, gj): 1}, {(gj, gk): 1}, {(gk, gi): 1})
                                )
    
    rank = len(terms_list)
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    
    for col, (u_term, v_term, w_term) in enumerate(terms_list):
        for (i, j), val in u_term.items():
            U[i * m + j, col] = val
        for (j, k), val in v_term.items():
            V[j * p + k, col] = val
        for (k, i), val in w_term.items():
            W[k * n + i, col] = val
    
    return U, V, W, rank

def _verify_decomp(U, V, W, n, m, p):
    """Verify decomposition is exact."""
    T = _build_tensor(n, m, p)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(recon - T)) < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n, m, p = DIMENSIONS
    
    # Try multiple block partitions, pick best
    partitions = [
        # (i_groups, j_groups, k_groups)
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),  # rank 84
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0,1,2],[3,4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0,1],[2,3,4]], [[0,1],[2,3],[4,5]]),
        ([[0,1,2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1,2],[3,4,5]]),
        ([[0,1],[2]], [[0,1,2],[3],[4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0],[1,2],[3,4]], [[0,1],[2,3],[4,5]]),
        ([[0],[1,2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0,1],[2],[3,4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2],[3,4,5]]),
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0],[1,2],[3,4,5]]),
    ]
    
    best_U, best_V, best_W, best_rank = None, None, None, n*m*p
    
    for ig, jg, kg in partitions:
        if time.time() > deadline - 60:
            break
        try:
            U, V, W, r = _build_block_decomp(n, m, p, ig, jg, kg)
            if _verify_decomp(U, V, W, n, m, p) and r < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, r
        except Exception:
            pass
    
    # If we found a block decomposition, try to compress it via ALS
    if best_U is not None and time.time() < deadline - 30:
        init_UVW = (best_U, best_V, best_W)
        try:
            als_result = _als_tensor_decomposition(
                n, m, p, max_rank=best_rank - 1,
                deadline=deadline - 5, seed=42, init_UVW=init_UVW
            )
            if als_result is not None:
                U_arr = np.array(als_result["U"])
                V_arr = np.array(als_result["V"])
                W_arr = np.array(als_result["W"])
                if _verify_decomp(U_arr, V_arr, W_arr, n, m, p):
                    return als_result
        except Exception:
            pass
        
        # Try more seeds
        if time.time() < deadline - 15:
            for s in [123, 456, 789]:
                if time.time() > deadline - 5:
                    break
                try:
                    als_result = _als_tensor_decomposition(
                        n, m, p, max_rank=best_rank - 1,
                        deadline=deadline - 5, seed=s, init_UVW=init_UVW
                    )
                    if als_result is not None:
                        U_arr = np.array(als_result["U"])
                        V_arr = np.array(als_result["V"])
                        W_arr = np.array(als_result["W"])
                        if _verify_decomp(U_arr, V_arr, W_arr, n, m, p):
                            return als_result
                except Exception:
                    pass
    
    # Return best block decomposition
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: full schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
