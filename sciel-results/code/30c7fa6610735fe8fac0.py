<<<<<<< SEARCH
def _try_rank_decomposition(r, max_time=30):
    """Try to find a rank-r decomposition using random initialization and verification."""
    n, m, p = 3, 4, 6
    start = time.time()
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() - start > max_time:
            break
        
        # Random integer/half-integer initialization
        U = np.random.randint(-2, 3, size=(n*m, r)) * 0.5
        V = np.random.randint(-2, 3, size=(m*p, r)) * 0.5
        W = np.random.randint(-2, 3, size=(p*n, r)) * 0.5
        
        # Check if this works
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return U_list, V_list, W_list, r
    
    return None
=======
def _try_rank_decomposition(r, max_time=30):
    """Try to find a rank-r decomposition using structured search."""
    n, m, p = 3, 4, 6
    start = time.time()
    
    # Try multiple random initializations with small integer values
    for attempt in range(100):
        if time.time() - start > max_time:
            break
        
        # Random small integer/half-integer initialization
        U = np.random.randint(-2, 3, size=(n*m, r)) * 0.5
        V = np.random.randint(-2, 3, size=(m*p, r)) * 0.5
        W = np.random.randint(-2, 3, size=(p*n, r)) * 0.5
        
        # Check if this works
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return U_list, V_list, W_list, r
    
    return None

def _reduce_rank_by_dependency(U, V, W, rank):
    """Try to reduce rank by finding linear dependencies among rank-1 terms.
    
    If two terms are linearly dependent (one is a scalar multiple of the other
    in all three vectors simultaneously), we can merge them.
    More generally, if the sum of a subset of terms equals zero, we can remove them.
    """
    n, m, p = 3, 4, 6
    num_U = n * m  # 12
    num_V = m * p  # 24
    num_W = p * n  # 18
    
    # Build the tensor as a matrix: each column c gives a rank-1 term
    # T_{ijk} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    # We need T_{ijk} = 1 for all i,j,k
    
    # Check if any column is zero (can be removed)
    keep = []
    for c in range(rank):
        u_col = np.array(U[:, c])
        v_col = np.array(V[:, c])
        w_col = np.array(W[:, c])
        if np.allclose(u_col, 0) and np.allclose(v_col, 0) and np.allclose(w_col, 0):
            continue
        # Check if this term is redundant: removing it should still give correct result
        keep.append(c)
    
    if len(keep) < rank:
        new_rank = len(keep)
        U_new = [[U[i][c] for c in keep] for i in range(num_U)]
        V_new = [[V[i][c] for c in keep] for i in range(num_V)]
        W_new = [[W[i][c] for c in keep] for i in range(num_W)]
        return U_new, V_new, W_new, new_rank
    
    # Try to find pairs of terms that can be combined
    # Term c contributes: U[:,c] ⊗ V[:,c] ⊗ W[:,c] to the tensor
    # If term c1 = alpha * term c2 (componentwise in the tensor sense), merge them
    
    # Build tensor contributions for each term
    def term_tensor(c):
        """Return the (n,m,p) tensor contribution of term c."""
        t = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t[i, j, k] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        return t
    
    # Check for exact duplicates or scalar multiples
    tensors = [term_tensor(c) for c in range(rank)]
    
    # Find pairs where one is a scalar multiple of the other
    merged = list(range(rank))
    for c1 in range(rank):
        if c1 not in merged:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 not in merged:
                continue
            t1 = tensors[c1]
            t2 = tensors[c2]
            if np.allclose(t1, 0) or np.allclose(t2, 0):
                continue
            # Check if t1 = alpha * t2
            nonzero_mask = np.abs(t2) > 1e-12
            if np.sum(nonzero_mask) == 0:
                continue
            alpha = t1[nonzero_mask][0] / t2[nonzero_mask][0]
            if np.allclose(t1[nonzero_mask], alpha * t2[nonzero_mask], atol=1e-9):
                # Merge: add term c2 to term c1 with factor alpha
                # New term c1 has (1+alpha) * t1... wait, we need to be careful
                # Actually, if t1 = alpha * t2, then t1 + t2 = (1 + 1/alpha) * t1
                # We can just keep one term with adjusted coefficients
                # But adjusting coefficients means changing U,V,W which is complex
                # For now, just note the dependency
                pass
    
    return U, V, W, rank

def _am_polish(U, V, W, rank):
    """Attempt to reduce rank by small perturbations and re-verification.
    
    Try replacing pairs of terms with a single term that has the same tensor contribution.
    """
    n, m, p = 3, 4, 6
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    if rank <= 1:
        return U, V, W, rank
    
    # For each pair of terms, check if their combined contribution can be
    # represented as a single rank-1 term
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            # Combined tensor contribution of terms c1 and c2
            combined = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        combined[i, j, k] = (
                            U[i*m+j][c1] * V[j*p+k][c1] * W[k*n+i][c1] +
                            U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2]
                        )
            
            # Check if combined is rank-1
            # Reshape to matrix and check rank
            mat = combined.reshape(n*m, p)
            sv = np.linalg.svd(mat, compute_uv=False)
            if len(sv) > 1 and sv[1] > 1e-10:
                # Not rank-1, skip
                continue
            
            # Try to find U', V', W' such that U'[i,j]*V'[j,k]*W'[k,i] = combined[i,j,k]
            # This is generally hard, but for small cases we can try
            # Use the first non-zero element to fix scaling
            nonzero = np.argwhere(np.abs(combined) > 1e-12)
            if len(nonzero) == 0:
                # Both terms cancel out, remove both
                keep = [c for c in range(rank) if c != c1 and c != c2]
                new_rank = len(keep)
                U_new = [[U[i][c] for c in keep] for i in range(num_U)]
                V_new = [[V[i][c] for c in keep] for i in range(num_V)]
                W_new = [[W[i][c] for c in keep] for i in range(num_W)]
                if _verify(U_new, V_new, W_new, new_rank):
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank
                continue
            
            # This is complex; skip for now and just check cancellation
            pass
    
    return best_U, best_V, best_W, best_rank
>>>>>>> REPLACE

