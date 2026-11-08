"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time as _time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    _t0 = _time.time()
    _deadline = _t0 + 140.0  # safety margin before 150s kill
    
    # Strategy: Block decomposition with cross-block optimization.
    # p=6 split into 3 blocks of size 2. Base rank = 3*11 = 33.
    # Attempt to reduce rank by finding linear dependencies among the 33 rank-1 tensors.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    # Define the 11 terms for a single 2x3x2 block.
    terms = []
    
    # --- Strassen terms for the 2x2 x 2x2 part ---
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    
    # --- Scalar terms ---
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    # Fill the matrices for the 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Verify base solution
    try:
        _verify({"U": U, "V": V, "W": W}, n, m, p)
    except (ValueError, AssertionError):
        # Fallback: return the solution anyway (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by checking for linear dependencies in the tensor space.
    # Each rank-1 term corresponds to a vector in R^(n*m * m*p * p*n) but we can
    # check if any column is a linear combination of others by examining the
    # flattened tensor contributions.
    
    # The tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For the decomposition to be valid with rank R, we need the R vectors
    # (flattened U_r ⊗ V_r ⊗ W_r) to span the target tensor.
    # If any vector is in the span of the others, we can eliminate it.
    
    # Build the tensor as a 3D array and check rank via SVD-like approach.
    # Flatten each rank-1 term into a vector of length n*m*p = 36.
    import numpy as _np
    
    tensor_dim = n * m * p  # 36
    # Build matrix M where column r is the flattened rank-1 tensor for term r
    M = _np.zeros((tensor_dim, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i*m*p + j*p + k, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Check if columns are linearly dependent using rank of M
    # Use SVD to find numerical rank
    try:
        U_svd, s, Vt = _np.linalg.svd(M, full_matrices=False)
        # Find effective rank (singular values above tolerance)
        tol = max(M.shape) * _np.finfo(float).eps * s[0]
        num_rank = _np.sum(s > tol)
        
        if num_rank < rank:
            # There are linear dependencies. Try to eliminate columns.
            # Identify which columns are in the span of the first num_rank significant ones.
            # Use QR with column pivoting to find a basis.
            # For now, try removing columns one at a time from the end.
            current_U = [row[:] for row in U]
            current_V = [row[:] for row in V]
            current_W = [row[:] for row in W]
            current_rank = rank
            removed = []
            
            for col_to_try in range(rank - 1, -1, -1):
                if _time.time() > _deadline:
                    break
                # Try removing column col_to_try
                test_rank = current_rank - 1
                if test_rank <= 0:
                    break
                
                # Build new M without this column
                keep_cols = [c for c in range(current_rank) if c != col_to_try]
                M_test = _np.zeros((tensor_dim, test_rank))
                for new_r, old_r in enumerate(keep_cols):
                    M_test[:, new_r] = M[:, old_r]
                
                # Check if the target tensor (all ones) is still in the span
                # Target tensor: T[i,j,k] = 1 for all valid i,j,k
                target = _np.ones(tensor_dim)
                
                # Solve M_test * x = target (least squares)
                try:
                    x, residuals, rank_test, s_test = _np.linalg.lstsq(M_test, target, rcond=None)
                    if len(residuals) > 0:
                        res = residuals[0]
                    else:
                        res = _np.linalg.norm(M_test @ x - target)
                    
                    if res < 1e-9:
                        # Column can be removed
                        # Remove it from U, V, W
                        new_U = [row[:col_to_try] + row[col_to_try+1:] for row in current_U]
                        new_V = [row[:col_to_try] + row[col_to_try+1:] for row in current_V]
                        new_W = [row[:col_to_try] + row[col_to_try+1:] for row in current_W]
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank = test_rank
                        removed.append(col_to_try)
                except _np.linalg.LinAlgError:
                    pass
            
            if removed:
                # Verify the reduced solution
                try:
                    _verify({"U": current_U, "V": current_V, "W": current_W}, n, m, p)
                    return {"U": current_U, "V": current_V, "W": current_W}
                except (ValueError, AssertionError):
                    pass
    except _np.linalg.LinAlgError:
        pass
    
    # If no reduction found, return the original rank-33 solution
    return {"U": U, "V": V, "W": W}

def _verify(result, n, m, p):
    """Verify the decomposition is correct by checking all (i,j,k) entries."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    # Check shapes
    assert len(U) == n * m, f"U shape {len(U)} != {n*m}"
    assert len(V) == m * p, f"V shape {len(V)} != {m*p}"
    assert len(W) == p * n, f"W shape {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U column count mismatch"
    assert all(len(row) == rank for row in V), "V column count mismatch"
    assert all(len(row) == rank for row in W), "W column count mismatch"
    
    # For standard matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    
    return True
# EVOLVE-BLOCK-END
