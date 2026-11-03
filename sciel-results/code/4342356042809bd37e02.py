"""Matrix multiplication 2x4x6 rank minimization via Strassen block decomposition with rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _construct_strassen_2x2x4(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2x2 blocks.
    
    Strategy: Split the 2x4 * 4x6 multiplication into blocks:
    - A is split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    - B is split into top (rows 0,1) and bottom (rows 2,3), each with 3 column blocks of width 2
    
    So C = A_left * B_top + A_right * B_bottom
    
    Each (2x2) * (2x2) block multiplication uses 7 Strassen terms.
    Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42 terms.
    """
    strassen_terms = _strassen_2x2x2_terms()
    
    rank = 6 * 7  # 42
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                # Update U: A block a_block has columns at offset a_block*2
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                
                # Update V: B rows depend on a_block (top if a_block=0, bottom if a_block=1)
                # B col block b_col_block has columns at offset b_col_block*2
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                
                # Update W: output C[i, k] where i is row of C, k is col of C
                # W index: k*n + i (official trace convention)
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    return U, V, W, rank

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _try_rank_reduction(U, V, W, T, max_iter=1000, deadline=None):
    """
    Try to reduce the rank by finding linear dependencies among the terms.
    
    We check if any column can be expressed as a linear combination of others
    in the tensor sense. This is done by checking if the rank-1 tensors are
    linearly independent.
    
    Returns: (U_reduced, V_reduced, W_reduced, rank_reduced) or None if no reduction found.
    """
    current_rank = U.shape[1]
    if current_rank <= 1:
        return None
    
    # Check for zero columns first
    nonzero_cols = []
    for c in range(current_rank):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            nonzero_cols.append(c)
    
    if len(nonzero_cols) < current_rank:
        U = U[:, nonzero_cols]
        V = V[:, nonzero_cols]
        W = W[:, nonzero_cols]
        current_rank = len(nonzero_cols)
    
    if current_rank <= 1:
        return U, V, W, current_rank
    
    # Try to find linear dependencies by checking if removing a column
    # still allows reconstruction
    # This is expensive, so we use a heuristic: check if any two columns
    # are proportional in a tensor sense
    
    # For each pair of columns, check if they are "equivalent" in the tensor sense
    # Two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 are linearly dependent
    # if there exists scalar λ such that u2=λu1, v2=λv1, w2=λw1
    
    # A more practical approach: try removing one column at a time and see
    # if the remaining columns can still span the tensor
    
    # Since this is expensive, we'll use a simpler heuristic:
    # Check if any column is a scalar multiple of another
    
    removed = []
    kept = list(range(current_rank))
    
    for i in range(len(kept)):
        if time.time() > deadline - 10:
            break
        
        ci = kept[i]
        # Check if column ci is a scalar multiple of any other kept column
        is_redundant = False
        for j in range(len(kept)):
            if j == i:
                continue
            cj = kept[j]
            # Check if U[:,ci] is proportional to U[:,cj]
            if np.any(U[:, cj] != 0):
                ratio_u = U[:, ci] / U[:, cj]
                mask_u = U[:, cj] != 0
                if np.allclose(ratio_u[mask_u], ratio_u[mask_u][0], atol=1e-10):
                    # Check V
                    if np.any(V[:, cj] != 0):
                        ratio_v = V[:, ci] / V[:, cj]
                        mask_v = V[:, cj] != 0
                        if np.allclose(ratio_v[mask_v], ratio_v[mask_v][0], atol=1e-10):
                            # Check W
                            if np.any(W[:, cj] != 0):
                                ratio_w = W[:, ci] / W[:, cj]
                                mask_w = W[:, cj] != 0
                                if np.allclose(ratio_w[mask_w], ratio_w[mask_w][0], atol=1e-10):
                                    # Check if ratios are consistent
                                    ru = ratio_u[mask_u][0]
                                    rv = ratio_v[mask_v][0]
                                    rw = ratio_w[mask_w][0]
                                    if np.isclose(ru, rv, atol=1e-10) and np.isclose(rv, rw, atol=1e-10):
                                        is_redundant = True
                                        break
        
        if is_redundant:
            removed.append(ci)
            kept.remove(ci)
    
    if removed:
        U = U[:, kept]
        V = V[:, kept]
        W = W[:, kept]
        new_rank = len(kept)
        
        # Verify
        if _verify(U, V, W, T):
            return U, V, W, new_rank
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Try Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_2x2x4(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Try rank reduction
    if time.time() < deadline - 20:
        reduced = _try_rank_reduction(U, V, W, T, deadline=deadline)
        if reduced is not None:
            U_r, V_r, W_r, rank_r = reduced
            if rank_r < best_rank:
                best_rank = rank_r
                best_U, best_V, best_W = U_r, V_r, W_r

    # Final verification
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
