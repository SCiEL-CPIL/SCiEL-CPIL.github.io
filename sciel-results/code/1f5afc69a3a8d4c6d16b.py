import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    _deadline = _start_time + 110  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Strategy: Try rank 27 by exploiting structure of 2x4x4 tensor.
    # The 2x4x4 tensor has a specific sparsity that may allow one less rank-1 term
    # than the naive 4x-Strassen (rank 28) construction.
    
    # First, build the rank-28 Strassen decomposition as baseline
    rank28 = _build_strassen_28(n, m, p)
    U28, V28, W28 = rank28
    
    # Verify rank-28 is correct
    if not _verify_decomp(U28, V28, W28, n, m, p):
        # Fallback to naive
        U_n, V_n, W_n = _build_naive(n, m, p)
        return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}
    
    # Attempt rank-27: try to express one column as linear combination of others
    # and verify if the tensor equality still holds.
    # We'll try each of the 28 columns as a candidate to eliminate.
    # For column c to be eliminated, we need:
    #   U[:,c] = sum_{c' != c} alpha_{c'} U[:,c']
    #   V[:,c] = sum_{c' != c} beta_{c'} V[:,c']
    #   W[:,c] = sum_{c' != c} gamma_{c'} W[:,c']
    # such that the tensor contribution of column c equals the weighted sum.
    # This is a strong condition. Let's try a different approach:
    # Use the fact that for 2x4x4, we can use a mixed approach.
    
    # Alternative: Use Strassen for the "main" 2x2x2 blocks but handle
    # the rectangular structure more cleverly.
    
    # Try: 2x4x4 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x2) - one shared term
    # The key insight: the M1 terms in diagonal blocks may be shareable.
    
    # Let's try a specific rank-27 construction:
    # Use 6 Strassen terms for 3 of the blocks, and 5 for one block
    # by sharing a term.
    
    # Actually, let's try a more systematic approach:
    # Build the tensor and use linear algebra to find if rank 27 is possible.
    
    # The tensor T has shape (8, 16, 8) where:
    # U-index: i*m + j (i in [0,2), j in [0,4))
    # V-index: j*p + k (j in [0,4), k in [0,4))
    # W-index: k*n + i (k in [0,4), i in [0,2))
    # T[u,v,w] = 1 if u,j matches and v,j matches and w,i matches appropriately
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the full tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Now, we have a rank-28 decomposition. Try to see if we can reduce to 27.
    # Use the method of "tensor rank reduction" by checking if one slice
    # is in the span of others.
    
    # For each candidate column to eliminate, check if the corresponding
    # rank-1 tensor is in the span of the others.
    
    U28_f = U28.astype(float)
    V28_f = V28.astype(float)
    W28_f = W28.astype(float)
    
    best_result = {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    best_rank = 28
    
    # Try to eliminate each column
    for elim_col in range(28):
        if time.time() > _deadline:
            break
        
        # Check if column elim_col can be expressed as combination of others
        # This requires solving a system. Due to complexity, use a heuristic:
        # Check if the outer product u_c ⊗ v_c ⊗ w_c is in the span
        # of {u_{c'} ⊗ v_{c'} ⊗ w_{c'} for c' != c}
        
        # This is a 8*16*8 = 1024-dimensional space.
        # Build the matrix of other terms and check if target is in span.
        
        target = np.zeros(n*m*m*p*p*n)
        target[(U28_f[:, elim_col], V28_f[:, elim_col], W28_f[:, elim_col])] = 1.0
        # Actually, the rank-1 tensor is the outer product
        u_c = U28_f[:, elim_col]
        v_c = V28_f[:, elim_col]
        w_c = W28_f[:, elim_col]
        target = np.einsum('i,j,k->ijk', u_c, v_c, w_c)
        target_flat = target.flatten()
        
        # Build matrix of other rank-1 tensors
        others = []
        for c in range(28):
            if c == elim_col:
                continue
            u_o = U28_f[:, c]
            v_o = V28_f[:, c]
            w_o = W28_f[:, c]
            t_o = np.einsum('i,j,k->ijk', u_o, v_o, w_o)
            others.append(t_o.flatten())
        
        others_mat = np.column_stack(others)
        
        # Check if target_flat is in the column space of others_mat
        # Use least squares to check residual
        try:
            coeffs, residuals, rank_o, sv = np.linalg.lstsq(others_mat, target_flat, rcond=None)
            pred = others_mat @ coeffs
            residual = np.linalg.norm(target_flat - pred)
            norm_target = np.linalg.norm(target_flat)
            
            if residual < 1e-10 * norm_target:
                # Found a valid rank-27 decomposition!
                # Remove column elim_col
                keep_cols = [c for c in range(28) if c != elim_col]
                U27 = U28[:, keep_cols]
                V27 = V28[:, keep_cols]
                W27 = W28[:, keep_cols]
                
                if _verify_decomp(U27, V27, W27, n, m, p):
                    best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                    best_rank = 27
                    break  # Found rank 27, stop searching
        except Exception:
            continue
    
    # If we didn't find rank 27, try a different structural approach
    if best_rank == 28 and time.time() < _deadline - 30:
        # Try: use a hybrid decomposition
        # For 2x4x4, we can think of it as 2 independent 1x4x4 multiplications
        # or 4 independent 2x1x4 multiplications.
        
        # Another approach: Use the fact that 2x4x4 can be embedded in 4x4x4
        # and use known 4x4x4 results, but that's overkill.
        
        # Try: direct rank-27 construction using integer linear programming
        # with a small search space.
        
        # Given time constraints, return the best we have
        pass
    
    return best_result


def _build_strassen_28(n, m, p):
    """Build the standard rank-28 Strassen decomposition for 2x4x4."""
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2: P = XY
    # M1 = (X00+X11)(Y00+Y11) -> P00+=1, P11+=1
    # M2 = (X10+X11)Y00 -> P10+=1, P11-=1
    # M3 = X00(Y01-Y11) -> P01+=1, P11+=1
    # M4 = X11(Y10-Y00) -> P00+=1, P10+=1
    # M5 = (X00+X01)Y11 -> P00-=1, P01+=1
    # M6 = (X10-X00)(Y00+Y01) -> P11+=1
    # M7 = (X01-X11)(Y10+Y11) -> P00+=1
    
    strassen_terms = [
        # (u_local, v_local, w_local)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # M7
    ]
    
    def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off * 2 + r) * p + (b_col_off * 2 + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off * 2 + c) * n + r
                W[g_idx, col] += coeff
    
    # 4 blocks: (A1,B11), (A2,B21), (A1,B12), (A2,B22)
    apply_block(0, 0, 0, 0, 0)   # A cols 0-1, B rows 0-1 cols 0-1 -> C cols 0-1
    apply_block(1, 1, 0, 0, 7)   # A cols 2-3, B rows 2-3 cols 0-1 -> C cols 0-1
    apply_block(0, 0, 1, 1, 14)  # A cols 0-1, B rows 0-1 cols 2-3 -> C cols 2-3
    apply_block(1, 1, 1, 1, 21)  # A cols 2-3, B rows 2-3 cols 2-3 -> C cols 2-3
    
    return U, V, W


def _build_naive(n, m, p):
    """Build naive rank-32 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    U_f = U.astype(float)
    V_f = V.astype(float)
    W_f = W.astype(float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    
    for i in range(n):
        for k in range(p):
            acc = 0.0
            for c in range(rank):
                for j in range(m):
                    acc += U_f[i * m + j, c] * V_f[j * p + k, c] * W_f[k * n + i, c]
            C_computed[i, k] = acc
    
    return np.allclose(C_expected, C_computed, atol=1e-10)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
