"""Hybrid 4x4x4 Strassen (rank 49) + 2x2 Strassen blocks for 4x4*4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    Convention: a = i*2+j, b = j*2+k, c = k*2+i for 2x2 matrices."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)*(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)*B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00*(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11*(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)*B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)*(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)*(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
    return U, V, W

def _strassen_4x4():
    """Return 49-term recursive Strassen decomposition for 4x4 matrix multiplication.
    Convention: a = i*4+j, b = j*4+k, c = k*4+i for 4x4 matrices.
    Built by applying Strassen at block level (2x2 blocks of 2x2),
    each block product uses 2x2 Strassen (rank 7), giving 7*7=49 terms."""
    U1, V1, W1 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    strassen_defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),   # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),             # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)]),     # M7
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        base_col = s * 7
        for sub_r in range(7):
            col = base_col + sub_r
            for (br, bc, coeff) in a_blocks:
                for li in range(2):
                    for lj in range(2):
                        row = 2*br + li
                        col_a = 2*bc + lj
                        a_idx = row * 4 + col_a
                        local_idx = li * 2 + lj
                        U[a_idx, col] += U1[local_idx, sub_r] * coeff
            for (br, bc, coeff) in b_blocks:
                for li in range(2):
                    for lj in range(2):
                        row = 2*br + li
                        col_b = 2*bc + lj
                        b_idx = row * 4 + col_b
                        local_idx = li * 2 + lj
                        V[b_idx, col] += V1[local_idx, sub_r] * coeff
            for (br, bc, coeff) in c_blocks:
                for li in range(2):
                    for lj in range(2):
                        row = 2*br + li
                        col_c = 2*bc + lj
                        c_idx = row * 4 + col_c
                        local_idx = li * 2 + lj
                        W[c_idx, col] += W1[local_idx, sub_r] * coeff
    
    return U, V, W

def _hybrid_4x4_4x6():
    """Hybrid: 4x4 Strassen for cols 0-3 (rank 49), 2x2 Strassen blocks for cols 4-5 (rank 28).
    Total rank: 77."""
    n, m, p = 4, 4, 6
    U4, V4, W4 = _strassen_4x4()
    U2, V2, W2 = _strassen_2x2()
    
    total_rank = 49 + 28
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    # Part 1: 4x4 * 4x4 for columns 0-3
    U[:, 0:49] = U4
    for r in range(16):
        b_row = r // 4
        b_col = r % 4
        V[b_row * 6 + b_col, 0:49] = V4[r, :]
    for r in range(16):
        c_row = r // 4
        c_col = r % 4
        W[c_row * 6 + c_col, 0:49] = W4[r, :]
    
    # Part 2: 4x4 * 4x2 for columns 4-5
    # Split into two 2-row blocks of B[:, 4:6], each 2x2
    # C[:, 4:6] = A[:, 0:2] * B_top + A[:, 2:4] * B_bottom
    # where B_top is rows 0-1 of B[:, 4:6] (2x2), B_bottom is rows 2-3 (2x2)
    # For each C block row ci in {0,1}:
    #   C[2*ci:2*ci+2, 4:6] = A[2*ci:2*ci+2, 0:2] * B_top + A[2*ci:2*ci+2, 2:4] * B_bottom
    # Each product is 2x2 * 2x2, using Strassen (rank 7)
    
    for ci in range(2):
        base_col = 49 + ci * 14
        for local_r in range(14):
            col = base_col + local_r
            if local_r < 7:
                # A[2*ci:2*ci+2, 0:2] * B_top (rows 0-1, cols 4-5)
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + lj
                        U[a_idx, col] += U2[li*2+lj, local_r]
                for li in range(2):
                    for lj in range(2):
                        b_idx = li * 6 + (4 + lj)
                        V[b_idx, col] += V2[li*2+lj, local_r]
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (4 + lj)
                        W[c_idx, col] += W2[li*2+lj, local_r]
            else:
                # A[2*ci:2*ci+2, 2:4] * B_bottom (rows 2-3, cols 4-5)
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + (2 + lj)
                        U[a_idx, col] += U2[li*2+lj, local_r - 7]
                for li in range(2):
                    for lj in range(2):
                        b_idx = (2 + li) * 6 + (4 + lj)
                        V[b_idx, col] += V2[li*2+lj, local_r - 7]
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (4 + lj)
                        W[c_idx, col] += W2[li*2+lj, local_r - 7]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True

def _to_list(mat):
    """Convert numpy array to list of lists with half-integer rounding."""
    result = []
    for row in mat:
        r = []
        for val in row:
            rounded = round(val * 2) / 2
            r.append(rounded)
        result.append(r)
    return result

def _build_expected_tensor(n, m, p):
    """Build the expected matrix multiplication tensor as a flat array."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from decomposition."""
    R = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return T

def _try_remove_column(U, V, W, col_idx, n, m, p, tol=1e-9):
    """Try to remove column col_idx and repair the rest using coordinate descent.
    Returns (success, U_new, V_new, W_new)."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    T_expected = _build_expected_tensor(n, m, p)
    
    # Compute current tensor
    T_current = _compute_tensor(U, V, W, n, m, p)
    
    # Compute contribution of column col_idx
    T_col = np.einsum('i,j,k->ijk', U[:,col_idx], V[:,col_idx], W[:,col_idx])
    
    # Residual if we remove this column: T_expected - (T_current - T_col)
    # = T_expected - T_current + T_col
    # But we want: can the remaining columns (without col_idx) represent T_expected?
    # i.e., is T_col in the span of the other rank-1 tensors?
    # More practically: zero out col_idx and see if we can fix the error.
    
    # Zero out the column
    U[:, col_idx] = 0
    V[:, col_idx] = 0
    W[:, col_idx] = 0
    
    # Compute error
    T_new = _compute_tensor(U, V, W, n, m, p)
    error = np.max(np.abs(T_new - T_expected))
    
    if error < tol:
        return True, U, V, W
    
    # Try coordinate descent to repair: adjust other columns to compensate
    # The deficit is T_col (what we removed). We need to distribute it across other columns.
    # For each other column r, we can adjust U[:,r], V[:,r], W[:,r] slightly.
    # This is complex; instead, try a simpler approach: check if col_idx is linearly
    # dependent on others in a tensor sense.
    
    # Alternative: try swapping - if two columns are similar, merge them.
    # For now, just return failure if simple removal doesn't work.
    return False, U, V, W

def _rank_reduction_search(U, V, W, n, m, p, deadline, max_attempts=200):
    """Search for rank reduction by trying to remove columns."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    tol = 1e-9
    
    # First verify the starting decomposition
    T_expected = _build_expected_tensor(n, m, p)
    
    # Try removing each column
    removed = set()
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Find a column to try removing
        best_col = None
        best_error = float('inf')
        
        for r in range(R):
            if r in removed:
                continue
            
            # Compute contribution of this column
            U_r = U[:, r]
            V_r = V[:, r]
            W_r = W[:, r]
            
            if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
                best_col = r
                break
            
            # Estimate: if this column has small norm, it might be removable
            norm = np.linalg.norm(U_r) * np.linalg.norm(V_r) * np.linalg.norm(W_r)
            if norm < best_error:
                best_error = norm
                best_col = r
        
        if best_col is None:
            break
        
        # Try removing best_col
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, best_col] = 0
        V_test[:, best_col] = 0
        W_test[:, best_col] = 0
        
        T_test = _compute_tensor(U_test, V_test, W_test, n, m, p)
        err = np.max(np.abs(T_test - T_expected))
        
        if err < tol:
            # Successfully removed
            U = U_test
            V = V_test
            W = W_test
            removed.add(best_col)
        else:
            # Try coordinate descent repair
            # Adjust other columns to compensate
            success, U_new, V_new, W_new = _try_remove_column(U, V, W, best_col, n, m, p, tol)
            if success:
                U = U_new
                V = V_new
                W = W_new
                removed.add(best_col)
            else:
                # Skip this column for now
                removed.add(best_col)  # Mark as tried
                if len(removed) >= R:
                    break
    
    return U, V, W

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try hybrid with rank 77
    U, V, W = _hybrid_4x4_4x6()
    
    if _verify_decomposition(U, V, W, n, m, p):
        # Try rank reduction
        U, V, W = _rank_reduction_search(U, V, W, n, m, p, deadline)
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
