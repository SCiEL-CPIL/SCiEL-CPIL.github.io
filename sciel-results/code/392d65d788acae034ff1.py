import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Compute via: for each r, outer product U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Use einsum for speed
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol

def _verify_single(U, V, W, n, m, p, tol=1e-9):
    """Verify a single candidate, returns (ok, max_err)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _snap_to_half_integer(mat, tol=1e-9):
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        out = []
        for row in snapped:
            out.append([float(x) for x in row])
        return out
    return [list(map(float, row)) for row in arr]

def _build_strassen_44(n, m, p):
    """Construct the rank-44 decomposition: Strassen on 2x4 blocks + schoolbook for row 2."""
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2 terms
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 4 blocks of 2x2 Strassen for the 2x4 * 4x4 part
    # Blocks cover: A[0:2, j:j+2] * B[j:j+2, k:k+2] -> C[0:2, k:k+2]
    blocks = [
        (0, 0, 0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 0, 2),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 0, 2),  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]
    
    col = 0
    for (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off) in blocks:
        for (x_coeff, y_coeff, z_coeff) in strassen_steps:
            for (xi, xj), val in x_coeff.items():
                U[(a_row_off + xi) * m + (a_col_off + xj)][col] += val
            for (yi, yj), val in y_coeff.items():
                V[(b_row_off + yi) * p + (b_col_off + yj)][col] += val
            for (zi, zj), val in z_coeff.items():
                W[(c_col_off + zj) * n + (c_row_off + zi)][col] += val
            col += 1
    
    # Schoolbook for row 2: A[2,j] * B[j,k] -> C[2,k]
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 2][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _try_column_elimination(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among columns to reduce rank.
    
    For each column c, check if it can be expressed as a linear combination
    of other columns. If so, we can eliminate it.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    R = U_arr.shape[1]
    
    best_rank = R
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try eliminating columns one at a time
    # For column c, we need to find coefficients alpha_r such that:
    # U[:,c] = sum_r alpha_r * U[:,r]  (for r != c)
    # V[:,c] = sum_r alpha_r * V[:,r]
    # W[:,c] = sum_r alpha_r * W[:,r]
    
    # This is a linear system. Let's try each column.
    eliminated = set()
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        current_R = best_U.shape[1]
        
        for c in range(current_R):
            if time.time() > deadline - 5:
                break
            if c in eliminated:
                continue
            
            # Build the linear system: find alpha such that
            # U[:,c] = U[:,others] @ alpha
            # V[:,c] = V[:,others] @ alpha
            # W[:,c] = W[:,others] @ alpha
            
            others = [r for r in range(current_R) if r != c and r not in eliminated]
            if len(others) < 1:
                continue
            
            U_others = best_U[:, others]
            V_others = best_V[:, others]
            W_others = best_W[:, others]
            
            U_c = best_U[:, c]
            V_c = best_V[:, c]
            W_c = best_W[:, c]
            
            # Stack all three: [U_c; V_c; W_c] = [U_others; V_others; W_others] @ alpha
            A_sys = np.vstack([U_others, V_others, W_others])
            b_sys = np.concatenate([U_c, V_c, W_c])
            
            try:
                alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-10)
            except:
                continue
            
            # Check if the fit is exact
            residual = np.linalg.norm(A_sys @ alpha - b_sys)
            if residual < 1e-9:
                # Column c is redundant! Remove it.
                keep = [r for r in range(current_R) if r != c and r not in eliminated]
                best_U = best_U[:, keep]
                best_V = best_V[:, keep]
                best_W = best_W[:, keep]
                eliminated.add(c)
                changed = True
                best_rank = best_U.shape[1]
                break
    
    if best_rank < R:
        return best_U, best_V, best_W
    return None

def _try_structured_reduction(U, V, W, n, m, p, deadline):
    """Try structured reductions by checking if certain column patterns allow merging."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    R = U_arr.shape[1]
    
    # Try: can we merge pairs of columns that have similar structure?
    # For each pair (c1, c2), check if U[:,c1]+U[:,c2], etc., still works
    # This is equivalent to checking if the sum of two rank-1 tensors is also valid
    
    # Actually, a better approach: try random linear combinations
    # Replace two columns with their sum and difference, check if rank reduces
    
    best_result = None
    best_rank = R
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Pick two random columns to try merging
        c1 = np.random.randint(0, R)
        c2 = np.random.randint(0, R)
        if c1 == c2:
            continue
        
        # Try: new_col = col1 + col2, remove col2
        # Or: new_col = col1 - col2, remove col2
        for sign in [1.0, -1.0]:
            U_try = U_arr.copy()
            V_try = V_arr.copy()
            W_try = W_arr.copy()
            
            U_try[:, c1] = U_arr[:, c1] + sign * U_arr[:, c2]
            V_try[:, c1] = V_arr[:, c1] + sign * V_arr[:, c2]
            W_try[:, c1] = W_arr[:, c1] + sign * W_arr[:, c2]
            
            # Remove column c2
            keep = [r for r in range(R) if r != c2]
            U_try = U_try[:, keep]
            V_try = V_try[:, keep]
            W_try = W_try[:, keep]
            
            ok, err = _verify_single(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p)
            if ok and U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_result = (U_try, V_try, W_try)
    
    if best_result is not None:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Start with the rank-44 Strassen decomposition
    result = _build_strassen_44(n, m, p)
    
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
    
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    best_rank = U_arr.shape[1]
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    
    # Attempt 1: Column elimination via linear dependency detection
    if time.time() < deadline - 30:
        elim_result = _try_column_elimination(
            best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, deadline
        )
        if elim_result is not None:
            eU, eV, eW = elim_result
            if eU.shape[1] < best_rank:
                best_rank = eU.shape[1]
                best_U, best_V, best_W = eU, eV, eW
    
    # Attempt 2: Structured pair merging
    if time.time() < deadline - 20:
        struct_result = _try_structured_reduction(
            best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, deadline
        )
        if struct_result is not None:
            sU, sV, sW = struct_result
            if sU.shape[1] < best_rank:
                best_rank = sU.shape[1]
                best_U, best_V, best_W = sU, sV, sW
    
    # Final: snap to half-integers and verify
    U_final = _snap_to_half_integer(best_U.tolist())
    V_final = _snap_to_half_integer(best_V.tolist())
    W_final = _snap_to_half_integer(best_W.tolist())
    
    if not _verify_fast(U_final, V_final, W_final, n, m, p):
        # Fallback to the unsnapped version if it's better
        if _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            U_final = best_U.tolist()
            V_final = best_V.tolist()
            W_final = best_W.tolist()
        else:
            # Ultimate fallback
            result = _schoolbook(n, m, p)
            return result
    
    return {"U": U_final, "V": V_final, "W": W_final}
