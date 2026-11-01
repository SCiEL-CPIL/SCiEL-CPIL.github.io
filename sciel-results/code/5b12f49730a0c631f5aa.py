"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices: B[j,k] -> V[j*p + k]
        v_e = k0        # B[0, k0]
        v_f = k1        # B[0, k1]
        v_g = p + k0    # B[1, k0]
        v_h = p + k1    # B[1, k1]
        
        # W indices: C[i,k] -> W[k*n + i]
        w_c00 = k0 * n      # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n      # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]
        
        # U indices: A[i,j] -> U[i*m + j]
        # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        
        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook 4 terms
    k2 = 10
    v_e2 = k2        # B[0, 10]
    v_g2 = p + k2    # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]
    
    # A00 * B[0,10] -> C[0,10]
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A01 * B[1,10] -> C[0,10]
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A10 * B[0,10] -> C[1,10]
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # A11 * B[1,10] -> C[1,10]
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _round_to_half_integers(arr, tol=1e-6):
    """Round array entries to nearest half-integer if close enough."""
    result = arr.copy()
    for idx in np.ndindex(result.shape):
        val = result[idx]
        nearest = round(2 * val) / 2.0
        if abs(val - nearest) < tol:
            result[idx] = nearest
    return result

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact arithmetic using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(str(U[i * m + j][c])) * Fraction(str(V[j * p + k][c])) * Fraction(str(W[k * n + i][c]))
                if s != 1:
                    return False
    return True

def _build_index_structures(n, m, p):
    """Precompute index arrays and group masks once."""
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    groups_a = {a: (ta_arr == a) for a in range(n * m)}
    groups_b = {b: (tb_arr == b) for b in range(m * p)}
    groups_c = {c: (tc_arr == c) for c in range(p * n)}
    
    return ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c

def _compute_error(U, V, W, ta_arr, tb_arr, tc_arr):
    """Compute sum of squared errors."""
    rank = U.shape[1]
    sums = np.zeros(len(ta_arr))
    for c in range(rank):
        sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
    return np.sum((sums - 1.0) ** 2)

def _polish(U, V, W, n, m, p, deadline, max_rounds=500, idx_struct=None):
    """High-precision polish pass on a near-converged solution."""
    rank = U.shape[1]
    if idx_struct is None:
        ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = _build_index_structures(n, m, p)
    else:
        ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = idx_struct
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = groups_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = groups_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = groups_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 10 == 0:
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break
    
    return best_U, best_V, best_W, best_err

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42, warm_start=None, idx_struct=None):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    if warm_start is not None:
        U, V, W = warm_start
    else:
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
    
    if idx_struct is None:
        ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = _build_index_structures(n, m, p)
    else:
        ta_arr, tb_arr, tc_arr, groups_a, groups_b, groups_c = idx_struct
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = groups_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = groups_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = groups_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 10 == 0:
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                # Polish and try to round
                pU, pV, pW, perr = _polish(U, V, W, n, m, p, deadline, max_rounds=300, idx_struct=idx_struct)
                if perr < 1e-12:
                    rU = _round_to_half_integers(pU)
                    rV = _round_to_half_integers(pV)
                    rW = _round_to_half_integers(pW)
                    if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                        return rU.tolist(), rV.tolist(), rW.tolist()
                return pU.tolist(), pV.tolist(), pW.tolist()
    
    if best_err < 1e-8:
        pU, pV, pW, perr = _polish(best_U, best_V, best_W, n, m, p, deadline, max_rounds=500, idx_struct=idx_struct)
        if perr < 1e-12:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
        return pU.tolist(), pV.tolist(), pW.tolist()
    
    return None

def _make_warm_start_for_rank38(U39, V39, W39):
    """Create a warm start for rank 38 by trying multiple column removal strategies."""
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    n, m, p = 2, 2, 11
    rank = 39
    
    ta_arr, tb_arr, tc_arr, _, _, _ = _build_index_structures(n, m, p)
    
    best_err = float('inf')
    best_U38, best_V38, best_W38 = None, None, None
    
    # Try removing each column and compute the residual error
    for c in range(rank):
        U_new = np.delete(U39_arr, c, axis=1)
        V_new = np.delete(V39_arr, c, axis=1)
        W_new = np.delete(W39_arr, c, axis=1)
        
        err = _compute_error(U_new, V_new, W_new, ta_arr, tb_arr, tc_arr)
        
        if err < best_err:
            best_err = err
            best_U38, best_V38, best_W38 = U_new, V_new, W_new
    
    # Also try merging pairs of columns (sum them, then remove one)
    # Focus on pairs within the same Strassen block or between Strassen and schoolbook
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            if time.time() > 110:  # Safety
                break
            U_new = U39_arr.copy()
            V_new = V39_arr.copy()
            W_new = W39_arr.copy()
            
            # Merge c2 into c1
            U_new[:, c1] += U_new[:, c2]
            V_new[:, c1] += V_new[:, c2]
            W_new[:, c1] += W_new[:, c2]
            
            # Remove column c2
            U_new = np.delete(U_new, c2, axis=1)
            V_new = np.delete(V_new, c2, axis=1)
            W_new = np.delete(W_new, c2, axis=1)
            
            err = _compute_error(U_new, V_new, W_new, ta_arr, tb_arr, tc_arr)
            
            if err < best_err:
                best_err = err
                best_U38, best_V38, best_W38 = U_new, V_new, W_new
    
    if best_U38 is not None:
        return best_U38, best_V38, best_W38
    
    # Fallback: just remove the last column
    return np.delete(U39_arr, -1, axis=1), np.delete(V39_arr, -1, axis=1), np.delete(W39_arr, -1, axis=1)

def _sparse_support_search(rank, n, m, p, deadline, max_attempts=50):
    """
    Try to find a low-rank decomposition using column removal + polish.
    """
    U39, V39, W39 = _construct_rank39()
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    
    idx_struct = _build_index_structures(n, m, p)
    ta_arr, tb_arr, tc_arr = idx_struct[0], idx_struct[1], idx_struct[2]
    
    # Try removing each column and polishing
    for c in range(rank):
        if time.time() > deadline:
            break
        
        U_new = np.delete(U39_arr, c, axis=1)
        V_new = np.delete(V39_arr, c, axis=1)
        W_new = np.delete(W39_arr, c, axis=1)
        
        # Quick error check before expensive polish
        err = _compute_error(U_new, V_new, W_new, ta_arr, tb_arr, tc_arr)
        if err < 1.0:  # Only polish if error is reasonable
            pU, pV, pW, perr = _polish(U_new, V_new, W_new, n, m, p, deadline, max_rounds=200, idx_struct=idx_struct)
            
            if perr < 1e-10:
                rU = _round_to_half_integers(pU)
                rV = _round_to_half_integers(pV)
                rW = _round_to_half_integers(pW)
                if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                    return rU.tolist(), rV.tolist(), rW.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if not _verify(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Precompute index structures once
    idx_struct = _build_index_structures(n, m, p)
    
    # Try to find a lower rank decomposition with remaining time
    elapsed = time.time() - start_time
    
    # Try rank 38 with warm start from rank 39
    if elapsed < 5:
        U38_ws, V38_ws, W38_ws = _make_warm_start_for_rank38(U, V, W)
        remaining = deadline - time.time()
        if remaining > 10:
            result = _try_low_rank(38, n, m, p, deadline=time.time() + remaining - 5, max_rounds=5000, seed=0, warm_start=(U38_ws, V38_ws, W38_ws), idx_struct=idx_struct)
            if result is not None:
                U2, V2, W2 = result
                if _verify(U2, V2, W2, n, m, p):
                    return {"U": U2, "V": V2, "W": W2}
    
    # Try sparse support search (column removal + polish)
    elapsed = time.time() - start_time
    if elapsed < 20:
        remaining = deadline - time.time()
        if remaining > 10:
            result = _sparse_support_search(38, n, m, p, deadline=time.time() + remaining - 5, max_attempts=39)
            if result is not None:
                U2, V2, W2 = result
                if _verify(U2, V2, W2, n, m, p):
                    return {"U": U2, "V": V2, "W": W2}
    
    # Try rank 37 with warm start (remove 2 columns from rank 39)
    elapsed = time.time() - start_time
    if elapsed < 40:
        U39_arr = np.array(U, dtype=float)
        V39_arr = np.array(V, dtype=float)
        W39_arr = np.array(W, dtype=float)
        ta_arr, tb_arr, tc_arr = idx_struct[0], idx_struct[1], idx_struct[2]
        
        # Find the two columns whose removal gives the best error
        best_err_37 = float('inf')
        best_U37, best_V37, best_W37 = None, None, None
        
        for c1 in range(39):
            for c2 in range(c1 + 1, 39):
                if time.time() > deadline - 10:
                    break
                cols_to_remove = [c1, c2]
                keep = [c for c in range(39) if c not in cols_to_remove]
                U_new = U39_arr[:, keep]
                V_new = V39_arr[:, keep]
                W_new = W39_arr[:, keep]
                
                err = _compute_error(U_new, V_new, W_new, ta_arr, tb_arr, tc_arr)
                if err < best_err_37:
                    best_err_37 = err
                    best_U37, best_V37, best_W37 = U_new, V_new, W_new
        
        if best_U37 is not None and best_err_37 < 10.0:
            remaining = deadline - time.time()
            if remaining > 5:
                result = _try_low_rank(37, n, m, p, deadline=time.time() + remaining - 2, max_rounds=3000, seed=0, warm_start=(best_U37, best_V37, best_W37), idx_struct=idx_struct)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
