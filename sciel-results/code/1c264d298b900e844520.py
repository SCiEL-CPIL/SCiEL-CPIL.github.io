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
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
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

def _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank):
    """Compute the sum of squared errors."""
    sums = np.zeros(len(ta_arr))
    for c in range(rank):
        sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
    return np.sum((sums - 1.0) ** 2)

def _polish(U, V, W, n, m, p, deadline, max_rounds=500):
    """High-precision polish pass on a near-converged solution."""
    rank = U.shape[1]
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
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
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
        
        if it % 20 == 0:
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break
    
    return best_U, best_V, best_W, best_err

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42, warm_start=None):
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
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
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
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                pU, pV, pW, perr = _polish(U, V, W, n, m, p, deadline, max_rounds=300)
                if perr < 1e-12:
                    rU = _round_to_half_integers(pU)
                    rV = _round_to_half_integers(pV)
                    rW = _round_to_half_integers(pW)
                    if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                        return rU.tolist(), rV.tolist(), rW.tolist()
                return pU.tolist(), pV.tolist(), pW.tolist()
    
    if best_err < 1e-8:
        pU, pV, pW, perr = _polish(best_U, best_V, best_W, n, m, p, deadline, max_rounds=500)
        if perr < 1e-12:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
        return pU.tolist(), pV.tolist(), pW.tolist()
    
    return None

def _make_warm_start_for_rank38(U39, V39, W39):
    """Create a warm start for rank 38 by projecting rank-39 solution."""
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    n, m, p = 2, 2, 11
    rank = 39
    
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
    
    best_col = 0
    best_err = float('inf')
    
    candidates = list(range(rank))
    candidates.sort(key=lambda c: 0 if c >= 35 else 1)
    
    for c in candidates[:12]:
        U_new = np.delete(U39_arr, c, axis=1)
        V_new = np.delete(V39_arr, c, axis=1)
        W_new = np.delete(W39_arr, c, axis=1)
        
        sums = np.zeros(len(ta_arr))
        for cc in range(rank - 1):
            sums += U_new[ta_arr, cc] * V_new[tb_arr, cc] * W_new[tc_arr, cc]
        err = np.sum((sums - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_col = c
    
    U38 = np.delete(U39_arr, best_col, axis=1)
    V38 = np.delete(V39_arr, best_col, axis=1)
    W38 = np.delete(W39_arr, best_col, axis=1)
    
    return U38, V38, W38

def _sparse_support_search(rank, n, m, p, deadline, max_attempts=50):
    """
    Try to find a low-rank decomposition using a sparse support approach.
    """
    U39, V39, W39 = _construct_rank39()
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    
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
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        c1 = attempt % 39
        c2 = (attempt + 1) % 39
        
        if c1 == c2:
            continue
        
        U_new = U39_arr.copy()
        V_new = V39_arr.copy()
        W_new = W39_arr.copy()
        
        U_new[:, c1] += U_new[:, c2]
        V_new[:, c1] += V_new[:, c2]
        W_new[:, c1] += W_new[:, c2]
        
        if c2 > c1:
            U_new = np.delete(U_new, c2, axis=1)
            V_new = np.delete(V_new, c2, axis=1)
            W_new = np.delete(W_new, c2, axis=1)
        else:
            U_new = np.delete(U_new, c1, axis=1)
            V_new = np.delete(V_new, c1, axis=1)
            W_new = np.delete(W_new, c1, axis=1)
        
        pU, pV, pW, perr = _polish(U_new, V_new, W_new, n, m, p, deadline, max_rounds=100)
        
        if perr < 1e-10:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
    
    return None

def _try_perturbed_search(rank, n, m, p, deadline, max_rounds=1500, seed=42):
    """Try low rank with perturbations to escape local minima."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
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
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    stall_count = 0
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
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
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank)
            if err < best_err - 1e-12:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > 20:
                # Perturb to escape local minimum
                scale = 0.1 / (1 + stall_count * 0.1)
                U += rng.randn(*U.shape) * scale
                V += rng.randn(*V.shape) * scale
                W += rng.randn(*W.shape) * scale
                stall_count = 0
            
            if err < 1e-10:
                pU, pV, pW, perr = _polish(U, V, W, n, m, p, deadline, max_rounds=200)
                if perr < 1e-12:
                    rU = _round_to_half_integers(pU)
                    rV = _round_to_half_integers(pV)
                    rW = _round_to_half_integers(pW)
                    if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                        return rU.tolist(), rV.tolist(), rW.tolist()
                return pU.tolist(), pV.tolist(), pW.tolist()
    
    if best_err < 1e-8:
        pU, pV, pW, perr = _polish(best_U, best_V, best_W, n, m, p, deadline, max_rounds=500)
        if perr < 1e-12:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
        return pU.tolist(), pV.tolist(), pW.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 10:
            # Try rank 38 first with warm start from rank 39
            if time.time() < deadline - 20:
                U38_ws, V38_ws, W38_ws = _make_warm_start_for_rank38(U, V, W)
                result = _try_low_rank(38, n, m, p, deadline=deadline - 15, max_rounds=3000, seed=0, warm_start=(U38_ws, V38_ws, W38_ws))
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Try sparse support search (column removal + polish)
            if time.time() < deadline - 12:
                result = _sparse_support_search(38, n, m, p, deadline=deadline - 10, max_attempts=30)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Try perturbed search for rank 38
            if time.time() < deadline - 8:
                for seed in range(5):
                    if time.time() > deadline - 5:
                        break
                    result = _try_perturbed_search(38, n, m, p, deadline=deadline - 4, max_rounds=1000, seed=seed)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
            
            # Try ranks with random initialization
            for rank in [38, 37]:
                if time.time() > deadline - 5:
                    break
                seeds = list(range(rank, rank + 4))
                for seed in seeds:
                    if time.time() > deadline - 3:
                        break
                    result = _try_low_rank(rank, n, m, p, deadline=deadline - 2, max_rounds=1200, seed=seed)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": W, "W": W}
    
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
# EVOLVE-BLOCK-END
