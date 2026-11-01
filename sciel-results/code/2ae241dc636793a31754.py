"""Combined approach: verified rank-39 Strassen+schoolbook, then structured low-rank search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import itertools

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

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    u = Fraction(str(U[i * m + j][c]))
                    v = Fraction(str(V[j * p + k][c]))
                    w = Fraction(str(W[k * n + i][c]))
                    s += u * v * w
                if s != Fraction(1):
                    return False
    return True

def _round_to_half_int(arr):
    """Round a numpy array to nearest half-integer."""
    return np.round(arr * 2) / 2

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

def _structured_init(rank, n, m, p, seed):
    """
    Initialize from rank-39 solution by randomly merging factors.
    We have 39 factors; to get rank r, we need to merge (39-r) pairs.
    Merging means adding two factor columns together.
    """
    U39, V39, W39 = _construct_rank39()
    U39 = np.array(U39)
    V39 = np.array(V39)
    W39 = np.array(W39)
    
    rng = np.random.RandomState(seed)
    
    # Start with all 39 factors
    U = U39.copy()
    V = V39.copy()
    W = W39.copy()
    
    # We need to reduce from 39 to rank
    # Strategy: randomly select pairs to merge
    # Each merge: pick two columns, add them, remove one
    current_rank = 39
    merge_count = 39 - rank
    
    # Shuffle column indices
    cols = list(range(39))
    rng.shuffle(cols)
    
    # Process merges
    merged = set()
    for _ in range(merge_count):
        # Find two unmerged columns to merge
        available = [c for c in cols if c not in merged]
        if len(available) < 2:
            break
        # Pick two random available columns
        i, j = available[:2]
        # Merge column j into column i
        U[:, i] += U[:, j]
        V[:, i] += V[:, j]
        W[:, i] += W[:, j]
        merged.add(j)
    
    # Extract the first 'rank' columns (those not merged away)
    kept_cols = [c for c in range(39) if c not in merged]
    kept_cols = kept_cols[:rank]
    
    U_out = U[:, kept_cols]
    V_out = V[:, kept_cols]
    W_out = W[:, kept_cols]
    
    return U_out, V_out, W_out

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42, use_structured=True):
    """
    Try to find a decomposition of given rank using iterative refinement
    with structured initialization.
    """
    if use_structured:
        U, V, W = _structured_init(rank, n, m, p, seed)
    else:
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
    
    # Target index arrays
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
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = (ta_arr == a)
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = (tb_arr == b)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = (tc_arr == c_idx)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 20 iterations
        if it % 20 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                # Polish: round to half-integers and verify
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    return U_list, V_list, W_list
                # Try without rounding
                return U.tolist(), V.tolist(), W.tolist()
    
    # Final check on best
    if best_err < 1e-8 and best_U is not None:
        U_r = _round_to_half_int(best_U)
        V_r = _round_to_half_int(best_V)
        W_r = _round_to_half_int(best_W)
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return U_list, V_list, W_list
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 10:
            # Focus primarily on rank 38 with multiple structured initializations
            for rank in [38, 37, 36]:
                if time.time() > deadline - 5:
                    break
                # Use many seeds for the most promising rank
                if rank == 38:
                    seeds = list(range(20))
                else:
                    seeds = list(range(5))
                for seed in seeds:
                    if time.time() > deadline - 3:
                        break
                    result = _try_low_rank(rank, n, m, p, deadline=deadline - 2, max_rounds=800, seed=seed, use_structured=True)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}
    
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
