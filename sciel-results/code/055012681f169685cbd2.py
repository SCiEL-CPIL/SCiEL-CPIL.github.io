"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
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
    """Verify that the decomposition is exact using Fractions."""
    rank = len(U[0])
    # Convert to Fractions for exact comparison
    U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
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

def _polish_to_rational(U, V, W, n, m, p, deadline):
    """
    Try to polish a float solution to exact half-integer/rational entries.
    Uses rounding to nearest 1/2 and verifies.
    """
    # Try rounding to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify with tolerance
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
    
    sums = np.zeros(len(ta_arr))
    rank = U.shape[1]
    for c in range(rank):
        sums += U_r[ta_arr, c] * V_r[tb_arr, c] * W_r[tc_arr, c]
    
    if np.max(np.abs(sums - 1.0)) < 1e-6:
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try rounding to nearest integer
    U_r2 = np.round(U)
    V_r2 = np.round(V)
    W_r2 = np.round(W)
    
    sums2 = np.zeros(len(ta_arr))
    for c in range(rank):
        sums2 += U_r2[ta_arr, c] * V_r2[tb_arr, c] * W_r2[tc_arr, c]
    
    if np.max(np.abs(sums2 - 1.0)) < 1e-6:
        return U_r2.tolist(), V_r2.tolist(), W_r2.tolist()
    
    return None

def _try_low_rank(rank, n, m, p, deadline, max_rounds=5000, seed=42, init=None):
    """
    Fast vectorized iterative refinement for low-rank decomposition.
    """
    rng = np.random.RandomState(seed)
    
    if init is not None:
        U, V, W = init
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
    else:
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
    
    # Precompute index arrays
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
    n_terms = len(ta_arr)
    
    # Precompute for each factor index, which terms it participates in
    # For U[a]: terms where ta_arr == a
    # For V[b]: terms where tb_arr == b  
    # For W[c]: terms where tc_arr == c
    
    # Group term indices by factor
    terms_by_a = {a: np.where(ta_arr == a)[0] for a in range(n * m)}
    terms_by_b = {b: np.where(tb_arr == b)[0] for b in range(m * p)}
    terms_by_c = {c: np.where(tc_arr == c)[0] for c in range(p * n)}
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    check_interval = 10
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U: for each a, U[a,c] = sum_t V[tb_t,c]*W[tc_t,c] / sum_t (V[tb_t,c]*W[tc_t,c])^2
        for a in range(n * m):
            idx = terms_by_a[a]
            if len(idx) == 0:
                continue
            V_sub = V[tb_arr[idx], :]  # (len(idx), rank)
            W_sub = W[tc_arr[idx], :]  # (len(idx), rank)
            prod = V_sub * W_sub        # (len(idx), rank)
            num = prod.sum(axis=0)      # (rank,)
            den = (prod ** 2).sum(axis=0)  # (rank,)
            valid = den > 1e-14
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            idx = terms_by_b[b]
            if len(idx) == 0:
                continue
            U_sub = U[ta_arr[idx], :]
            W_sub = W[tc_arr[idx], :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-14
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            idx = terms_by_c[c_idx]
            if len(idx) == 0:
                continue
            U_sub = U[ta_arr[idx], :]
            V_sub = V[tb_arr[idx], :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-14
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence
        if it % check_interval == 0:
            sums = np.zeros(n_terms)
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = float(np.sum((sums - 1.0) ** 2))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
    
    if best_err < 1e-10 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
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
    
    # Try to find a lower rank decomposition with remaining time
    # Use SVD to check if rank 39 can be reduced
    U39_np = np.array(U39, dtype=float)
    V39_np = np.array(V39, dtype=float)
    W39_np = np.array(W39, dtype=float)
    
    # Check numerical rank via SVD of the tensor unfolding
    # If some columns are linearly dependent, we might reduce rank
    # For now, just try direct low-rank search
    
    best_result = {"U": U39, "V": V39, "W": W39}
    best_rank = 39
    
    # Try ranks from 38 down, with multiple seeds
    for target_rank in [38, 37, 36, 35, 34, 33, 32]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        num_seeds = 20 if target_rank >= 36 else 15
        seeds = list(range(target_rank * 7, target_rank * 7 + num_seeds))
        
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            # For rank 38, try warm start from rank 39
            init = None
            if target_rank == 38:
                # Keep first 38 columns
                init = (U39_np[:, :38].copy(), V39_np[:, :38].copy(), W39_np[:, :38].copy())
            
            result = _try_low_rank(target_rank, n, m, p, 
                                   deadline=deadline - 3, 
                                   max_rounds=3000, 
                                   seed=seed, 
                                   init=init)
            
            if result is not None:
                U2, V2, W2 = result
                # Try to polish to exact rationals
                polished = _polish_to_rational(np.array(U2), np.array(V2), np.array(W2), n, m, p, deadline)
                if polished is not None:
                    U3, V3, W3 = polished
                    if _verify(U3, V3, W3, n, m, p):
                        return {"U": U3, "V": V3, "W": W3}
                # Check if float solution is exact enough
                if _verify(U2, V2, W2, n, m, p, tol=1e-9):
                    return {"U": U2, "V": V2, "W": W2}
    
    return best_result
# EVOLVE-BLOCK-END
