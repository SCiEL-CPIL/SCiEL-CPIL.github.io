"""Combined: recursive Strassen baseline (rank 53) + ALS refinement with half-integer rounding."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}})
    terms.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}})
    terms.append({'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}})
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}})
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}})
    return terms


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using recursive Strassen: 7*7 = 49 terms."""
    n = 4
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            for (bi, bj), cu in term1['U'].items():
                for (li, lj), cv in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i * n + j][idx] += cu * cv
            for (bj, bk), cv in term1['V'].items():
                for (lj, lk), cw in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j * n + k][idx] += cv * cw
            for (bk, bi), cw in term1['W'].items():
                for (lk, li), cu2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k * n + i][idx] += cw * cu2
    return {"U": U, "V": V, "W": W}


def _build_4x4x5_strassen():
    """Build 4x4x5 decomposition: recursive Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 4). Total rank 53."""
    n, m, p = DIMENSIONS
    base = _build_4x4x4_strassen()
    rank_base = 49
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(16):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_base):
                V[j * p + k][r] = base["V"][j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_base):
                W[k * n + i][r] = base["W"][k * 4 + i][r]
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i * m + j][r] = 1
        V[j * p + 4][r] = 1
        for i in range(n):
            W[4 * n + i][r] = 1
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Fallback schoolbook decomposition: rank = n*m*p = 80."""
    n, m, p = DIMENSIONS
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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_fast_numpy(U, V, W, n, m, p, atol=1e-9):
    """Fast numpy-based verification."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wc.T
    return bool(np.allclose(T, 1.0, atol=atol))


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _round_to_half_integer(x):
    return round(x * 2) / 2.0


def _round_matrix_to_half_int(M):
    return [[_round_to_half_integer(x) for x in row] for row in M]


def _als_decompose(n, m, p, target_rank, U_init, V_init, W_init, 
                   max_iters=200, tol=1e-8, deadline=None, seed=42):
    """ALS refinement: try to reduce rank by zeroing out small columns and re-solving.
    
    Strategy: Start with a valid decomposition at some rank R. Try to find a valid
    decomposition at rank R' < R by:
    1. Identify columns with small norm
    2. Set them to zero
    3. Re-solve the remaining system via ALS
    4. Round to half-integers
    5. Verify
    """
    rng = random.Random(seed)
    Ua = np.array(U_init, dtype=np.float64)
    Va = np.array(V_init, dtype=np.float64)
    Wa = np.array(W_init, dtype=np.float64)
    R = Ua.shape[1]
    
    if R < target_rank:
        return None
    
    # Try progressively lower ranks
    for target in range(target_rank, min(R, target_rank + 1)):
        if deadline and time.time() > deadline:
            break
        
        # Try removing (R - target) columns
        # Strategy: remove columns with smallest L2 norm across all three matrices
        col_norms = []
        for r in range(R):
            norm = np.linalg.norm(Ua[:, r]) + np.linalg.norm(Va[:, r]) + np.linalg.norm(Wa[:, r])
            col_norms.append((norm, r))
        col_norms.sort()
        
        # Try different subsets to remove
        num_remove = R - target
        removed = set()
        for _, idx in col_norms[:num_remove]:
            removed.add(idx)
        
        kept = [r for r in range(R) if r not in removed]
        if len(kept) < target:
            continue
        
        # Now try ALS on the kept columns to see if we can maintain the decomposition
        U_work = Ua[:, kept].copy()
        V_work = Va[:, kept].copy()
        W_work = Wa[:, kept].copy()
        
        # ALS iterations
        converged = False
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve U[i*m+j, :] = T[i,j,:] @ (V*W)^T
            # T[i,j,k] = 1 for all i,j,k
            # U[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    # Target: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is: U_row @ (V_col * W_col) = 1 for each k
                    # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then U_row @ M = ones(p)
                    M = np.zeros((p, target))
                    for k in range(p):
                        M[k, :] = V_work[j*p+k, :] * W_work[k*n+i, :]
                    # Solve U_row @ M = ones(p)
                    try:
                        U_work[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve V[j*p+k, :] = T[:,j,k] @ (U*W)^T
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target))
                    for i in range(n):
                        M[i, :] = U_work[i*m+j, :] * W_work[k*n+i, :]
                    try:
                        V_work[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve W[k*n+i, :] = T[i,:,k] @ (U*V)^T
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target))
                    for j in range(m):
                        M[j, :] = U_work[i*m+j, :] * V_work[j*p+k, :]
                    try:
                        W_work[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            T_check = np.zeros((n, m, p), dtype=np.float64)
            for r in range(target):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check[i,j,k] += U_work[i*m+j, r] * V_work[j*p+k, r] * W_work[k*n+i, r]
            if np.allclose(T_check, 1.0, atol=tol):
                converged = True
                break
        
        if converged:
            # Round to half-integers
            U_round = [[_round_to_half_integer(x) for x in row] for row in U_work.tolist()]
            V_round = [[_round_to_half_integer(x) for x in row] for row in V_work.tolist()]
            W_round = [[_round_to_half_integer(x) for x in row] for row in W_work.tolist()]
            
            if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                return {"U": U_round, "V": V_round, "W": W_round}
        
        # Also try without rounding
        if converged:
            if _verify_decomposition(U_work.tolist(), V_work.tolist(), W_work.tolist(), n, m, p):
                # Check if entries are close to half-integers
                if _verify_integer_half_integer(U_work.tolist(), V_work.tolist(), W_work.tolist(), tol=1e-4):
                    U_round = _round_matrix_to_half_int(U_work.tolist())
                    V_round = _round_matrix_to_half_int(V_work.tolist())
                    W_round = _round_matrix_to_half_int(W_work.tolist())
                    if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                        return {"U": U_round, "V": V_round, "W": W_round}
    
    return None


def _als_from_random(n, m, p, target_rank, max_iters=100, deadline=None, seed=42):
    """Try ALS from random initialization to find a rank-target_rank decomposition."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    U = np_rng.randn(n*m, target_rank) * 0.5
    V = np_rng.randn(m*p, target_rank) * 0.5
    W = np_rng.randn(p*n, target_rank) * 0.5
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check
        T_check = np.zeros((n, m, p), dtype=np.float64)
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if np.allclose(T_check, 1.0, atol=1e-8):
            U_round = _round_matrix_to_half_int(U.tolist())
            V_round = _round_matrix_to_half_int(V.tolist())
            W_round = _round_matrix_to_half_int(W.tolist())
            if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                return {"U": U_round, "V": V_round, "W": W_round}
    
    return None


def _try_rank_reduction(U, V, W, n, m, p, current_rank, deadline):
    """Try to reduce rank below current_rank using ALS."""
    # Try ranks from current_rank-1 down
    for target in range(current_rank - 1, max(20, current_rank - 10), -1):
        if deadline and time.time() > deadline:
            break
        
        # Try ALS from the existing decomposition (with some columns zeroed)
        result = _als_decompose(n, m, p, target, U, V, W, 
                               max_iters=50, deadline=deadline, seed=target)
        if result:
            return result, target
        
        # Try ALS from random init
        if deadline and time.time() < deadline - 5:
            result = _als_from_random(n, m, p, target, max_iters=30, 
                                      deadline=deadline, seed=target + 1000)
            if result:
                return result, target
    
    return None, current_rank


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Recursive Strassen 4x4x4 + schoolbook 4x4x1 (rank 53)
    try:
        result = _build_4x4x5_strassen()
        rank = len(result["U"][0])
        if _verify_fast_numpy(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Try ALS rank reduction from the Strassen baseline
    if best_result is not None and time.time() < deadline - 15:
        try:
            U_list = best_result["U"]
            V_list = best_result["V"]
            W_list = best_result["W"]
            
            # Try reducing rank from 53
            reduced, new_rank = _try_rank_reduction(U_list, V_list, W_list, n, m, p, 
                                                     best_rank, deadline)
            if reduced and new_rank < best_rank:
                best_result = reduced
                best_rank = new_rank
        except Exception:
            pass
    
    # Approach 3: Schoolbook (rank 80) as fallback
    if best_result is None:
        try:
            result = _build_schoolbook()
            rank = len(result["U"][0])
            if _verify_fast_numpy(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return _build_schoolbook()

# EVOLVE-BLOCK-END
