import numpy as np
import random
import time
from itertools import combinations, product

DIMENSIONS = [2, 3, 10]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            Uw = U_mat[i] * W_mat[k, i]
            M_ik = Uw @ V_mat[:, k].T
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True

def _is_valid_entries(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _build_strassen55(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _try_drop_term(U55, V55, W55, n, m, p, drop_idx, deadline):
    rank55 = 55
    keep_idx = [i for i in range(rank55) if i != drop_idx]
    U_keep = U55[:, keep_idx]
    V_keep = V55[:, keep_idx]
    W_keep = W55[:, keep_idx]
    
    eqs = []
    b_vec = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                row = []
                for r in range(54):
                    row.append(U_keep[a, r] * V_keep[b, r] * W_keep[c, r])
                eqs.append(row)
                b_vec.append(1.0)
    
    A = np.array(eqs, dtype=np.float64)
    b = np.array(b_vec, dtype=np.float64)
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    valid = True
    for cv in coeffs:
        if abs(cv * 2 - round(cv * 2)) > 1e-6:
            valid = False
            break
    if not valid:
        return None
    
    U_new = U_keep * coeffs
    V_new = V_keep * coeffs
    W_new = W_keep * coeffs
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def _try_solve_W(U55, V55, W55, n, m, p, target_rank, deadline):
    """Fix U and V from Strassen-55, solve for W coefficients to achieve target_rank."""
    if target_rank >= 55:
        return None
    
    # Use first target_rank columns of U and V
    U_fix = U55[:, :target_rank]
    V_fix = V55[:, :target_rank]
    
    # Build linear system: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Unknowns: W[k*n+i,r] for r in range(target_rank)
    # Total unknowns: p * n * target_rank
    # Total equations: n * m * p = 60
    
    num_unknowns = p * n * target_rank
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_unknowns), dtype=np.float64)
    b = np.ones(num_eqs, dtype=np.float64)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                for r in range(target_rank):
                    c_idx = k * n + i
                    col = c_idx * target_rank + r
                    A[eq_idx, col] = U_fix[a_idx, r] * V_fix[b_idx, r]
                eq_idx += 1
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    # Check if coefficients are valid (integer or half-integer)
    valid = True
    for cv in coeffs:
        if abs(cv * 2 - round(cv * 2)) > 1e-6:
            valid = False
            break
    if not valid:
        return None
    
    # Reconstruct W
    W_new = np.zeros((p * n, target_rank), dtype=np.float64)
    for c in range(p * n):
        for r in range(target_rank):
            W_new[c, r] = coeffs[c * target_rank + r]
    
    U_new = U_fix
    V_new = V_fix
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def _try_solve_UV_fixed(U55, V55, W55, n, m, p, target_rank, deadline):
    """Try multiple subsets of columns for U and V, solve for W."""
    if target_rank >= 55:
        return None
    
    # Try different column subsets for U and V
    # First try: use first target_rank columns
    result = _try_solve_W(U55, V55, W55, n, m, p, target_rank, deadline)
    if result is not None:
        return result
    
    # Try dropping different columns from U and V
    if target_rank >= 54 and time.time() < deadline - 5:
        # Try dropping column 0 from U and V
        for drop_u in range(min(5, target_rank)):
            if time.time() > deadline:
                break
            keep_u = [i for i in range(55) if i != drop_u][:target_rank]
            if len(keep_u) < target_rank:
                continue
            U_sub = U55[:, keep_u]
            V_sub = V55[:, keep_u]
            
            num_unknowns = p * n * target_rank
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_unknowns), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        for r in range(target_rank):
                            c_idx = k * n + i
                            col = c_idx * target_rank + r
                            A[eq_idx, col] = U_sub[a_idx, r] * V_sub[b_idx, r]
                        eq_idx += 1
            
            try:
                coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            valid = True
            for cv in coeffs:
                if abs(cv * 2 - round(cv * 2)) > 1e-6:
                    valid = False
                    break
            if not valid:
                continue
            
            W_new = np.zeros((p * n, target_rank), dtype=np.float64)
            for c in range(p * n):
                for r in range(target_rank):
                    W_new[c, r] = coeffs[c * target_rank + r]
            
            U_new = U_sub
            V_new = V_sub
            
            if _verify_fast(U_new, V_new, W_new, n, m, p):
                if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                    return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    U55 = np.array(U_best, dtype=np.float64)
    V55 = np.array(V_best, dtype=np.float64)
    W55 = np.array(W_best, dtype=np.float64)

    # Try solving for W with fixed U,V at rank 54
    if time.time() < deadline - 30:
        result = _try_solve_UV_fixed(U55, V55, W55, n, m, p, 54, deadline)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try dropping individual terms
    if time.time() < deadline - 20:
        for i in range(55):
            if time.time() > deadline:
                break
            result = _try_drop_term(U55, V55, W55, n, m, p, i, deadline)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = 54
                return {"U": U_best, "V": V_best, "W": W_best}

    return {"U": U_best, "V": V_best, "W": W_best}
