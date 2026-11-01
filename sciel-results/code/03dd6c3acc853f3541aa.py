"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Constructive combinatorial design:
   - Two 2x2x2 Strassen blocks (rank 7 each) for column pairs (0,1) and (2,3)
   - A 2x2x1 schoolbook block (rank 4) for column 4
   - Total rank 18 baseline
2. Attempt to reduce rank by sharing/absorbing terms between blocks
3. Numerical ALS + exact rational polish as fallback
4. Schoolbook fallback (rank 20)

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    """Verify decomposition matches target tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    R = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = 0.0
                for r in range(R):
                    total += U_arr[u, r] * V_arr[v, r] * W_arr[w, r]
                if abs(total - T[u, v, w]) > tol:
                    return False
    return True


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < 1e-7:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(64)
        if abs(x - float(f)) < 1e-7:
            return f
        return Fraction(x).limit_denominator(64)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    """Add 7 Strassen terms for 2x2 x 2x2 block with B columns col0, col1."""
    # M1: f=A00+A11, g=B[0,col0]+B[1,col1]
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1
    
    # M2: f=A10+A11, g=B[0,col0]
    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1
    
    # M3: f=A00, g=B[0,col1]-B[1,col1]
    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1
    
    # M4: f=A11, g=B[1,col0]-B[0,col0]
    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1
    
    # M5: f=A00+A01, g=B[1,col1]
    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1
    
    # M6: f=A10-A00, g=B[0,col0]+B[0,col1]
    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1
    
    # M7: f=A01-A11, g=B[1,col0]+B[1,col1]
    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    """Construct the rank-18 decomposition: 2 Strassen blocks + schoolbook for col 4."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen block 1: columns 0, 1
    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    
    # Strassen block 2: columns 2, 3
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W


def _try_reduce_rank18_to_17():
    """Attempt to find a rank-17 decomposition by modifying the rank-18 structure.
    
    Strategy: Try to absorb one schoolbook term for column 4 into the Strassen
    terms by finding a linear combination that preserves the decomposition.
    """
    n, m, p = DIMENSIONS
    U18, V18, W18 = _build_rank18()
    T = _build_target()
    
    # Try removing each of the 4 schoolbook terms (indices 14-17) and see if
    # we can compensate by modifying the Strassen terms
    for remove_idx in [14, 15, 16, 17]:
        # Create rank-17 candidate by zeroing out the removed term
        U17 = [[x for x in row] for row in U18]
        V17 = [[x for x in row] for row in V18]
        W17 = [[x for x in row] for row in W18]
        
        # Zero out column remove_idx
        for i in range(n * m):
            U17[i][remove_idx] = 0
        for i in range(m * p):
            V17[i][remove_idx] = 0
        for i in range(p * n):
            W17[i][remove_idx] = 0
        
        # Now try to add small corrections to other terms to compensate
        # This is a heuristic search
        val_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        best_err = np.inf
        best_candidate = None
        
        # Try modifying one other term at a time
        for modify_idx in range(14):
            for du in val_set:
                for dv in val_set:
                    for dw in val_set:
                        if du == 0 and dv == 0 and dw == 0:
                            continue
                        
                        U_test = [[x for x in row] for row in U17]
                        V_test = [[x for x in row] for row in V17]
                        W_test = [[x for x in row] for row in W17]
                        
                        # Apply modification to term modify_idx
                        # Only modify non-zero positions
                        for i in range(n * m):
                            if U17[i][modify_idx] != 0:
                                U_test[i][modify_idx] += du
                        for i in range(m * p):
                            if V17[i][modify_idx] != 0:
                                V_test[i][modify_idx] += dv
                        for i in range(p * n):
                            if W17[i][modify_idx] != 0:
                                W_test[i][modify_idx] += dw
                        
                        # Check decomposition
                        err = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    u = i * m + j
                                    v = j * p + k
                                    w = k * n + i
                                    total = 0.0
                                    for r in range(17):
                                        total += U_test[u][r] * V_test[v][r] * W_test[w][r]
                                    err += (total - T[u, v, w]) ** 2
                        
                        if err < best_err:
                            best_err = err
                            best_candidate = (U_test, V_test, W_test)
        
        if best_err < 1e-10:
            return best_candidate
    
    return None


def _try_als_rank(R, T, deadline, n_seeds=5, max_iters=100):
    """Try ALS for a given rank with rational polish."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    seeds = [42, 123, 7, 999, 2024]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        inits = []
        inits.append((np.random.randn(U_rows, R) * 0.3,
                      np.random.randn(V_rows, R) * 0.3,
                      np.random.randn(W_rows, R) * 0.3))
        inits.append((np.random.randn(U_rows, R) * 1.0,
                      np.random.randn(V_rows, R) * 1.0,
                      np.random.randn(W_rows, R) * 1.0))

        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U
                A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
                for u in range(U_rows):
                    b = T[u].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[u] = sol
                
                # Update V
                A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
                for v in range(V_rows):
                    b = T[:, v, :].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[v] = sol
                
                # Update W
                A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
                for w in range(W_rows):
                    b = T[:, :, w].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[w] = sol
                
                result = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
                for r in range(R):
                    result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(result - T.astype(np.float64))
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

            if _check_decomposition(U, V, W, T, tol=1e-7):
                # Try to snap to rationals
                for denom in [1, 2]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()
    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    T = _build_target()

    # First, try the constructive rank-18 solution
    U18, V18, W18 = _build_rank18()
    if _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Try to reduce to rank 17
        result17 = _try_reduce_rank18_to_17()
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}
        
        # Return rank-18 solution
        return {"U": U18, "V": V18, "W": W18}

    # Fallback: try ALS for lower ranks
    for R in range(7, 19):
        if time.time() > deadline:
            break
        result = _try_als_rank(R, T, deadline, n_seeds=3, max_iters=50)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}

    # Final fallback: schoolbook
    U_l, V_l, W_l = _schoolbook_decomposition()
    return {"U": U_l, "V": V_l, "W": W_l}
