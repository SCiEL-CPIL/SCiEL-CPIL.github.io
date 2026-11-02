"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Hard-coded structured block decomposition with optimal column grouping
2. Fast ALS refinement with tight deadlines
3. Rational polish (half-integer / small-denominator snapping + Fraction verify)
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


def _decompose(U, V, W):
    """Reconstruct tensor from factor matrices."""
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            num = round(x * denom)
            candidate = Fraction(int(num), denom)
            if abs(x - float(candidate)) < 1e-9:
                return candidate
        return Fraction(x).limit_denominator(128)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(U.shape[0]):
        for v in range(V.shape[0]):
            for w in range(W.shape[0]):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    """Try to snap U,V,W to small-denominator rationals that still decompose exactly."""
    for denom in [1, 2, 4, 8, 16, 32, 64]:
        U_s = _snap(U, denom)
        V_s = _snap(V, denom)
        W_s = _snap(W, denom)
        if _check_decomposition(U_s, V_s, W_s, T, tol=1e-6):
            if _check_exact(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
    for dU in [1, 2, 4]:
        for dV in [1, 2, 4]:
            for dW in [1, 2, 4]:
                U_s = _snap(U, dU)
                V_s = _snap(V, dV)
                W_s = _snap(W, dW)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    return None


def _als_update_U(T, V, W):
    R = V.shape[1]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    U = np.zeros((T.shape[0], R), dtype=np.float64)
    for u in range(T.shape[0]):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    V = np.zeros((T.shape[1], R), dtype=np.float64)
    for v in range(T.shape[1]):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    W = np.zeros((T.shape[2], R), dtype=np.float64)
    for w in range(T.shape[2]):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W[w] = sol
    return W


def _build_structured_rank18():
    """
    Build rank-18 structured decomposition:
    - Strassen for columns (0,1): rank 7
    - Strassen for columns (2,3): rank 7
    - Schoolbook for column 4: rank 4
    Total: rank 18
    """
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1

        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1

        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1

        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1

        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1

        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1

        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)

    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1

    return U, V, W


def _build_structured_rank17_attempt():
    """
    Attempt rank-17 by using a smarter column grouping.
    Group columns as (0,4), (1,2), (3) to allow sharing of Strassen terms.
    
    Strategy: Use Strassen-like decompositions where the "diagonal" terms
    can be shared across different column pairs.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    U = np.zeros((n * m, target_rank), dtype=np.float64)
    V = np.zeros((m * p, target_rank), dtype=np.float64)
    W = np.zeros((p * n, target_rank), dtype=np.float64)

    # Try a hybrid: Strassen for (0,1), modified Strassen for (2,3,4)
    # with one shared term
    
    # Block 1: Strassen for columns 0, 1 (terms 0-6)
    # M1: f=A00+A11, g=B[0,0]+B[1,1]
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0*p+0, r] = 1; V[1*p+1, r] = 1
    W[0*n+0, r] = 1; W[1*n+1, r] = 1

    # M2: f=A10+A11, g=B[0,0]
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0*p+0, r] = 1
    W[0*n+1, r] = 1; W[1*n+1, r] = -1

    # M3: f=A00, g=B[0,1]-B[1,1]
    r = 2
    U[0, r] = 1
    V[0*p+1, r] = 1; V[1*p+1, r] = -1
    W[1*n+0, r] = 1; W[1*n+1, r] = 1

    # M4: f=A11, g=B[1,0]-B[0,0]
    r = 3
    U[3, r] = 1
    V[1*p+0, r] = 1; V[0*p+0, r] = -1
    W[0*n+0, r] = 1; W[0*n+1, r] = 1

    # M5: f=A00+A01, g=B[1,1]
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[1*p+1, r] = 1
    W[0*n+0, r] = -1; W[1*n+0, r] = 1

    # M6: f=A10-A00, g=B[0,0]+B[0,1]
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0*p+0, r] = 1; V[0*p+1, r] = 1
    W[1*n+1, r] = 1

    # M7: f=A01-A11, g=B[1,0]+B[1,1]
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[1*p+0, r] = 1; V[1*p+1, r] = 1
    W[0*n+0, r] = 1

    # Block 2: Modified Strassen for columns 2, 3, 4 (terms 7-16)
    # Use 10 terms for 3 columns (instead of 7+4=11)
    
    # M8: f=A00+A11, g=B[0,2]+B[1,3]
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[0*p+2, r] = 1; V[1*p+3, r] = 1
    W[2*n+0, r] = 1; W[3*n+1, r] = 1

    # M9: f=A10+A11, g=B[0,2]
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[0*p+2, r] = 1
    W[2*n+1, r] = 1; W[3*n+1, r] = -1

    # M10: f=A00, g=B[0,3]-B[1,3]
    r = 9
    U[0, r] = 1
    V[0*p+3, r] = 1; V[1*p+3, r] = -1
    W[3*n+0, r] = 1; W[3*n+1, r] = 1

    # M11: f=A11, g=B[1,2]-B[0,2]
    r = 10
    U[3, r] = 1
    V[1*p+2, r] = 1; V[0*p+2, r] = -1
    W[2*n+0, r] = 1; W[2*n+1, r] = 1

    # M12: f=A00+A01, g=B[1,3]
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[1*p+3, r] = 1
    W[2*n+0, r] = -1; W[3*n+0, r] = 1

    # M13: f=A10-A00, g=B[0,2]+B[0,3]
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[0*p+2, r] = 1; V[0*p+3, r] = 1
    W[3*n+1, r] = 1

    # M14: f=A01-A11, g=B[1,2]+B[1,3]
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[1*p+2, r] = 1; V[1*p+3, r] = 1
    W[2*n+0, r] = 1

    # M15: f=A00, g=B[0,4]
    r = 14
    U[0, r] = 1
    V[0*p+4, r] = 1
    W[4*n+0, r] = 1

    # M16: f=A10, g=B[0,4]
    r = 15
    U[2, r] = 1
    V[0*p+4, r] = 1
    W[4*n+1, r] = 1

    # M17: f=A11, g=B[1,4] + f=A01, g=B[1,4] combined
    r = 16
    U[1, r] = 1; U[3, r] = 1
    V[1*p+4, r] = 1
    W[4*n+0, r] = 1; W[4*n+1, r] = 1

    return U, V, W


def _try_als_refinement(U_init, V_init, W_init, T, max_iters=50, deadline=None):
    """Run ALS refinement from initialization, return improved solution if exact."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        U = _als_update_U(T, V, W)
        V = _als_update_V(T, U, W)
        W = _als_update_W(T, U, V)
        
        err = np.linalg.norm(_decompose(U, V, W) - T.astype(np.float64))
        if abs(prev_err - err) < 1e-10:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T, tol=1e-7):
        polished = _polish_rational(U, V, W, T)
        if polished is not None:
            return polished
        return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_rank_reduction_fast(U, V, W, T, deadline, target_rank):
    """
    Try to reduce rank by removing one column and re-solving.
    Fast version with tight deadline.
    """
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = U.shape[0], V.shape[0], W.shape[0]
    R = U.shape[1]

    if target_rank < 1 or time.time() > deadline:
        return None

    # Try removing each column and re-solving with ALS
    for r_remove in range(R):
        if time.time() > deadline:
            break

        mask = [i for i in range(R) if i != r_remove]
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]

        # Initialize with the sub-decomposition plus a small random column
        np.random.seed(42 + r_remove)
        U_new = np.hstack([U_sub, np.random.randn(U_rows, 1) * 0.1])
        V_new = np.hstack([V_sub, np.random.randn(V_rows, 1) * 0.1])
        W_new = np.hstack([W_sub, np.random.randn(W_rows, 1) * 0.1])

        result = _try_als_refinement(U_new, V_new, W_new, T, max_iters=30, deadline=deadline)
        if result is not None:
            return result

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

    # Build the structured decomposition (rank 18)
    U_struct, V_struct, W_struct = _build_structured_rank18()

    # Verify the structured decomposition is valid
    if not _check_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        U_l, V_l, W_l = _schoolbook_decomposition()
        return {"U": U_l, "V": V_l, "W": W_l}

    # Try structured rank-17 attempt
    if time.time() < deadline:
        U17, V17, W17 = _build_structured_rank17_attempt()
        if _check_decomposition(U17, V17, W17, T, tol=1e-7):
            polished = _polish_rational(U17, V17, W17, T)
            if polished is not None:
                return {"U": polished[0], "V": polished[1], "W": polished[2]}
            return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
        else:
            # Try ALS refinement from the rank-17 initialization
            result = _try_als_refinement(U17, V17, W17, T, max_iters=40, deadline=deadline)
            if result is not None:
                return {"U": result[0], "V": result[1], "W": result[2]}

    # Try rank reduction from rank-18 structured solution
    if time.time() < deadline:
        result = _try_rank_reduction_fast(U_struct, V_struct, W_struct, T, deadline, 17)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

    # If no reduction found, return the polished structured solution (rank 18)
    polished = _polish_rational(U_struct, V_struct, W_struct, T)
    if polished is not None:
        return {"U": polished[0], "V": polished[1], "W": polished[2]}

    return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
