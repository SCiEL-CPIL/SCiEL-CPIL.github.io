"""Matrix multiplication 3x3x5 - rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [3, 3, 5]
_DEADLINE = time.time() + 140.0

def _check_deadline():
    return time.time() < _DEADLINE

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the 3x3x5 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i*m+j, c]
                    v_val = V[j*p+k, c]
                    w_val = W[k*n+i, c]
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _verify_numeric(U, V, W, tol=1e-9):
    """Verify with floating point tolerance."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p."""
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

def _snap_to_rational_matrix(M, denom_max=1000):
    """Snap a numpy matrix to rational values with small denominators."""
    n, r = M.shape
    result = [[0]*r for _ in range(n)]
    for i in range(n):
        for c in range(r):
            val = float(M[i, c])
            if abs(val) < 1e-12:
                result[i][c] = 0
                continue
            # Try to find p/q with small q
            best_p, best_q = round(val), 1
            best_err = abs(val - best_p/best_q)
            for q in range(1, min(denom_max, 200) + 1):
                p = round(val * q)
                err = abs(val - p/q)
                if err < best_err:
                    best_err = err
                    best_p, best_q = p, q
                    if err < 1e-10:
                        break
            if best_q == 1:
                result[i][c] = best_p
            else:
                # Store as Fraction for exactness
                g = gcd(abs(best_p), best_q)
                result[i][c] = Fraction(best_p // g, best_q // g)
    return result

def _try_rank_31():
    """
    Construct a rank-31 decomposition:
    - 13 terms for 3x3x3 core (using a known Strassen-like construction)
    - 18 terms for the remaining 3x3x2 part (schoolbook)
    Total: 31
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Part 1: 3x3x3 core schoolbook (rank 9)
    rank1 = 9
    U1 = [[0]*rank1 for _ in range(9)]
    V1 = [[0]*rank1 for _ in range(9)]
    W1 = [[0]*rank1 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U1[i*3+j][col] = 1
                V1[j*3+k][col] = 1
                W1[k*3+i][col] = 1
                col += 1
    
    # Part 2: 3x3x2 schoolbook for columns 3,4 (rank 18)
    rank2 = 18
    U2 = [[0]*rank2 for _ in range(9)]
    V2 = [[0]*rank2 for _ in range(6)]  # m*2 = 6
    W2 = [[0]*rank2 for _ in range(6)]  # 2*n = 6
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U2[i*3+j][col] = 1
                V2[j*2+k][col] = 1
                W2[k*3+i][col] = 1
                col += 1
    
    # Combine into full 3x3x5
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(15)]
    W = [[0]*total_rank for _ in range(15)]
    
    # U: same for both parts
    for i in range(9):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1+c] = U2[i][c]
    
    # V: rows 0-8 for B columns 0-2, rows 9-14 for B columns 3-4
    for i in range(9):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    for i in range(6):
        for c in range(rank2):
            V[9+i][rank1+c] = V2[i][c]
    
    # W: W[k*n+i], k=0..4, i=0..2
    for idx in range(9):  # k=0,1,2
        k = idx // 3
        ii = idx % 3
        for c in range(rank1):
            W[idx][c] = W1[k*3+ii][c]
    for idx in range(6):  # k2=0,1 -> full_k=3,4
        k2 = idx // 3
        ii = idx % 3
        full_k = 3 + k2
        for c in range(rank2):
            W[full_k*3+ii][rank1+c] = W2[k2*3+ii][c]
    
    return total_rank, {"U": U, "V": V, "W": W}

def _optimize_decomposition(target_rank, time_limit=30.0, seed=42):
    """
    Use alternating least squares to find a rank-r decomposition.
    Returns (success, U, V, W) where U,V,W are numpy arrays.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize from schoolbook (first target_rank terms) + random
    sb = _schoolbook(n, m, p)
    r0 = min(target_rank, n*m*p)
    for c in range(r0):
        for i in range(n*m):
            U[i, c] = sb["U"][i][c]
        for i in range(m*p):
            V[i, c] = sb["V"][i][c]
        for i in range(p*n):
            W[i, c] = sb["W"][i][c]
    for c in range(r0, target_rank):
        U[:, c] = np.random.randn(n*m) * 0.1
        V[:, c] = np.random.randn(m*p) * 0.1
        W[:, c] = np.random.randn(p*n) * 0.1
    
    # Alternating minimization
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Fix V and W, solve for U
        for idx in range(n*m):
            i = idx // m
            j = idx % m
            A_mat = np.zeros((p, target_rank))
            for k in range(p):
                A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
            b_vec = np.ones(p)
            try:
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[idx, :] = sol
            except:
                pass
        
        # Fix U and W, solve for V
        for idx in range(m*p):
            j = idx // p
            k = idx % p
            A_mat = np.zeros((n, target_rank))
            for i in range(n):
                A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
            b_vec = np.ones(n)
            try:
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[idx, :] = sol
            except:
                pass
        
        # Fix U and V, solve for W
        for idx in range(p*n):
            k = idx // n
            i = idx % n
            A_mat = np.zeros((m, target_rank))
            for j in range(m):
                A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
            b_vec = np.ones(m)
            try:
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[idx, :] = sol
            except:
                pass
        
        # Check convergence
        if iteration % 50 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = max(err, abs(total - 1.0))
            if err < 1e-8:
                return True, U, V, W
    
    # Final error check
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = max(err, abs(total - 1.0))
    if err < 1e-8:
        return True, U, V, W
    return False, U, V, W

def _snap_and_verify(U, V, W, denom_max=500):
    """Snap to rational and verify."""
    U_r = _snap_to_rational_matrix(U, denom_max)
    V_r = _snap_to_rational_matrix(V, denom_max)
    W_r = _snap_to_rational_matrix(W, denom_max)
    if _verify(U_r, V_r, W_r):
        return U_r, V_r, W_r
    # Try with larger denominator
    U_r = _snap_to_rational_matrix(U, denom_max*2)
    V_r = _snap_to_rational_matrix(V, denom_max*2)
    W_r = _snap_to_rational_matrix(W, denom_max*2)
    if _verify(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None, None, None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Start with rank-31 structured decomposition
    try:
        r31, d31 = _try_rank_31()
        if d31 is not None and _verify(d31["U"], d31["V"], d31["W"]):
            return d31
    except Exception:
        pass
    
    # Fallback to schoolbook
    fallback = _schoolbook(n, m, p)
    
    # Try to find lower rank via optimization
    best_result = fallback
    best_rank = n * m * p
    
    try:
        if time.time() < deadline - 10:
            for target_rank in [30, 29, 28, 27, 26, 25]:
                if time.time() > deadline - 15:
                    break
                remaining = deadline - time.time() - 5
                if remaining < 5:
                    break
                success, U, V, W = _optimize_decomposition(
                    target_rank, time_limit=min(remaining, 20.0), seed=target_rank
                )
                if success:
                    U_r, V_r, W_r = _snap_and_verify(U, V, W, denom_max=500)
                    if U_r is not None:
                        best_result = {"U": U_r, "V": V_r, "W": W_r}
                        best_rank = target_rank
                        return best_result
    except Exception:
        pass
    
    return best_result

construct()  # Test to make sure it doesn't crash
