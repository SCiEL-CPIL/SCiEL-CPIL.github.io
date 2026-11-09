import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 10.0


def _ok():
    return time.time() < DEADLINE


def _verify_3x6(U, V, W, iters=8, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.einsum('ijc,jkc,kic->ik', U3 * a[:, :, None], V3 * b[:, :, None], W3)
        # Actually need: sum_c U[i,j,c]*a[i,j] * V[j,k,c]*b[j,k] * W[k,i,c]
        # = sum_c (sum_j U[i,j,c]*a[i,j]) * (sum_j V[j,k,c]*b[j,k]) * W[k,i,c]
        # Let's do it properly:
        Cc2 = np.zeros((n, p))
        for c in range(R):
            ua = np.einsum('ij,ijc->ic', a, U3)  # (n, R)
            vb = np.einsum('jk,jkc->kc', b, V3)  # (m, R) -> but we need per (k)
            # Actually: for each c, contribution = (U^T a)_i * (V^T b)_k * W[k,i,c]
            # (U^T a)_i = sum_j U[i,j,c] * a[i,j]  -- NO, U is (n*m, R) = U[i*m+j, c]
            # Let me just use the direct formula
            pass
        
        # Direct verification (slower but correct)
        Cc = np.zeros((n, p))
        for c in range(R):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j, c] * a[i, j] * V[j * p + k, c] * b[j, k]
                    Cc[i, k] += s * W[k * n + i, c]
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True


def _verify_tensor(U, V, W, n=3, m=3, p=6):
    """Verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta_{i,k} for all i,j,k"""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True


def _schoolbook():
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


def _solve_w_exact(A, b):
    """Solve A @ w = b exactly using Fraction arithmetic. Returns None if no exact half-integer solution."""
    R = A.shape[1]
    n_eq = A.shape[0]
    
    # Convert to fractions
    A_frac = np.zeros((n_eq, R), dtype=object)
    b_frac = np.zeros(n_eq, dtype=object)
    for i in range(n_eq):
        for j in range(R):
            A_frac[i, j] = Fraction(A[i, j]).limit_denominator(1000)
        b_frac[i] = Fraction(b[i]).limit_denominator(1000)
    
    # Gaussian elimination
    try:
        M = [list(A_frac[i]) + [b_frac[i]] for i in range(n_eq)]
        rank_actual = 0
        for col in range(R):
            if rank_actual >= n_eq:
                break
            # Find pivot
            pivot_row = None
            for row in range(rank_actual, n_eq):
                if M[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            M[rank_actual], M[pivot_row] = M[pivot_row], M[rank_actual]
            # Eliminate
            pivot = M[rank_actual][col]
            for j in range(col, R + 1):
                M[rank_actual][j] /= pivot
            for row in range(n_eq):
                if row != rank_actual and M[row][col] != 0:
                    factor = M[row][col]
                    for j in range(col, R + 1):
                        M[row][j] -= factor * M[rank_actual][j]
            rank_actual += 1
        
        # Check consistency
        for row in range(rank_actual, n_eq):
            if any(M[row][j] != 0 for j in range(R)):
                return None
        
        # Get solution (set free variables to 0)
        w = [Fraction(0)] * R
        for i in range(rank_actual):
            # Find the column that is pivot for row i
            for col in range(R):
                if M[i][col] == 1 and all(M[i][j] == 0 for j in range(R) if j != col):
                    w[col] = M[i][R]
                    break
        
        # Check if all are half-integers (denominator divides 2)
        for wi in w:
            if wi.denominator > 2:
                return None
        
        return np.array([float(wi) for wi in w])
    except Exception:
        return None


def _find_3x3x2_decomp(rank, seed, trials=200):
    """Find rank decomposition for 3x3 * 3x2 matrix multiplication."""
    n, m, p = 3, 3, 2
    rng = np.random.default_rng(seed)
    
    for trial in range(trials):
        if not _ok():
            return None
        
        # Generate sparse U and V with small integer/half-integer values
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = np.zeros((rank, n, m))
        V = np.zeros((rank, m, p))
        
        # Each column of U and V has 1-2 non-zeros
        for t in range(rank):
            nz_u = rng.integers(1, 3)
            for _ in range(nz_u):
                i = rng.integers(0, n)
                k = rng.integers(0, m)
                U[t, i, k] = rng.choice(vals)
            
            nz_v = rng.integers(1, 3)
            for _ in range(nz_v):
                k = rng.integers(0, m)
                j = rng.integers(0, p)
                V[t, k, j] = rng.choice(vals)
        
        W = np.zeros((rank, n, p))
        valid = True
        
        for i in range(n):
            for j in range(p):
                A = np.zeros((m * m, rank))
                b = np.zeros(m * m)
                for k in range(m):
                    for kp in range(m):
                        A[k * m + kp, :] = U[:, i, k] * V[:, kp, j]
                        b[k * m + kp] = 1.0 if k == kp else 0.0
                
                w_col = _solve_w_exact(A, b)
                if w_col is None:
                    valid = False
                    break
                W[:, i, j] = w_col
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        check = True
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    for kp in range(m):
                        s = np.sum(U[:, i, k] * V[:, kp, j] * W[:, i, j])
                        expected = 1.0 if k == kp else 0.0
                        if abs(s - expected) > 1e-9:
                            check = False
                            break
                    if not check:
                        break
                if not check:
                    break
            if not check:
                break
        
        if check:
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None


def _build_3x6_from_3x3x2(r, d0, d1, d2):
    n, m, p = 3, 3, 6
    R = 3 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1, d2)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(2):
                    V[k * 6 + (q * 2 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(2):
                    W[(q * 2 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def _direct_search_3x3x6(rank, seed, trials=100):
    """Direct search for 3x3x6 decomposition."""
    n, m, p = 3, 3, 6
    rng = np.random.default_rng(seed)
    vals = np.array([-1, -0.5, 0, 0.5, 1])
    
    for trial in range(trials):
        if not _ok():
            return None
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for c in range(rank):
            nz_u = rng.integers(1, 3)
            for _ in range(nz_u):
                idx = rng.integers(0, n * m)
                U[idx, c] = rng.choice(vals)
            nz_v = rng.integers(1, 3)
            for _ in range(nz_v):
                idx = rng.integers(0, m * p)
                V[idx, c] = rng.choice(vals)
        
        # Solve for W: for each (k, i), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k}
        # Group by (k, i): equation for W[k*n+i, :]
        W = np.zeros((p * n, rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # A[j, c] = sum over j' of U[i*m+j', c] * V[j'*p+k, c]... 
                # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} for all j
                # So for each j: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = delta_{i,k}
                A = np.zeros((m, rank))
                b = np.zeros(m)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b[j] = 1.0 if i == k else 0.0
                
                w_col = _solve_w_exact(A, b)
                if w_col is None:
                    valid = False
                    break
                W[k * n + i, :] = w_col
            
            if not valid:
                break
        
        if not valid:
            continue
        
        if _verify_tensor(U, V, W):
            return U, V, W
    
    return None


def construct():
    # Try 3x3x2 block approach with various ranks
    for r in range(12, 5, -1):
        if not _ok():
            break
        d0 = _find_3x3x2_decomp(r, seed=1000 + r)
        if d0 is None:
            continue
        d1 = _find_3x3x2_decomp(r, seed=3000 + r)
        if d1 is None:
            continue
        d2 = _find_3x3x2_decomp(r, seed=5000 + r)
        if d2 is None:
            continue
        U, V, W = _build_3x6_from_3x3x2(r, d0, d1, d2)
        if _verify_3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Direct search
    for rank in range(20, 55, 2):
        if not _ok():
            break
        result = _direct_search_3x3x6(rank, seed=rank * 777, trials=50)
        if result is not None:
            U, V, W = result
            if _verify_3x6(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook()
