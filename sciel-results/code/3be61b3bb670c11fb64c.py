"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS
DEADLINE = 110.0


def _build_rank60():
    """Build a rank-60 decomposition for 3x4 x 4x6.
    
    Split the 6 output columns into two blocks of 3.
    For each block: 27 terms for 3x3x3 schoolbook + 3 terms for 3x1x3.
    Total = 2 * (27 + 3) = 60.
    """
    R = 60
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        k_base = block * 3
        # 27 terms for 3x3x3 part (j in 0..2, k in k_base..k_base+2)
        for i in range(3):
            for j in range(3):
                for k_loc in range(3):
                    k = k_base + k_loc
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        # 3 terms for 3x1x3 part (j=3, k in k_base..k_base+2)
        # For 3x1x3, rank is 3 (each diagonal term is independent)
        for r in range(3):
            i = r
            k = k_base + r
            U[i * m + 3][col] = 1.0
            V[3 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Build the schoolbook rank-72 decomposition."""
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6, full_zero=True):
    """Verify the decomposition exactly using numpy for speed."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(U_np[a, :], V_np[b, :] * W_np[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries
    if full_zero:
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = float(np.dot(U_np[a, :], V_np[b, :] * W_np[c, :]))
                        if abs(val) > tol:
                            return False
    
    return True


def _try_alternating_least_squares(rank, max_iter=2000, seed=42, start_time=0.0, deadline=110.0):
    """Try to find a rank-r decomposition using proper alternating least squares."""
    def check_time():
        return time.time() - start_time < deadline
    
    if not check_time():
        return None
    
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Precompute all target triples
    nz = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if not check_time():
            break
        
        # Update U: for each row a, solve least squares
        for a in range(n * m):
            # Find all (b, c) pairs where this row a is used
            # a = i*m + j, so i = a//m, j = a%m
            i = a // m
            j = a % m
            # b = j*p + k for k in range(p), c = k*n + i
            # We want to minimize sum_k (U[a,:] dot (V[b_k,:] * W[c_k,:]) - 1)^2
            # Let M be a matrix where M[k, r] = V[b_k, r] * W[c_k, r]
            # Then we minimize ||M @ U[a,:] - 1||^2
            # Solution: U[a,:] = (M^T M)^{-1} M^T 1
            M = np.zeros((p, rank))
            for k in range(p):
                b = j * p + k
                c = k * n + i
                M[k, :] = V[b, :] * W[c, :]
            # Solve least squares
            try:
                sol = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                U[a, :] = sol
            except:
                pass
        
        # Update V: for each row b, solve least squares
        for b in range(m * p):
            j = b // p
            k = b % p
            # b = j*p + k, so for each i in range(n): a = i*m + j, c = k*n + i
            M = np.zeros((n, rank))
            for i in range(n):
                a = i * m + j
                c = k * n + i
                M[i, :] = U[a, :] * W[c, :]
            try:
                sol = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                V[b, :] = sol
            except:
                pass
        
        # Update W: for each row c, solve least squares
        for c in range(p * n):
            k = c // n
            i = c % n
            # c = k*n + i, so for each j in range(m): a = i*m + j, b = j*p + k
            M = np.zeros((m, rank))
            for j in range(m):
                a = i * m + j
                b = j * p + k
                M[j, :] = U[a, :] * V[b, :]
            try:
                sol = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                W[c, :] = sol
            except:
                pass
        
        # Compute loss
        loss = 0.0
        for a, b, c in nz:
            s = float(np.dot(U[a, :], V[b, :] * W[c, :]))
            loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if it % 100 == 0:
            U_l, V_l, W_l = best_U.tolist(), best_V.tolist(), best_W.tolist()
            if _verify(U_l, V_l, W_l, tol=1e-4, full_zero=False):
                if _verify(U_l, V_l, W_l, tol=1e-6, full_zero=True):
                    return {"U": U_l, "V": V_l, "W": W_l}
    
    # Final check
    U_l, V_l, W_l = best_U.tolist(), best_V.tolist(), best_W.tolist()
    if _verify(U_l, V_l, W_l, tol=1e-6, full_zero=True):
        return {"U": U_l, "V": V_l, "W": W_l}
    
    return None


def construct():
    start_time = time.time()
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    # Primary: rank-60 solution
    best = _build_rank60()
    if _verify(best["U"], best["V"], best["W"], tol=1e-6, full_zero=True):
        best_rank = 60
    else:
        # Fallback to schoolbook rank 72
        best = _build_schoolbook()
        best_rank = 72
    
    # Try to improve with numerical search
    for r in range(40, best_rank):
        if not check_time():
            break
        for seed in range(10):
            if not check_time():
                break
            res = _try_alternating_least_squares(r, max_iter=1500, seed=seed,
                                                 start_time=start_time, deadline=DEADLINE)
            if res is not None:
                if _verify(res["U"], res["V"], res["W"], tol=1e-6, full_zero=True):
                    best_rank = r
                    best = res
                    break
    
    return best
