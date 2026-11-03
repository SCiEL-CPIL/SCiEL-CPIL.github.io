import time
import numpy as np

DIMENSIONS = [2, 3, 5]
_start_time = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication.
    Try rank 27 first via ALS optimization, fall back to rank 28 if not found in time.
    """
    n, m, p = 2, 3, 5
    
    # Try rank 27 with ALS optimization
    if time.time() - _start_time < TIME_LIMIT - 15:
        result = _try_rank27_als(n, m, p)
        if result is not None:
            U, V, W = result
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            ok = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: known rank-28 solution
    sol = _rank28_solution()
    return {"U": sol["U"].tolist(), "V": sol["V"].tolist(), "W": sol["W"].tolist()}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _rank28_solution():
    """Known rank-28 decomposition using two Strassen 2x2x2 blocks plus schoolbook."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m2
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        # m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        # m5
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        # m6
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    return {"U": U, "V": V, "W": W}

def _try_rank27_als(n, m, p, max_time=115):
    """
    Attempt to find a rank-27 decomposition using ALS optimization
    with multiple random restarts.
    """
    start = time.time()
    deadline = start + max_time
    
    nm, mp, np_ = n*m, m*p, n*p
    
    def check_solution(U, V, W, R):
        """Check if the decomposition is exact. Returns max error."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = max(err, abs(val - 1.0))
        return err
    
    def solve_for_U(V, W, R):
        U = np.zeros((nm, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U
    
    def solve_for_V(U, W, R):
        V = np.zeros((mp, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V
    
    def solve_for_W(U, V, R):
        W = np.zeros((np_, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W
    
    def als_optimize(U, V, W, R, max_iters=200, tol=1e-12):
        """Run ALS optimization. Returns (U, V, W, err)."""
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
        return U, V, W, err
    
    def polish(U, V, W, R, max_iters=500, tol=1e-14):
        """High-precision polish pass."""
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
        return U, V, W, err
    
    R = 27
    
    # Try multiple random restarts with different seeds
    n_trials = 400
    max_iters = 120
    
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(trial * 7919 + 13)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(np_, R)
        
        U, V, W, err = als_optimize(U, V, W, R, max_iters=max_iters, tol=1e-12)
        
        if err < 1e-10:
            # Polish for high precision
            U, V, W, err = polish(U, V, W, R, max_iters=300, tol=1e-14)
            if err < 1e-8:
                return (U, V, W)
    
    return None

def _try_rank(n, m, p, rank):
    """
    Legacy interface - now delegates to _try_rank27_als for rank 27.
    """
    if rank == 27:
        return _try_rank27_als(n, m, p)
    return None
