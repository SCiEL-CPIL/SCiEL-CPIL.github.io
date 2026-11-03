"""Matrix multiplication 2x3 x 3x4 rank-minimal decomposition.
Uses structured Strassen-based construction with careful indexing.
Partitions the middle dimension as 2+1, applies Strassen to 2x2 blocks,
and handles the remaining 1-dimensional interaction optimally.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p, tol=1e-8):
    """Exact verification of the decomposition using vectorized computation."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j): U[i*m+j, :] is a vector of length rank
    # For each (jp,k): V[jp*p+k, :] is a vector of length rank
    # For each (kp,ip): W[kp*n+ip, :] is a vector of length rank
    # The tensor entry T[i,j,kp,ip,jp,k] = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
    # We need T[i,j,k,ip,jp,k] = delta(i,ip)*delta(j,jp)*delta(k,k)
    # i.e., T[i,j,k,i,j,k] = 1 and all other entries = 0
    for i in range(n):
        for j in range(m):
            u = U[i * m + j]
            for ip in range(n):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            v = V[jp * p + k]
                            w = W[kp * n + ip]
                            val = np.dot(u, v * w)
                            if abs(val - expected) > tol:
                                return False
    return True

def _snap_to_half_int(arr):
    return np.round(np.asarray(arr, dtype=float) * 2) / 2.0

def _is_half_integer(arr, tol=1e-7):
    arr = np.asarray(arr, dtype=float)
    return np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute the total squared error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            val = np.dot(U[i * m + j], V[jp * p + k] * W[kp * n + ip])
                            err += (val - expected) ** 2
    return err

def _als_step_U(U, V, W, n, m, p, rank):
    """Solve for U given V, W via least squares."""
    new_U = U.copy()
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                new_U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
            except:
                pass
    return new_U

def _als_step_V(U, V, W, n, m, p, rank):
    """Solve for V given U, W via least squares."""
    new_V = V.copy()
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                new_V[j * p + k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
            except:
                pass
    return new_V

def _als_step_W(U, V, W, n, m, p, rank):
    """Solve for W given U, V via least squares."""
    new_W = W.copy()
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                new_W[k * n + i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
            except:
                pass
    return new_W

def _als_solve(n, m, p, rank, U, V, W, max_iter=500, tol=1e-12, deadline=None):
    """Run alternating least squares to minimize tensor error."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        U = _als_step_U(U, V, W, n, m, p, rank)
        V = _als_step_V(U, V, W, n, m, p, rank)
        W = _als_step_W(U, V, W, n, m, p, rank)
        err = _tensor_error(U, V, W, n, m, p)
        if err < tol:
            return U, V, W, True
    return U, V, W, False

def _try_rank_als(n, m, p, rank, n_restarts, deadline, rng):
    """Try to find a rank-r decomposition via ALS with random restarts."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            return None
        
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        U, V, W, converged = _als_solve(n, m, p, rank, U, V, W, max_iter=300, tol=1e-12, deadline=deadline)
        
        if converged:
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                if _is_half_integer(U_s) and _is_half_integer(V_s) and _is_half_integer(W_s):
                    return (U_s, V_s, W_s)
            
            if _verify_exact(U, V, W, n, m, p, tol=1e-6):
                U_s = _snap_to_half_int(U)
                V_s = _snap_to_half_int(V)
                W_s = _snap_to_half_int(W)
                if _verify_exact(U_s, V_s, W_s, n, m, p):
                    return (U_s, V_s, W_s)
        
        err = _tensor_error(U, V, W, n, m, p)
        if err < 1e-6:
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return (U_s, V_s, W_s)
    
    return None

def _strassen_2x2_decomposition():
    """
    Construct the Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7) respectively.
    
    For C = A * B where A, B are 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    Strassen's 7 products:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C[0,0] = m1+m4-m5+m7
    C[0,1] = m3+m5
    C[1,0] = m2+m4
    C[1,1] = m1-m2+m3+m6
    
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    W indices: k*n+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # m1 = (a+d)(e+h)
    # U: a(0)+d(3), V: e(0)+h(3)
    # W: C[0,0](k=0,i=0)=0: +1, C[1,1](k=1,i=1)=3: +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    # U: c(2)+d(3), V: e(0)
    # W: C[1,0](k=0,i=1)=1: +1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # m3 = a(f-h)
    # U: a(0), V: f(1)-h(3)
    # W: C[0,1](k=1,i=0)=2: +1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # m4 = d(g-e)
    # U: d(3), V: g(2)-e(0)
    # W: C[0,0](k=0,i=0)=0: +1, C[1,0](k=0,i=1)=1: +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # m5 = (a+b)h
    # U: a(0)+b(1), V: h(3)
    # W: C[0,0](k=0,i=0)=0: -1, C[0,1](k=1,i=0)=2: +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # m6 = (c-a)(e+f)
    # U: c(2)-a(0), V: e(0)+f(1)
    # W: C[1,1](k=1,i=1)=3: +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    # U: b(1)-d(3), V: g(2)+h(3)
    # W: C[0,0](k=0,i=0)=0: +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _try_strassen_based(n, m, p, deadline):
    """
    Construct a rank-22 decomposition for 2x3 * 3x4 using Strassen.
    
    A is 2x3: A[i,j] with i=0,1; j=0,1,2
    B is 3x4: B[j,k] with j=0,1,2; k=0,1,2,3
    C is 2x4: C[i,k] with i=0,1; k=0,1,2,3
    
    Partition: A = [A1 | a2] where A1 is 2x2 (j=0,1), a2 is 2x1 (j=2)
               B = [B1; b2] where B1 is 2x4 (j=0,1), b2 is 1x4 (j=2)
    
    C = A1*B1 + a2*b2
    
    A1*B1: 2x2 * 2x4. Split B1 = [B1a | B1b] where B1a, B1b are 2x2.
    Use Strassen (rank 7) for each: total rank 14.
    
    a2*b2: 2x1 * 1x4. Need A[0,2]*B[2,k] -> C[0,k] and A[1,2]*B[2,k] -> C[1,k].
    That's 8 rank-1 terms: total rank 8.
    
    Grand total: 14 + 8 = 22.
    
    U shape: (n*m, rank) = (6, 22), indexed by i*m+j
    V shape: (m*p, rank) = (12, 22), indexed by j*p+k
    W shape: (p*n, rank) = (8, 22), indexed by k*n+i
    """
    S_U, S_V, S_W = _strassen_2x2_decomposition()
    
    rank = 22
    U = np.zeros((n * m, rank))  # (6, 22)
    V = np.zeros((m * p, rank))  # (12, 22)
    W = np.zeros((p * n, rank))  # (8, 22)
    
    # U row indices: i*m+j -> (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # V row indices: j*p+k -> (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7, (2,0)=8, (2,1)=9, (2,2)=10, (2,3)=11
    # W row indices: k*n+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5, (3,0)=6, (3,1)=7
    
    # --- Strassen for A1*B1a (B columns k=0,1) ---
    # A1 rows: (0,0)=U[0], (0,1)=U[1], (1,0)=U[3], (1,1)=U[4]
    # B1a rows: (0,0)=V[0], (0,1)=V[1], (1,0)=V[4], (1,1)=V[5]
    # C1a cols: (0,0)=W[0], (0,1)=W[2], (1,0)=W[4], (1,1)=W[6]
    # Wait: W[k*n+i]: C[0,0] is i=0,k=0 -> W[0]; C[0,1] is i=0,k=1 -> W[2]
    # C[1,0] is i=1,k=0 -> W[1]; C[1,1] is i=1,k=1 -> W[3]
    # So C1a: (k=0,i=0)=W[0], (k=1,i=0)=W[2], (k=0,i=1)=W[1], (k=1,i=1)=W[3]
    
    # Strassen W mapping: W_s[k*n+i] where n=2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Our W for C1a: k=0,1; i=0,1 -> W[0], W[1], W[2], W[3]
    # Strassen (0,0)->our(0,0)=W[0], (0,1)->our(0,1)=W[1], (1,0)->our(1,0)=W[2], (1,1)->our(1,1)=W[3]
    
    U[0, 0:7] = S_U[0, :]  # A[0,0]
    U[1, 0:7] = S_U[1, :]  # A[0,1]
    U[3, 0:7] = S_U[2, :]  # A[1,0]
    U[4, 0:7] = S_U[3, :]  # A[1,1]
    
    V[0, 0:7] = S_V[0, :]  # B[0,0]
    V[1, 0:7] = S_V[1, :]  # B[0,1]
    V[4, 0:7] = S_V[2, :]  # B[1,0]
    V[5, 0:7] = S_V[3, :]  # B[1,1]
    
    W[0, 0:7] = S_W[0, :]  # C[0,0] (k=0,i=0)
    W[1, 0:7] = S_W[1, :]  # C[1,0] (k=0,i=1)
    W[2, 0:7] = S_W[2, :]  # C[0,1] (k=1,i=0)
    W[3, 0:7] = S_W[3, :]  # C[1,1] (k=1,i=1)
    
    # --- Strassen for A1*B1b (B columns k=2,3) ---
    # B1b rows: (0,2)=V[2], (0,3)=V[3], (1,2)=V[6], (1,3)=V[7]
    # C1b: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    # W[4]=(k=2,i=0), W[5]=(k=2,i=1), W[6]=(k=3,i=0), W[7]=(k=3,i=1)
    
    U[0, 7:14] = S_U[0, :]
    U[1, 7:14] = S_U[1, :]
    U[3, 7:14] = S_U[2, :]
    U[4, 7:14] = S_U[3, :]
    
    V[2, 7:14] = S_V[0, :]  # B[0,2]
    V[3, 7:14] = S_V[1, :]  # B[0,3]
    V[6, 7:14] = S_V[2, :]  # B[1,2]
    V[7, 7:14] = S_V[3, :]  # B[1,3]
    
    W[4, 7:14] = S_W[0, :]  # C[0,2] (k=2,i=0)
    W[5, 7:14] = S_W[1, :]  # C[1,2] (k=2,i=1)
    W[6, 7:14] = S_W[2, :]  # C[0,3] (k=3,i=0)
    W[7, 7:14] = S_W[3, :]  # C[1,3] (k=3,i=1)
    
    # --- a2*b2: 8 rank-1 terms ---
    # For k=0,1,2,3:
    #   A[0,2]*B[2,k] -> C[0,k]: U[2], V[8+k], W[k*2+0]=W[2k]
    #   A[1,2]*B[2,k] -> C[1,k]: U[5], V[8+k], W[k*2+1]=W[2k+1]
    
    col = 14
    for k in range(4):
        # Top row: A[0,2]*B[2,k] -> C[0,k]
        U[2, col] = 1
        V[8 + k, col] = 1
        W[2 * k, col] = 1  # k*n+i = k*2+0
        col += 1
        # Bottom row: A[1,2]*B[2,k] -> C[1,k]
        U[5, col] = 1
        V[8 + k, col] = 1
        W[2 * k + 1, col] = 1  # k*n+i = k*2+1
        col += 1
    
    if _verify_exact(U, V, W, n, m, p):
        return (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    np.random.seed(42)
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s kill
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_fb, V_fb, W_fb)
    
    # Try Strassen-based construction first (rank 22)
    if time.time() < deadline - 20:
        result = _try_strassen_based(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    # Try ALS from rank 19 upward (known lower bound is ~19)
    for target_rank in range(19, best_rank):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_als(n, m, p, target_rank, n_restarts=30, deadline=deadline - 5, rng=rng)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
