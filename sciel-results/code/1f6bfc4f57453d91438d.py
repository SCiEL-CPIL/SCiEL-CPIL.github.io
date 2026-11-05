"""Matrix multiplication 3x3x5 rank-minimal decomposition via ALS search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS

def _verify(U, V, W, trials=10, seed=0):
    """Return True if decomposition exactly computes A@B for random integer matrices."""
    rng = np.random.default_rng(seed)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        C_true = A @ B
        C = np.zeros((n, p), dtype=float)
        for r in range(rank):
            ua = np.sum(U[:, r] * A.reshape(-1))
            vb = np.sum(V[:, r] * B.reshape(-1))
            # W[:, r] is length p*n, indexed as W[k*n + i]
            C += (ua * vb) * W[:, r].reshape(p, n).T
        if not np.allclose(C, C_true, atol=1e-6):
            return False
    return True

def _schoolbook():
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer."""
    return (2 * U).round() / 2.0, (2 * V).round() / 2.0, (2 * W).round() / 2.0

def _als_decomposition(rank, rng, iters=200, deadline=None):
    """
    Use Alternating Least Squares to find a rank-`rank` decomposition.
    Returns (U, V, W) or None if time exceeded.
    """
    # Initialize
    U = rng.normal(size=(n * m, rank))
    V = rng.normal(size=(m * p, rank))
    W = rng.normal(size=(p * n, rank))
    
    # Precompute index mappings for speed
    # U index: i*m+j -> (i,j)
    # V index: j*p+k -> (j,k)
    # W index: k*n+i -> (k,i)
    
    # Target tensor T = 1
    T = np.ones((n, m, p))
    
    # We will maintain the residual R = T - sum(T_r)
    # Initially R = T
    R = T.copy()
    
    # Helper to compute T_r from u, v, w
    def compute_T_r(u, v, w):
        # u: shape (n*m,), v: shape (m*p,), w: shape (p*n,)
        # T_r[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        Tr = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_val = u[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    Tr[i,j,k] = u_val * v[j*p+k] * w[k*n+i]
        return Tr

    # Helper to build system matrix for U update
    # M_U[a, idx] such that vec(T_r) = M_U @ u
    # idx is linear index of (i,j,k)
    # a is index of u (i*m+j)
    def build_M_U(v, w):
        M = np.zeros((n*m*p, n*m))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    a = i*m + j
                    M[idx, a] = v[j*p+k] * w[k*n+i]
        return M

    # Helper to build system matrix for V update
    def build_M_V(u, w):
        M = np.zeros((n*m*p, m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    b = j*p + k
                    M[idx, b] = u[i*m+j] * w[k*n+i]
        return M

    # Helper to build system matrix for W update
    def build_M_W(u, v):
        M = np.zeros((n*m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    c = k*n + i
                    M[idx, c] = u[i*m+j] * v[j*p+k]
        return M

    M_U = build_M_U(V[:, 0], W[:, 0]) # Placeholder, will be updated
    M_V = build_M_V(U[:, 0], W[:, 0])
    M_W = build_M_W(U[:, 0], V[:, 0])
    
    # Precompute matrices for all r? No, they depend on current V,W.
    # We will recompute M for each r in each iteration.
    
    for it in range(iters):
        if deadline and time.time() > deadline:
            return None
            
        # Update U
        M_U_list = []
        for r in range(rank):
            M_U_list.append(build_M_U(V[:, r], W[:, r]))
            
        for r in range(rank):
            # Solve M_U[r] @ u = R_vec
            R_vec = R.flatten()
            # Least squares
            u_new, _, _, _ = np.linalg.lstsq(M_U_list[r], R_vec, rcond=None)
            
            # Update R: R = R + T_old - T_new
            T_old = compute_T_r(U[:, r], V[:, r], W[:, r])
            T_new = compute_T_r(u_new, V[:, r], W[:, r])
            R += T_old - T_new
            U[:, r] = u_new
            
        # Update V
        M_V_list = []
        for r in range(rank):
            M_V_list.append(build_M_V(U[:, r], W[:, r]))
            
        for r in range(rank):
            R_vec = R.flatten()
            v_new, _, _, _ = np.linalg.lstsq(M_V_list[r], R_vec, rcond=None)
            
            T_old = compute_T_r(U[:, r], V[:, r], W[:, r])
            T_new = compute_T_r(U[:, r], v_new, W[:, r])
            R += T_old - T_new
            V[:, r] = v_new
            
        # Update W
        M_W_list = []
        for r in range(rank):
            M_W_list.append(build_M_W(U[:, r], V[:, r]))
            
        for r in range(range(rank)):
            R_vec = R.flatten()
            w_new, _, _, _ = np.linalg.lstsq(M_W_list[r], R_vec, rcond=None)
            
            T_old = compute_T_r(U[:, r], V[:, r], W[:, r])
            T_new = compute_T_r(U[:, r], V[:, r], w_new)
            R += T_old - T_new
            W[:, r] = w_new
            
    # Final check
    err = np.linalg.norm(R)
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(20240517)
    
    # Try ranks from low to high
    # Known rank for 3x3x5 is 15.
    for rank in range(15, 45):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            res = _als_decomposition(rank, rng, iters=100, deadline=deadline)
            if res is not None:
                U, V, W = res
                U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
                if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist()):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        # If not found, try next rank
        
    # Fallback: schoolbook rank 45
    return _schoolbook()
