import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    return time.time() > deadline

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, each shape (4, 7)."""
    U = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=float).T  # (4, 7)
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float).T  # (4, 7)
    W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, -1, 1, 0],
    ], dtype=float).T  # (4, 7)
    # Fix: W should have 7 columns
    W = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)
    return U, V, W

def _build_structured_init(n, m, p, target_rank, seed=42):
    """Build a structured initial guess using Kronecker products of Strassen blocks."""
    np.random.seed(seed)
    
    # Pad dimensions to multiples of 2 for block structure
    # 3 -> 4, 4 -> 4, 5 -> 6
    n_pad = ((n + 1) // 2) * 2
    m_pad = ((m + 1) // 2) * 2
    p_pad = ((p + 1) // 2) * 2
    
    # Build padded Strassen-based decomposition
    # For padded dimensions n_pad x m_pad x p_pad
    # Use 2x2 blocks: n_pad/2 * m_pad/2 * p_pad/2 blocks, each rank 7
    # Total padded rank = (n_pad/2) * (m_pad/2) * (p_pad/2) * 7
    
    blocks_n = n_pad // 2
    blocks_m = m_pad // 2
    blocks_p = p_pad // 2
    padded_rank = blocks_n * blocks_m * blocks_p * 7
    
    # Build padded U, V, W
    U_pad = np.zeros((n_pad * m_pad, padded_rank))
    V_pad = np.zeros((m_pad * p_pad, padded_rank))
    W_pad = np.zeros((p_pad * n_pad, padded_rank))
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    col = 0
    for bi in range(blocks_n):
        for bj in range(blocks_m):
            for bk in range(blocks_p):
                for r in range(7):
                    c = col + r
                    # U block: rows (bi*2 + 0..1) * m_pad + (bj*2 + 0..1)
                    for di in range(2):
                        for dj in range(2):
                            U_pad[(bi*2 + di) * m_pad + (bj*2 + dj), c] = U_s[di*2+dj, r]
                    # V block: rows (bj*2 + 0..1) * p_pad + (bk*2 + 0..1)
                    for dj in range(2):
                        for dk in range(2):
                            V_pad[(bj*2 + dj) * p_pad + (bk*2 + dk), c] = V_s[dj*2+dk, r]
                    # W block: rows (bk*2 + 0..1) * n_pad + (bi*2 + 0..1)
                    for dk in range(2):
                        for di in range(2):
                            W_pad[(bk*2 + dk) * n_pad + (bi*2 + di), c] = W_s[dk*2+di, r]
                col += 7
    
    # Trim to actual dimensions
    U_actual = U_pad[:n*m, :]
    V_actual = V_pad[:m*p, :]
    W_actual = W_pad[:p*n, :]
    
    # If padded_rank > target_rank, project down
    if padded_rank > target_rank:
        # Use SVD to project
        U_svd = np.linalg.svd(U_actual, full_matrices=False)
        V_svd = np.linalg.svd(V_actual, full_matrices=False)
        W_svd = np.linalg.svd(W_actual, full_matrices=False)
        
        # Keep top target_rank components
        U_init = U_actual @ np.eye(padded_rank)[:, :target_rank]
        V_init = V_actual @ np.eye(padded_rank)[:, :target_rank]
        W_init = W_actual @ np.eye(padded_rank)[:, :target_rank]
    elif padded_rank < target_rank:
        # Pad with random
        U_init = np.hstack([U_actual, np.random.randn(n*m, target_rank - padded_rank) * 0.1])
        V_init = np.hstack([V_actual, np.random.randn(m*p, target_rank - padded_rank) * 0.1])
        W_init = np.hstack([W_actual, np.random.randn(p*n, target_rank - padded_rank) * 0.1])
    else:
        U_init = U_actual.copy()
        V_init = V_actual.copy()
        W_init = W_actual.copy()
    
    return U_init, V_init, W_init

def _als_refine(U, V, W, n, m, p, max_iters=100, deadline=None):
    """Vectorized ALS refinement for the tensor decomposition."""
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve least squares
        # min sum_k (1 - sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])^2
        for i in range(n):
            for j in range(m):
                # Build A matrix: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = V[j*p:(j+1)*p, :] * W[i:(i+1)*p, :].T  # shape (p, rank)
                # Wait, W is indexed by (k*n + i), so for fixed i, W[k*n+i, c] for k=0..p-1
                A = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(rank)] for k in range(p)])
                b = np.ones(p)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                A = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(rank)] for i in range(n)])
                b = np.ones(n)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                A = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(rank)] for j in range(m)])
                b = np.ones(m)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (1.0 - val) ** 2
        
        if err < 1e-14:
            break
    
    return U, V, W, err

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                pass  # Not needed, use tensor form
    
    # Use tensor form
    for i in range(n):
        for k in range(p):
            total = 0.0
            for c in range(U.shape[1]):
                u_sum = sum(U[i*m+j, c] * A[i,j] for j in range(m))
                v_sum = sum(V[j*p+k, c] * B[j,k] for j in range(m))
                total += W[k*n+i, c] * u_sum * v_sum
            C_computed[i, k] = total
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 110
    
    # Schoolbook fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from low to high
    for target_rank in range(45, 61):
        if time.time() > deadline - 15:
            break
        
        trials = 3 if target_rank <= 50 else 2
        for trial in range(trials):
            if time.time() > deadline - 10:
                break
            
            U_init, V_init, W_init = _build_structured_init(n, m, p, target_rank, seed=trial*100+target_rank)
            
            U_r, V_r, W_r, err = _als_refine(U_init, V_init, W_init, n, m, p, max_iters=80, deadline=deadline-5)
            
            if err < 1e-12:
                if _verify(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also try random init
            np.random.seed(trial * 7 + target_rank)
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            U_r2, V_r2, W_r2, err2 = _als_refine(U_rand, V_rand, W_rand, n, m, p, max_iters=60, deadline=deadline-5)
            
            if err2 < 1e-12:
                if _verify(U_r2, V_r2, W_r2, n, m, p):
                    return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
    
    return best
