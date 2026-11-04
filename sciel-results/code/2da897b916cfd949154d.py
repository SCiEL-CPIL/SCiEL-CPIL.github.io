"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
RANK_MAX = N * M * P  # 18

# Deadline guard: evaluator kills at 150s wall-clock. We reserve margin.
_DEADLINE = time.time() + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for (n,m,p) = (2,3,3).
    The tensor is the canonical multiplication tensor: T[i,j,k] = 1 for all valid indices."""
    return np.ones((N, M, P), dtype=float)


def _compute_residuals(U, V, W):
    """Vectorized computation of residuals for given U, V, W matrices."""
    nm, mp, pn = N*M, M*P, P*N
    rank = U.shape[1]
    
    # Reshape U, V, W to 3D: U[i, j, r], V[j, k, r], W[k, i, r]
    U_3d = U.reshape(N, M, rank)
    V_3d = V.reshape(M, P, rank)
    W_3d = W.reshape(P, N, rank)
    
    # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # full has shape (N, M, P, rank)
    full = U_3d[:, :, None, :] * V_3d[None, :, :, :] * W_3d[None, None, :, :]
    result = np.sum(full, axis=3)  # shape (N, M, P)
    
    return (result - 1.0).flatten()


def _try_rank_decomposition(rank, timeout=None):
    """
    Attempt to find a rank-`rank` decomposition of the multiplication tensor.
    Uses scipy.optimize.least_squares with multiple random restarts.
    """
    from scipy.optimize import least_squares
    
    def _residuals(params, rank):
        """Unpack parameters into U, V, W and compute residuals."""
        nm, mp, pn = N*M, M*P, P*N
        u_size = nm * rank
        v_size = mp * rank
        w_size = pn * rank
        
        U = params[:u_size].reshape(nm, rank)
        V = params[u_size:u_size+v_size].reshape(mp, rank)
        W = params[u_size+v_size:u_size+v_size+w_size].reshape(pn, rank)
        
        return _compute_residuals(U, V, W)
    
    best_result = None
    best_cost = float('inf')
    max_attempts = 20  # More restarts to find a solution
    
    for attempt in range(max_attempts):
        if _remaining() < 5:
            break
        # Use different seeds for diversity
        seed = attempt * 12345 + 67890
        rng = np.random.RandomState(seed)
        
        # Initialize with random values, scaled appropriately
        scale = 0.5
        u0 = rng.randn(N*M, rank) * scale
        v0 = rng.randn(M*P, rank) * scale
        w0 = rng.randn(P*N, rank) * scale
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])
        
        try:
            result = least_squares(
                _residuals, 
                params0, 
                args=(rank,), 
                method='lm', 
                max_nfev=2000,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15
            )
            if result.cost < best_cost:
                best_cost = result.cost
                if result.cost < 1e-12:
                    # Found a solution
                    u_size = N*M*rank
                    v_size = M*P*rank
                    w_size = P*N*rank
                    U = result.x[:u_size].reshape(N*M, rank)
                    V = result.x[u_size:u_size+v_size].reshape(M*P, rank)
                    W = result.x[u_size+v_size:u_size+v_size+w_size].reshape(P*N, rank)
                    
                    # Try to round to nearest 0.5 or 0.25 if close
                    for denom in [2, 4, 8]:
                        U_r = np.round(U * denom) / denom
                        V_r = np.round(V * denom) / denom
                        W_r = np.round(W * denom) / denom
                        
                        # Verify the rounded version
                        res = _compute_residuals(U_r, V_r, W_r)
                        if np.max(np.abs(res)) < 1e-8:
                            return U_r, V_r, W_r
                    
                    # If rounding doesn't work, return the float solution
                    return U, V, W
        except Exception as e:
            pass
    
    return None


def _strassen_2x2x2():
    """
    Returns the rank-7 decomposition for 2x2 * 2x2 -> 2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B01
    U[2, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # r=2: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = 1
    
    # r=3: A11(B00-B01)
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[1, 3] = 1; W[2, 3] = 1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W


def _construct_rank17():
    """
    Construct a rank-17 decomposition for 2x3 * 3x3 -> 2x3.
    
    Strategy:
    - Use Strassen (rank 7) for the top-left 2x2 block: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    - Use direct multiplication (rank 4) for A[0:2,0:2] * B[0:2,2] -> C[0:2,2]
    - Use direct multiplication (rank 6) for A[2,0:3] * B[2,0:3] -> C[0:2,0:3]
    
    Total: 7 + 4 + 6 = 17
    """
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = n*m, m*p, p*n  # 6, 9, 6
    
    U = np.zeros((n_a, 17))
    V = np.zeros((n_b, 17))
    W = np.zeros((n_c, 17))
    
    # --- Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # A indices: i in {0,1}, j in {0,1} -> U index = i*3 + j (rows 0,1,3,4)
    # B indices: j in {0,1}, k in {0,1} -> V index = j*3 + k (rows 0,1,3,4)
    # C indices: k in {0,1}, i in {0,1} -> W index = k*2 + i (rows 0,1,2,3)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Map 2x2 indices to 3x3 positions
    # A row indices: 0->0, 1->1, 2->3, 3->4  (i*3+j for i,j in {0,1})
    a_map = [0, 1, 3, 4]
    # B row indices: 0->0, 1->1, 2->3, 3->4  (j*3+k for j,k in {0,1})
    b_map = [0, 1, 3, 4]
    # C row indices: 0->0, 1->1, 2->2, 3->3  (k*2+i for k,i in {0,1})
    c_map = [0, 1, 2, 3]
    
    for r in range(7):
        for a_old, a_new in enumerate(a_map):
            U[a_new, r] = U_s[a_old, r]
        for b_old, b_new in enumerate(b_map):
            V[b_new, r] = V_s[b_old, r]
        for c_old, c_new in enumerate(c_map):
            W[c_new, r] = W_s[c_old, r]
    
    # --- Part 2: Direct for A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2] ---
    # A[i, j] for i in {0,1}, j in {0,1} -> U index = i*3 + j
    # B[j, 2] for j in {0,1} -> V index = j*3 + 2 = 2, 5
    # C[2, i] for i in {0,1} -> W index = 2*2 + i = 4, 5
    
    col = 7
    for i in range(2):
        for j in range(2):
            U[i*3 + j, col] = 1
            V[j*3 + 2, col] = 1
            W[4 + i, col] = 1
            col += 1
    
    # --- Part 3: Direct for A[2, 0:3] * B[2, 0:3] -> C[0:2, 0:3] ---
    # A[2, j] for j in {0,1,2} -> U index = 2*3 + j = 6, 7, 8
    # B[2, k] for k in {0,1,2} -> V index = 2*3 + k = 6, 7, 8
    # C[k, i] for i in {0,1}, k in {0,1,2} -> W index = k*2 + i
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[6 + j, col] = 1
                V[6 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
    
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify the decomposition exactly reproduces the multiplication tensor."""
    T = _build_tensor()
    res = _compute_residuals(U, V, W)
    return np.max(np.abs(res)) < tol


def _try_rank16_search(start_time, deadline):
    """
    Try to find a rank-16 decomposition by perturbing the rank-17 solution.
    Uses least_squares optimization starting from the rank-17 solution with one column removed
    or with random perturbations.
    """
    from scipy.optimize import least_squares
    
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 16
    
    U17, V17, W17 = _construct_rank17()
    
    def _residuals(params, rank):
        u_size = n_a * rank
        v_size = n_b * rank
        w_size = n_c * rank
        U = params[:u_size].reshape(n_a, rank)
        V = params[u_size:u_size+v_size].reshape(n_b, rank)
        W = params[u_size+v_size:u_size+v_size+w_size].reshape(n_c, rank)
        return _compute_residuals(U, V, W)
    
    best_cost = float('inf')
    best_params = None
    
    # Strategy 1: Drop each column of the rank-17 solution and optimize
    for drop_col in range(17):
        if time.time() > deadline - 15:
            break
        
        keep = [r for r in range(17) if r != drop_col]
        U0 = U17[:, keep].copy()
        V0 = V17[:, keep].copy()
        W0 = W17[:, keep].copy()
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(
                _residuals,
                params0,
                args=(rank,),
                method='lm',
                max_nfev=1000,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-12:
                    u_size = n_a * rank
                    v_size = n_b * rank
                    U = result.x[:u_size].reshape(n_a, rank)
                    V = result.x[u_size:u_size+v_size].reshape(n_b, rank)
                    W = result.x[u_size+v_size:u_size+v_size+w_size].reshape(n_c, rank)
                    return U, V, W
        except Exception:
            pass
    
    # Strategy 2: Random perturbations of the full rank-17, then project to rank 16
    rng = np.random.default_rng(123)
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        
        U0 = U17[:, :16].copy() + rng.normal(0, 0.1, (n_a, 16))
        V0 = V17[:, :16].copy() + rng.normal(0, 0.1, (n_b, 16))
        W0 = W17[:, :16].copy() + rng.normal(0, 0.1, (n_c, 16))
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(
                _residuals,
                params0,
                args=(rank,),
                method='lm',
                max_nfev=1000,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-12:
                    u_size = n_a * rank
                    v_size = n_b * rank
                    U = result.x[:u_size].reshape(n_a, rank)
                    V = result.x[u_size:u_size+v_size].reshape(n_b, rank)
                    W = result.x[u_size+v_size:u_size+v_size+w_size].reshape(n_c, rank)
                    return U, V, W
        except Exception:
            pass
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # First, try to find a rank-16 decomposition
    if _remaining() > 30:
        result_16 = _try_rank16_search(start_time, deadline)
        if result_16 is not None:
            U, V, W = result_16
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-17 solution
    U17, V17, W17 = _construct_rank17()
    if _verify(U17, V17, W17):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Fallback: schoolbook decomposition (rank 18)
    rank = n * m * p
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
