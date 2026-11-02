import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return np.allclose(T_computed, T_target, atol=tol)

def _schoolbook():
    """Return schoolbook decomposition with rank n*m*p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank34():
    """
    Construct a rank-34 decomposition.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] using Strassen
    S_U, S_V, S_W = _strassen_2x2x2()
    
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U1[i * m + j, r] = S_U[i * 2 + j, r]
                V1[i * p + j, r] = S_V[i * 2 + j, r]
                W1[j * n + i, r] = S_W[i * 2 + j, r]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Part 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] using Strassen
    S_U2, S_V2, S_W2 = _strassen_2x2x2()
    
    U2 = np.zeros((n * m, 7))
    V2 = np.zeros((m * p, 7))
    W2 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U2[i * m + j, r] = S_U2[i * 2 + j, r]
                V2[i * p + (j + 2), r] = S_V2[i * 2 + j, r]
                W2[(j + 2) * n + i, r] = S_W2[i * 2 + j, r]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Part 3: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    U3 = np.zeros((n * m, 8))
    V3 = np.zeros((m * p, 8))
    W3 = np.zeros((p * n, 8))
    
    col = 0
    for i in range(2):
        for k in range(4):
            U3[i * m + 2, col] = 1
            V3[2 * p + k, col] = 1
            W3[k * n + i, col] = 1
            col += 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Part 4: A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]
    U4 = np.zeros((n * m, 12))
    V4 = np.zeros((m * p, 12))
    W4 = np.zeros((p * n, 12))
    
    col = 0
    for j in range(3):
        for k in range(4):
            U4[2 * m + j, col] = 1
            V4[j * p + k, col] = 1
            W4[k * n + 2, col] = 1
            col += 1
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _construct_rank22():
    """
    Construct a rank-22 decomposition using subspace partitioning:
    - 2x2x3 block (rows 0-1, cols 0-2) with Strassen rank 7
    - 2x2x1 block (rows 0-1, col 3) with schoolbook rank 4
    - 1x3x4 block (row 2) with schoolbook rank 12
    Total: 7 + 4 + 12 = 23... let me recalculate.
    
    Actually, let me try a different partition:
    - 2x2x2 block (rows 0-1, cols 0-1) with Strassen rank 7
    - 2x2x2 block (rows 0-1, cols 2-3) with Strassen rank 7
    - 1x3x4 block (row 2) with schoolbook rank 12
    Total: 7 + 7 + 12 = 26. Not 22.
    
    Let me try:
    - 2x3x4 block (rows 0-1) decomposed into:
      - 2x2x2 (cols 0-1) Strassen rank 7
      - 2x2x2 (cols 2-3) Strassen rank 7
      Total for rows 0-1: 14
    - 1x3x4 block (row 2) schoolbook rank 12
    Total: 14 + 12 = 26.
    
    Hmm, the directive says rank 22. Let me try a different approach.
    
    Actually, re-reading the directive: "partition the 3x3x4 multiplication tensor into a direct sum of a 2x2x3 block (rank 12) and a 1x3x4 block (rank 12), then apply Strassen's algorithm to the 2x2x3 block to reduce its rank to 10"
    
    So: 2x2x3 + 1x3x4 = 10 + 12 = 22.
    
    2x2x3 means: A is 2x2, B is 2x3, C is 2x2? No, that doesn't work dimensionally.
    
    Wait, the tensor is for multiplying (n x m) by (m x p) = (n x p).
    So 3x3 * 3x4 = 3x4.
    
    A "2x2x3 block" might mean: rows 0-1 of A, cols 0-1 of B (which are rows 0-1 of B), producing rows 0-1, cols 0-1 of C. That's 2x2x2.
    
    I think the directive might be using different notation. Let me just try to construct a valid low-rank decomposition.
    
    Let me try:
    - Block 1: A[0:2, 0:3] * B[0:3, 0:3] -> C[0:2, 0:3]
      This is 2x3 * 3x3 = 2x3. Can use Strassen-like or just do it as 2x2x2 + 2x1x3.
    
    This is getting complicated. Let me just keep the rank-34 solution and focus on the rank-33 search with better initialization.
    """
    # For now, return the rank-34 solution as a placeholder
    return _construct_rank34()

def _flatten(U, V, W):
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten(x, n, m, p, R):
    size_U = n * m * R
    size_V = m * p * R
    U = x[:size_U].reshape(n * m, R)
    V = x[size_U:size_U + size_V].reshape(m * p, R)
    W = x[size_U + size_V:].reshape(p * n, R)
    return U, V, W

def _residual(x, n, m, p, R):
    U, V, W = _unflatten(x, n, m, p, R)
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return (T_computed - T_target).flatten()

def _residual_jac(x, n, m, p, R):
    """Compute Jacobian of the residual for faster optimization."""
    U, V, W = _unflatten(x, n, m, p, R)
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    # Jacobian computation would be complex; use numerical approximation
    # For now, return None to let scipy use numerical Jacobian
    return None

def _try_rank(R, start_time, deadline, seed=0, max_nfev=2000):
    np.random.seed(seed)
    # Initialize randomly with better scaling
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Scale to reasonable magnitude
    scale = np.sqrt(R)
    U /= scale
    V /= scale
    W /= scale
    
    x0 = _flatten(U, V, W)
    
    try:
        result = scipy.optimize.least_squares(
            _residual, x0, args=(n, m, p, R), 
            max_nfev=max_nfev,
            xtol=1e-14, ftol=1e-14, gtol=1e-14,
            method='trf'
        )
        
        if result.success or np.linalg.norm(result.fun) < 1e-8:
            U_opt, V_opt, W_opt = _unflatten(result.x, n, m, p, R)
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify_decomposition(U_round, V_round, W_round, n, m, p, tol=1e-10):
                return U_round, V_round, W_round
                
    except Exception:
        pass
    return None

def _try_rank_structured(R, start_time, deadline, seed=0):
    """Try rank R with structured initialization based on the rank-34 solution."""
    np.random.seed(seed)
    
    # Start from the rank-34 solution and truncate
    U34, V34, W34 = _construct_rank34()
    
    if R < 34:
        # Use first R columns
        U = U34[:, :R].copy()
        V = V34[:, :R].copy()
        W = W34[:, :R].copy()
        
        # Add small random perturbation to escape local minima
        U += np.random.randn(*U.shape) * 0.01
        V += np.random.randn(*V.shape) * 0.01
        W += np.random.randn(*W.shape) * 0.01
    else:
        U = U34.copy()
        V = V34.copy()
        W = W34.copy()
        # Add extra random columns
        extra_U = np.random.randn(n * m, R - 34)
        extra_V = np.random.randn(m * p, R - 34)
        extra_W = np.random.randn(p * n, R - 34)
        U = np.hstack([U, extra_U])
        V = np.hstack([V, extra_V])
        W = np.hstack([W, extra_W])
    
    x0 = _flatten(U, V, W)
    
    try:
        result = scipy.optimize.least_squares(
            _residual, x0, args=(n, m, p, R), 
            max_nfev=1000,
            xtol=1e-14, ftol=1e-14, gtol=1e-14,
            method='trf'
        )
        
        if result.success or np.linalg.norm(result.fun) < 1e-8:
            U_opt, V_opt, W_opt = _unflatten(result.x, n, m, p, R)
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify_decomposition(U_round, V_round, W_round, n, m, p, tol=1e-10):
                return U_round, V_round, W_round
                
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Get the rank-34 solution
    try:
        U34, V34, W34 = _construct_rank34()
        if _verify_decomposition(U34, V34, W34, n, m, p, tol=1e-10):
            best_U, best_V, best_W = U34, V34, W34
            best_rank = 34
        else:
            U34, V34, W34 = _schoolbook()
            best_U, best_V, best_W = U34, V34, W34
            best_rank = 36
    except Exception:
        U34, V34, W34 = _schoolbook()
        best_U, best_V, best_W = U34, V34, W34
        best_rank = 36

    # Try to find a rank-33 solution with structured initialization
    if time.time() < deadline - 30:
        for seed in [0, 1, 2, 3, 4]:
            if time.time() > deadline - 10:
                break
            res = _try_rank_structured(33, start_time, deadline, seed)
            if res is not None:
                U33, V33, W33 = res
                best_U, best_V, best_W = U33, V33, W33
                best_rank = 33
                break
    
    # If still at 34, try random initialization
    if best_rank == 34 and time.time() < deadline - 20:
        for seed in [10, 20, 30]:
            if time.time() > deadline - 10:
                break
            res = _try_rank(33, start_time, deadline, seed, max_nfev=1500)
            if res is not None:
                U33, V33, W33 = res
                best_U, best_V, best_W = U33, V33, W33
                best_rank = 33
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
