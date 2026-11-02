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

def _try_rank(R, start_time, deadline, seed=0):
    np.random.seed(seed)
    # Initialize randomly
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Scale to reasonable magnitude
    U /= np.sqrt(R)
    V /= np.sqrt(R)
    W /= np.sqrt(R)
    
    x0 = _flatten(U, V, W)
    
    try:
        result = scipy.optimize.least_squares(
            _residual, x0, args=(n, m, p, R), 
            max_nfev=500, # Limit function evaluations
            xtol=1e-15, ftol=1e-15, gtol=1e-15
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
    deadline = start_time + 110  # Safety margin
    
    # Get the rank-34 solution
    try:
        U34, V34, W34 = _construct_rank34()
        if _verify_decomposition(U34, V34, W34, n, m, p, tol=1e-10):
            best_U, best_V, best_W = U34, V34, W34
            best_rank = 34
        else:
            # Fallback to schoolbook if rank-34 fails verification (should not happen)
            U34, V34, W34 = _schoolbook()
            best_U, best_V, best_W = U34, V34, W34
            best_rank = 36
    except Exception:
        U34, V34, W34 = _schoolbook()
        best_U, best_V, best_W = U34, V34, W34
        best_rank = 36

    # Try to find a rank-33 solution
    # We have about 100 seconds left.
    # Let's try 2 restarts with different seeds.
    if time.time() < deadline - 20:
        for seed in [0, 1]:
            if time.time() > deadline - 10:
                break
            res = _try_rank(33, start_time, deadline, seed)
            if res is not None:
                U33, V33, W33 = res
                best_U, best_V, best_W = U33, V33, W33
                best_rank = 33
                break # Found a better one, stop searching for even lower? 
                       # Actually, we could try rank 32, but it's unlikely.
                       # Let's stick to 33 for now.

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
