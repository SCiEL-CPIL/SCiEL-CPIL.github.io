"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _solve_rank(r, time_limit=10.0, seed=42):
    """
    Try to find a rank-r decomposition for 3x3x6.
    
    Uses a structured approach:
    1. Fix U and V to be random matrices.
    2. Compute the "contraction" tensor A[i,k,s] = sum_j U[i*m+j,s] * V[j*p+k,s]
    3. Solve for W such that sum_s W[k*n+i,s] * A[i,k,s] = 1 for all (i,k).
    
    The key insight: for fixed U and V, the equations for W are linear and
    decoupled by (i,k) pair. For each (i,k), we need:
        sum_s W[k*n+i, s] * A[i,k,s] = 1
    
    This is a single linear equation in r unknowns (W[k*n+i, 0..r-1]).
    For r >= 1, this is underdetermined (1 equation, r unknowns), so we can
    always find a solution IF A[i,k,:] is not the zero vector.
    
    So the real question is: can we find U and V such that A[i,k,:] != 0 for
    all (i,k)?
    
    For r=1: A[i,k,0] = sum_j U[i*m+j,0] * V[j*p+k,0]. We need this nonzero
    for all 18 (i,k) pairs. This is a system of bilinear equations.
    
    For larger r, it's easier to satisfy.
    """
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    np.random.seed(seed)
    
    for attempt in range(500):
        if time.time() - t0 > time_limit:
            break
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        
        # Compute A[i,k,s] = sum_j U[i*m+j, s] * V[j*p+k, s]
        A = np.zeros((n, p, r))
        for i in range(n):
            for k in range(p):
                for s in range(r):
                    for j in range(m):
                        A[i, k, s] += U[i * m + j, s] * V[j * p + k, s]
        
        # Check that A[i,k,:] is not zero for all (i,k)
        valid = True
        for i in range(n):
            for k in range(p):
                if np.allclose(A[i, k, :], 0, atol=1e-15):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Solve for W: for each (i,k), find W[k*n+i, :] such that
        # W[k*n+i, :] @ A[i,k,:] = 1
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                a_vec = A[i, k, :]
                norm_sq = np.dot(a_vec, a_vec)
                if norm_sq < 1e-15:
                    valid = False
                    break
                # Minimum norm solution: W = a_vec / norm_sq
                W[k * n + i, :] = a_vec / norm_sq
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-9:
            U_list = [[float(U[i, r_idx]) for r_idx in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, r_idx]) for r_idx in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, r_idx]) for r_idx in range(r)] for i in range(p * n)]
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_grad_descent(r, time_limit=15.0, seed=42):
    """
    Try gradient descent to find a rank-r decomposition.
    """
    n, m, p = 3, 3, 6
    t0 = time.time()
    np.random.seed(seed)
    
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(2000):
        if time.time() - t0 > time_limit:
            break
        
        # Compute loss and gradients
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    res = s - 1.0
                    loss += res * res
                    for r_idx in range(r):
                        dU[i * m + j, r_idx] += 2 * res * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        dV[j * p + k, r_idx] += 2 * res * U[i * m + j, r_idx] * W[k * n + i, r_idx]
                        dW[k * n + i, r_idx] += 2 * res * U[i * m + j, r_idx] * V[j * p + k, r_idx]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if loss < 1e-14:
                break
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        lr *= 0.995
    
    if best_loss < 1e-10:
        U_list = [[float(best_U[i, r_idx]) for r_idx in range(r)] for i in range(n * m)]
        V_list = [[float(best_V[i, r_idx]) for r_idx in range(r)] for i in range(m * p)]
        W_list = [[float(best_W[i, r_idx]) for r_idx in range(r)] for i in range(p * n)]
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Try increasingly low ranks
    for r in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 21, 27, 36, 54]:
        if time.time() > deadline - 5:
            break
        
        # Method 1: structured linear solve
        if time.time() < deadline - 10:
            result = _solve_rank(r, time_limit=8.0, seed=r)
            if result is not None:
                return result
        
        # Method 2: gradient descent
        if time.time() < deadline - 10:
            result = _try_grad_descent(r, time_limit=10.0, seed=r)
            if result is not None:
                return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
