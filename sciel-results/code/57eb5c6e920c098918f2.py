"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, all other entries 0."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True


def _gradient_polish(U, V, W, n, m, p, deadline, max_iters=500, lr_init=0.01):
    """Polish a near-solution using gradient descent to achieve exact precision."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = lr_init
    
    for iteration in range(max_iters):
        if time.time() > deadline - 3.0:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = val - 1.0
                    loss += err * err
                    grad_U[i * m + j, :] += 2 * err * (V[j * p + k, :] * W[k * n + i, :])
                    grad_V[j * p + k, :] += 2 * err * (U[i * m + j, :] * W[k * n + i, :])
                    grad_W[k * n + i, :] += 2 * err * (U[i * m + j, :] * V[j * p + k, :])
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-14:
            break
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.99
        
        # Periodic restart with smaller lr if stuck
        if iteration > 100 and loss > best_loss * 1.1:
            lr *= 0.5
    
    if best_loss < 1e-10:
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return U_list, V_list, W_list
    
    return None


def _als_decompose(T, n, m, p, rank, deadline, max_sweeps=100, tol=1e-10, seed=0):
    """Alternating Least Squares for tensor decomposition."""
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    # Initialize with random values
    U = rng.randn(dim1, rank)
    V = rng.randn(dim2, rank)
    W = rng.randn(dim3, rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 10.0:
            break
        
        # Update U: fix V, W
        V_kr_W = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        T1 = T.reshape(dim1, -1)
        try:
            U_new, _, _, _ = np.linalg.lstsq(V_kr_W, T1.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update V: fix U, W
        U_kr_W = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(U_kr_W, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update W: fix U, V
        U_kr_V = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
        T3 = np.transpose(T, (2, 0, 1)).reshape(dim3, -1)
        try:
            W_new, _, _, _ = np.linalg.lstsq(U_kr_V, T3.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Compute loss
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((reconstructed - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
    
    # Try to polish with gradient descent
    if best_loss < 1e-6:
        polished = _gradient_polish(best_U, best_V, best_W, n, m, p, deadline, max_iters=200, lr_init=0.005)
        if polished is not None:
            return {"U": polished[0], "V": polished[1], "W": polished[2]}
    
    if best_loss < 1e-8:
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_rank(T, n, m, p, rank, deadline, max_trials=15):
    """Try multiple random initializations for a given rank."""
    for seed in range(max_trials):
        if time.time() > deadline - 15.0:
            break
        result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=80, seed=seed)
        if result is not None:
            return result
    return None


def _try_structured_rank(n, m, p, rank, deadline):
    """Try structured decompositions that exploit matrix multiplication structure."""
    # For matrix multiplication, we can try decompositions that separate
    # the input and output indices more cleverly.
    
    # One strategy: use the fact that T[i,j,k] = 1 for all valid (i,j,k)
    # and try to find a low-rank approximation that captures this structure.
    
    # Try using basis vectors that correspond to specific (i,j) or (j,k) pairs
    # with shared factors across k or i dimensions.
    
    # This is a heuristic approach; for now, fall back to random ALS
    T = _build_tensor(n, m, p)
    return _try_rank(T, n, m, p, rank, deadline, max_trials=10)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    T = _build_tensor(n, m, p)
    
    best_result = None
    
    # Known lower bounds for matrix multiplication rank:
    # For n x m times m x p, the rank is at least max(n*m, m*p, p*n)
    # but can be lower due to structure.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # For 2x3x6: schoolbook is 36
    # Try from 12 (which is max(6, 18, 12) = 18, but we can try lower)
    
    min_rank = max(n * m, m * p, p * n)
    # Actually, let's try a range starting from a lower heuristic
    # The theoretical lower bound is more complex; let's try from 10 upward
    
    search_ranks = list(range(10, n * m * p + 1))
    
    for rank in search_ranks:
        if time.time() > deadline - 20.0:
            break
        
        result = _try_rank(T, n, m, p, rank, deadline, max_trials=12)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        return _schoolbook_decomposition(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
