"""Tensor decomposition for 2x3x6 matrix multiplication; SCiEL evolves the EVOLVE-BLOCK."""
import time
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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    # Also verify that all other entries are 0
    T = _build_tensor(n, m, p)
    recon = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
    if np.max(np.abs(recon - T)) > tol:
        return False
    return True


def _round_to_halfint(arr, tol=1e-6):
    """Round array entries to nearest half-integer if close enough."""
    result = np.array(arr, dtype=float)
    # Check if entries are close to half-integers
    scaled = 2 * result
    nearest = np.round(scaled)
    if np.max(np.abs(scaled - nearest)) < tol:
        return (nearest / 2.0).tolist()
    return result.tolist()


def _gradient_polish(U, V, W, n, m, p, deadline, max_iters=2000, lr_init=0.01):
    """Polish a near-solution using gradient descent to achieve exact precision."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = lr_init
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Compute loss and gradients using einsum
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        T = _build_tensor(n, m, p)
        err = recon - T
        loss = np.sum(err * err)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-20:
            break
        
        # Compute gradients
        # dL/dU[i,c] = 2 * sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
        grad_U = 2 * np.einsum('ijk,jc,kc->ic', err, V, W)
        grad_V = 2 * np.einsum('ijk,ic,kc->jc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ic,jc->kc', err, U, V)
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.995
        
        # Periodic restart with smaller lr if stuck
        if iteration > 200 and loss > best_loss * 1.5:
            lr *= 0.5
    
    if best_loss < 1e-12:
        U_list = _round_to_halfint(best_U)
        V_list = _round_to_halfint(best_V)
        W_list = _round_to_halfint(best_W)
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-5):
            return U_list, V_list, W_list
    
    return None


def _als_decompose(T, n, m, p, rank, deadline, max_sweeps=200, tol=1e-12, seed=0):
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
        if time.time() > deadline - 15.0:
            break
        
        # Update U: fix V, W
        # T1[i, (j,k)] = T[i,j,k], A[(j,k), c] = V[j,c]*W[k,c]
        V_kr_W = (V[:, :, None] * W[None, :, :]).reshape(-1, rank)
        T1 = T.reshape(dim1, -1)
        try:
            U_new, _, _, _ = np.linalg.lstsq(V_kr_W, T1.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update V: fix U, W
        U_kr_W = (U[:, :, None] * W[None, :, :]).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(U_kr_W, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update W: fix U, V
        U_kr_V = (U[:, :, None] * V[None, :, :]).reshape(-1, rank)
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
    if best_loss < 1e-4:
        polished = _gradient_polish(best_U, best_V, best_W, n, m, p, deadline, max_iters=1000, lr_init=0.005)
        if polished is not None:
            return {"U": polished[0], "V": polished[1], "W": polished[2]}
    
    if best_loss < 1e-8:
        U_list = _round_to_halfint(best_U)
        V_list = _round_to_halfint(best_V)
        W_list = _round_to_halfint(best_W)
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-5):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_rank(T, n, m, p, rank, deadline, max_trials=30):
    """Try multiple random initializations for a given rank."""
    for seed in range(max_trials):
        if time.time() > deadline - 20.0:
            break
        result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=150, seed=seed)
        if result is not None:
            return result
    return None


def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Try to exploit Kronecker product structure.
    For 2x3x6: note that 6 = 2*3, 3 = 1*3, 2 = 2*1
    We can think of this as a tensor product of smaller problems.
    
    The key insight: if we can decompose the 2x3x6 problem using
    Kronecker products of smaller matrices, we might get a lower rank.
    
    For now, this is a placeholder that will be enhanced.
    """
    T = _build_tensor(n, m, p)
    
    # Try to find a decomposition using structured initialization
    # based on Kronecker products
    rank = max(n * m, m * p, p * n)  # Start from lower bound
    
    # Try ranks from lower bound up
    for r in range(rank, n * m * p + 1):
        if time.time() > deadline - 25.0:
            break
        
        # Try structured initialization
        for seed in range(15):
            if time.time() > deadline - 20.0:
                break
            
            rng = np.random.RandomState(seed)
            
            # Initialize with Kronecker-like structure
            # Split the rank into factors if possible
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Try ALS
            result = _als_decompose(T, n, m, p, r, deadline, max_sweeps=100, seed=seed)
            if result is not None:
                return result
    
    return None


def _try_hankel_decomposition(n, m, p, deadline):
    """
    Try a Hankel-like structured decomposition.
    The matrix multiplication tensor has a special structure where
    T[i*m+j, j*p+k, k*n+i] = 1.
    
    We can try to exploit this by using structured basis vectors.
    """
    T = _build_tensor(n, m, p)
    
    # The lower bound for rank is max(n*m, m*p, p*n)
    lower_bound = max(n * m, m * p, p * n)
    
    # Try from lower bound up to schoolbook
    for rank in range(lower_bound, n * m * p + 1):
        if time.time() > deadline - 30.0:
            break
        
        # Try multiple random initializations with different structures
        for seed in range(20):
            if time.time() > deadline - 25.0:
                break
            
            result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=150, seed=seed)
            if result is not None:
                return result
    
    return None


def _try_random_decomposition(n, m, p, deadline):
    """Try random decompositions with various initializations."""
    T = _build_tensor(n, m, p)
    
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try from lower bound up to schoolbook
    for rank in range(lower_bound, n * m * p + 1):
        if time.time() > deadline - 40.0:
            break
        
        # Try multiple random initializations
        for seed in range(25):
            if time.time() > deadline - 35.0:
                break
            
            result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=200, seed=seed)
            if result is not None:
                return result
    
    return None


def _try_lattice_decomposition(n, m, p, deadline):
    """
    Try a lattice-based decomposition.
    For 2x3x6, we can try to use the fact that the indices have
    a lattice structure.
    """
    T = _build_tensor(n, m, p)
    
    # Try to find a decomposition using integer/half-integer entries
    # by starting from random initializations and rounding
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, n * m * p + 1):
        if time.time() > deadline - 50.0:
            break
        
        for seed in range(20):
            if time.time() > deadline - 45.0:
                break
            
            result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=150, seed=seed)
            if result is not None:
                return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # First try the most promising approaches
    # 1. Try Kronecker-like structured decomposition
    result = _try_kronecker_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # 2. Try Hankel-like structured decomposition
    result = _try_hankel_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # 3. Try random decompositions
    result = _try_random_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # 4. Try lattice-based decomposition
    result = _try_lattice_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook decomposition
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
