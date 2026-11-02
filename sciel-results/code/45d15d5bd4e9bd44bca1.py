"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    return np.ones((n, m, p))

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.all(np.abs(T - 1.0) < tol)

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - target."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return T - 1.0

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _is_half_integer(x, tol=1e-8):
    """Check if all entries are half-integers."""
    return np.all(np.abs(x * 2 - np.round(x * 2)) < tol)

def _gradient_step(U, V, W, n, m, p, lr=0.01):
    """
    Perform a gradient descent step to reduce the residual.
    The residual is R = einsum('ijr,jkr,kir->ijk', U, V, W) - 1
    We minimize ||R||^2 = sum_{i,j,k} R[i,j,k]^2
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute residual
    R = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r) - 1.0
    
    # Gradient with respect to U_r[i,j,r]:
    # dR[i,j,k]/dU_r[i,j,r] = V_r[j,k,r] * W_r[k,i,r]
    # d||R||^2/dU_r[i,j,r] = 2 * sum_k R[i,j,k] * V_r[j,k,r] * W_r[k,i,r]
    
    # For each r, grad_U_r[i,j] = 2 * sum_k R[i,j,k] * V_r[j,k,r] * W_r[k,i,r]
    # We can compute this efficiently:
    
    # Let's compute gradient for each factor
    # grad_U_r[i,j] = 2 * einsum('ijk,jkr,kir->ij', R, V_r, W_r)  -- but this mixes r
    # Actually, for each r independently:
    # grad_U[:, :, r] = 2 * einsum('ijk,jkr,kir->ij', R, V_r[:,:,r], W_r[:,:,r])
    
    grad_U = np.zeros_like(U_r)
    grad_V = np.zeros_like(V_r)
    grad_W = np.zeros_like(W_r)
    
    for r in range(rank):
        # grad_U_r[i,j] = 2 * sum_k R[i,j,k] * V_r[j,k,r] * W_r[k,i,r]
        # = 2 * einsum('ijk,jk,ki->ij', R, V_r[:,:,r], W_r[:,:,r])
        grad_U[:,:,r] = 2.0 * np.einsum('ijk,jk,ki->ij', R, V_r[:,:,r], W_r[:,:,r])
        
        # grad_V_r[j,k] = 2 * sum_i R[i,j,k] * U_r[i,j,r] * W_r[k,i,r]
        # = 2 * einsum('ijk,ij,ki->jk', R, U_r[:,:,r], W_r[:,:,r])
        grad_V[:,:,r] = 2.0 * np.einsum('ijk,ij,ki->jk', R, U_r[:,:,r], W_r[:,:,r])
        
        # grad_W_r[k,i] = 2 * sum_j R[i,j,k] * U_r[i,j,r] * V_r[j,k,r]
        # = 2 * einsum('ijk,ij,jk->ki', R, U_r[:,:,r], V_r[:,:,r])
        grad_W[:,:,r] = 2.0 * np.einsum('ijk,ij,jk->ki', R, U_r[:,:,r], V_r[:,:,r])
    
    U -= lr * grad_U.reshape(n*m, rank)
    V -= lr * grad_V.reshape(m*p, rank)
    W -= lr * grad_W.reshape(p*n, rank)
    
    return U, V, W, np.sum(R**2)

def _try_optimize(n, m, p, rank, time_limit=30, max_iter=2000):
    """
    Try to find a decomposition of given rank using gradient descent.
    Initialize with a structured pattern and optimize.
    """
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initialize with a structured pattern
    # Use a block structure: partition indices into blocks
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize each rank component with a random sparse pattern
    for r in range(rank):
        # Choose a random subset of indices for each factor
        n_active = max(2, min(5, n*m))
        m_active = max(2, min(5, m*p))
        p_active = max(2, min(5, p*n))
        
        u_idx = rng.choice(n*m, size=n_active, replace=False)
        v_idx = rng.choice(m*p, size=m_active, replace=False)
        w_idx = rng.choice(p*n, size=p_active, replace=False)
        
        for idx in u_idx:
            U[idx, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        for idx in v_idx:
            V[idx, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        for idx in w_idx:
            W[idx, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
    
    # Gradient descent
    lr = 0.005
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        U, V, W, loss = _gradient_step(U, V, W, n, m, p, lr)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    # Round to half-integers
    best_U = _round_to_half(best_U)
    best_V = _round_to_half(best_V)
    best_W = _round_to_half(best_W)
    
    # Verify
    if _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-6):
        return best_U, best_V, best_W
    
    return None, None, None

def _local_search_polish(U, V, W, n, m, p, time_limit=10, max_iter=500):
    """
    Local search to polish a near-solution.
    Try changing individual entries to half-integers to reduce residual.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    half_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if _verify_fast(U, V, W, n, m, p, tol=1e-8):
            break
        
        # Compute residual
        R = _compute_residual(U, V, W, n, m, p)
        loss = np.sum(R**2)
        
        if loss < 1e-10:
            break
        
        # Find the worst entry
        worst_i, worst_j, worst_k = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        
        # Try changing entries in U, V, W that affect this position
        improved = False
        
        # U entries that affect (worst_i, worst_j, *)
        for r in range(rank):
            ui = worst_i * m + worst_j
            old = U[ui, r]
            for new_val in half_vals:
                if new_val == old:
                    continue
                U[ui, r] = new_val
                new_loss = np.sum(_compute_residual(U, V, W, n, m, p)**2)
                if new_loss < loss:
                    loss = new_loss
                    improved = True
                    break
                U[ui, r] = old
        
        if improved:
            continue
        
        # V entries
        for r in range(rank):
            vj = worst_j * p + worst_k
            old = V[vj, r]
            for new_val in half_vals:
                if new_val == old:
                    continue
                V[vj, r] = new_val
                new_loss = np.sum(_compute_residual(U, V, W, n, m, p)**2)
                if new_loss < loss:
                    loss = new_loss
                    improved = True
                    break
                V[vj, r] = old
        
        if improved:
            continue
        
        # W entries
        for r in range(rank):
            wk = worst_k * n + worst_i
            old = W[wk, r]
            for new_val in half_vals:
                if new_val == old:
                    continue
                W[wk, r] = new_val
                new_loss = np.sum(_compute_residual(U, V, W, n, m, p)**2)
                if new_loss < loss:
                    loss = new_loss
                    improved = True
                    break
                W[wk, r] = old
    
    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None, None

def _build_block_decomposition(n, m, p):
    """
    Build a block-structured decomposition.
    Partition A (n x m) into 2x2 blocks of size 2x2 (since n=4, m=5).
    Actually, let's use a simpler approach: use the known structure of matrix multiplication.
    
    For 4x5 * 5x5, we can use a block decomposition where we split:
    A = [A1; A2] where A1 is 2x5, A2 is 2x5
    B = [B1 B2] where B1 is 5x3, B2 is 5x2
    C = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    Each block product uses schoolbook, but we can share computations.
    """
    # This is getting complex. Let's use a simpler approach.
    # For now, return None and rely on optimization.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Safety margin before 150s
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try decreasing target ranks with optimization
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But in practice, for 4x5x5, we can likely do better than 100
    
    target_ranks = [80, 70, 60, 50, 45, 40, 35, 30]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        # Try optimization
        U_t, V_t, W_t = _try_optimize(n, m, p, target_rank, 
                                       time_limit=min(20, deadline - time.time() - 10))
        
        if U_t is not None:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
        
        # If optimization didn't converge, try local search from a random init
        rng = np.random.RandomState(42 + target_rank)
        U_rand = np.zeros((n*m, target_rank))
        V_rand = np.zeros((m*p, target_rank))
        W_rand = np.zeros((p*n, target_rank))
        
        for r in range(target_rank):
            for idx in rng.choice(n*m, size=3, replace=False):
                U_rand[idx, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
            for idx in rng.choice(m*p, size=3, replace=False):
                V_rand[idx, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
            for idx in rng.choice(p*n, size=3, replace=False):
                W_rand[idx, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        
        # Polish
        U_pol, V_pol, W_pol = _local_search_polish(U_rand, V_rand, W_rand, n, m, p,
                                                     time_limit=min(10, deadline - time.time() - 10))
        if U_pol is not None:
            best_U, best_V, best_W = U_pol, V_pol, W_pol
            best_rank = target_rank
            break
    
    # Final verification
    if not _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
