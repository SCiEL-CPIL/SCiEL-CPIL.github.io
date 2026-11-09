"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np

# The dimensions: n x m * m x p = 3x4 * 4x6
n_dim = 3
m_dim = 4
p_dim = 6

def _verify_fast(U, V, W, tensor, tol=1e-9):
    """Fast verification using vectorization."""
    rank = U.shape[1]
    n, m, p = tensor.shape
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        err = np.max(np.abs(reconstructed - tensor))
        return err < tol
    except Exception:
        return False

def _compute_error(U, V, W, tensor, n, m, p, rank):
    """Compute reconstruction error."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.max(np.abs(reconstructed - tensor))

def _grad_step(U, V, W, tensor, n, m, p, rank, lr):
    """Perform a gradient descent step on the squared error."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Residual: R[i,j,k] = sum_c U[i,j,c]V[j,k,c]W[k,i,c] - T[i,j,k]
    # We want to minimize sum R^2
    # Grad U[i,j,c] = 2 * sum_k R[i,j,k] * V[j,k,c] * W[k,i,c]
    # Grad V[j,k,c] = 2 * sum_i R[i,j,k] * U[i,j,c] * W[k,i,c]
    # Grad W[k,i,c] = 2 * sum_j R[i,j,k] * U[i,j,c] * V[j,k,c]
    
    # Compute Residual
    Rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    R = Rec - tensor
    
    # Gradients
    # G_U[i,j,c] = 2 * einsum('ijk,jkc,kic->ijc', R, V_r, W_r)
    G_U = 2 * np.einsum('ijk,jkc,kic->ijc', R, V_r, W_r, optimize=True)
    # G_V[j,k,c] = 2 * einsum('ijk,ijc,kic->jkc', R, U_r, W_r)
    G_V = 2 * np.einsum('ijk,ijc,kic->jkc', R, U_r, W_r, optimize=True)
    # G_W[k,i,c] = 2 * einsum('ijk,ijc,jkc->kic', R, U_r, V_r)
    G_W = 2 * np.einsum('ijk,ijc,jkc->kic', R, U_r, V_r, optimize=True)
    
    U_new = U - lr * G_U.reshape(n*m, rank)
    V_new = V - lr * G_V.reshape(m*p, rank)
    W_new = W - lr * G_W.reshape(p*n, rank)
    
    return U_new, V_new, W_new, np.max(np.abs(R))

def _optimize_rank(U, V, W, tensor, n, m, p, rank, deadline):
    """Run gradient descent for a specific rank."""
    start_lr = 1e-3
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(200):
        if time.time() > deadline:
            break
            
        U, V, W, err = _grad_step(U, V, W, tensor, n, m, p, rank, start_lr)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-8:
            # Try to polish with smaller lr
            break
            
        # Adaptive LR: if err increased, reduce lr
        # Simple heuristic: if stuck, reduce lr
        if it > 10 and err > best_err * 1.1:
            start_lr *= 0.5
            
    return best_U, best_V, best_W, best_err

def _round_and_verify(U, V, W, tensor, n, m, p, rank, tol=1e-6):
    """Round to half-integers and verify."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Round to nearest 0.5
    U_round = np.round(U_r * 2) / 2
    V_round = np.round(V_r * 2) / 2
    W_round = np.round(W_r * 2) / 2
    
    # Verify
    Rec = np.einsum('ijc,jkc,kic->ijk', U_round, V_round, W_round, optimize=True)
    err = np.max(np.abs(Rec - tensor))
    
    if err < tol:
        return U_round.reshape(n*m, rank), V_round.reshape(m*p, rank), W_round.reshape(p*n, rank), True
    return None, None, None, False

def _structured_init(n, m, p, rank, rng):
    """Generate structured initializations."""
    inits = []
    
    # Init 1: Random
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    inits.append((U, V, W))
    
    # Init 2: Sparse
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        idx_u = np.random.choice(n * m, size=max(1, n * m // rank + 1), replace=False)
        idx_v = np.random.choice(m * p, size=max(1, m * p // rank + 1), replace=False)
        idx_w = np.random.choice(p * n, size=max(1, p * n // rank + 1), replace=False)
        U[idx_u, r] = rng.standard_normal(len(idx_u))
        V[idx_v, r] = rng.standard_normal(len(idx_v))
        W[idx_w, r] = rng.standard_normal(len(idx_w))
    inits.append((U, V, W))
    
    return inits

def construct():
    deadline = 145  # seconds, safe margin before 150s limit
    start_time = time.time()
    
    n = n_dim
    m = m_dim
    p = p_dim
    
    # Create the standard tensor: tensor[i,j,k] = 1 for all i,j,k
    tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_schoolbook
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    
    rng = np.random.default_rng(42)
    
    # Search ranks from low to high
    # We try a few lower ranks first. If they fail, we return schoolbook.
    # To save time, we limit the number of ranks we try deeply.
    ranks_to_try = list(range(20, rank_schoolbook + 1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Generate initializations
        inits = _structured_init(n, m, p, target_rank, rng)
        
        found = False
        for U0, V0, W0 in inits:
            if time.time() > deadline - 5:
                break
            
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            
            # Run gradient descent
            U, V, W, err = _optimize_rank(U, V, W, tensor, n, m, p, target_rank, deadline - 2)
            
            if err < 1e-6:
                # Try to round to half-integers
                U_r, V_r, W_r, is_valid = _round_and_verify(U, V, W, tensor, n, m, p, target_rank)
                if is_valid:
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
                else:
                    # If rounding fails, we might still accept the float solution if the evaluator allows it?
                    # The prompt says "integer/half-integer entries". So we MUST round.
                    # If rounding doesn't work, this rank is not valid.
                    pass
            
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
