"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
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

def _add_schoolbook_block(U, V, W, n, m, p, i0, j0, k0, start_col):
    """Add schoolbook decomposition for sub-block M[n,m,p] at offsets i0,j0,k0."""
    col = start_col
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i0 + i) * m + (j0 + j)][col] = 1
                V[(j0 + j) * p + (k0 + k)][col] = 1
                W[(k0 + k) * n + (i0 + i)][col] = 1
                col += 1
    return col

def _add_strassen_2x2x2(U, V, W, n, m, p, i0, j0, k0, start_col):
    """
    Add Strassen decomposition for 2x2x2 block at offsets i0,j0,k0.
    Uses 7 rank terms.
    """
    col = start_col
    
    def u_idx(ii, jj):
        return (i0 + ii) * m + (j0 + jj)
    
    def v_idx(jj, kk):
        return (j0 + jj) * p + (k0 + kk)
    
    def w_idx(kk, ii):
        return (k0 + kk) * n + (i0 + ii)
    
    # M1 = (A00+A11)(B00+B11)
    U[u_idx(0, 0)][col] = 1
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 0)][col] = 1
    V[v_idx(1, 1)][col] = 1
    W[w_idx(0, 0)][col] = 1
    W[w_idx(1, 1)][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[u_idx(1, 0)][col] = 1
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 0)][col] = 1
    W[w_idx(0, 1)][col] = 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[u_idx(0, 0)][col] = 1
    V[v_idx(1, 0)][col] = 1
    V[v_idx(1, 1)][col] = -1
    W[w_idx(1, 0)][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 1)][col] = 1
    V[v_idx(0, 0)][col] = -1
    W[w_idx(0, 0)][col] = 1
    W[w_idx(0, 1)][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[u_idx(0, 0)][col] = 1
    U[u_idx(0, 1)][col] = 1
    V[v_idx(1, 1)][col] = 1
    W[w_idx(0, 0)][col] = -1
    W[w_idx(1, 0)][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[u_idx(1, 0)][col] = 1
    U[u_idx(0, 0)][col] = -1
    V[v_idx(0, 0)][col] = 1
    V[v_idx(0, 1)][col] = 1
    W[w_idx(1, 1)][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[u_idx(0, 1)][col] = 1
    U[u_idx(1, 1)][col] = -1
    V[v_idx(1, 0)][col] = 1
    V[v_idx(1, 1)][col] = 1
    W[w_idx(0, 0)][col] = 1
    col += 1
    
    return col

def _build_rank34():
    """Build the rank-34 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 3, 3, 4
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 2, col)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    col = _add_schoolbook_block(U, V, W, 2, 1, 4, 0, 2, 0, col)
    
    # Block 4: A[2:3, 0:2] * B[0:2, 0:4] -> C[2:3, 0:4]
    col = _add_schoolbook_block(U, V, W, 1, 2, 4, 2, 0, 0, col)
    
    # Block 5: A[2:3, 2:3] * B[2:3, 0:4] -> C[2:3, 0:4]
    col = _add_schoolbook_block(U, V, W, 1, 1, 4, 2, 2, 0, col)
    
    if col != rank:
        print(f"Warning: Rank mismatch. Expected {rank}, got {col}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _loss_and_grad(U, V, W, n=3, m=3, p=4):
    """
    Compute loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    and analytical gradients for U, V, W.
    """
    rank = U.shape[1]
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i, j, k] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
    
    loss = np.sum((S - 1.0) ** 2)
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err = 2.0 * (S[i, j, k] - 1.0)
                grad_U[i * m + j] += err * V[j * p + k] * W[k * n + i]
                grad_V[j * p + k] += err * U[i * m + j] * W[k * n + i]
                grad_W[k * n + i] += err * U[i * m + j] * V[j * p + k]
    
    return loss, grad_U, grad_V, grad_W

def _structured_init(n, m, p, rank, seed=42):
    """
    Initialize U, V, W with a structured pattern that partially satisfies the constraints.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Add small random perturbations to break symmetry
    noise_scale = 0.01
    U += rng.randn(n * m, rank) * noise_scale
    V += rng.randn(m * p, rank) * noise_scale
    W += rng.randn(p * n, rank) * noise_scale
    
    return U, V, W

def _optimize(U, V, W, n, m, p, max_iters=2000, lr=0.01, deadline=None):
    """Gradient descent optimization with adaptive learning rate."""
    loss, _, _ = _loss_and_grad(U, V, W, n, m, p)
    best_loss = loss
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if grad_norm > 1e-10:
            effective_lr = lr * min(1.0, 10.0 / grad_norm)
            U -= effective_lr * gU
            V -= effective_lr * gV
            W -= effective_lr * gW
        else:
            break
        
        # Reduce learning rate over time
        if it % 500 == 499:
            lr *= 0.5
    
    return best_U, best_V, best_W

def _round_to_half_integers(U, V, W):
    """Round values to nearest half-integer."""
    return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try the rank-34 construction first
    result = _build_rank34()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If rank-34 fails, try to find a working decomposition via optimization
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions via optimization
    for target_rank in range(31, best_rank):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U, V, W = _structured_init(n, m, p, target_rank, seed=42 + attempt)
            U, V, W = _optimize(U, V, W, n, m, p, max_iters=1500, lr=0.01, deadline=deadline - 30)
            
            # Round to half-integers and verify
            U_r, V_r, W_r = _round_to_half_integers(U, V, W)
            
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-9):
                best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = target_rank
                break
    
    return best
