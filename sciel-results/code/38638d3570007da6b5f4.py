"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Use einsum for speed
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return bool(np.allclose(T, 1.0, atol=tol))

def _polish(U, V, W, n, m, p, max_iters=5000, tol=1e-14):
    """High-precision polish with rational recovery."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 2:
            break
        
        # Update U rows
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-15)
                U[i * m + j, :] = u
        
        # Update V rows
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-15)
                V[j * p + k, :] = v
        
        # Update W rows
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-15)
                W[k * n + i, :] = w
        
        if _verify_fast(U, V, W, n, m, p, tol=tol):
            # Try rational recovery
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-10):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            # Try integer recovery
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if _verify_fast(U_i, V_i, W_i, n, m, p, tol=1e-10):
                return (U_i.tolist(), V_i.tolist(), W_i.tolist())
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _try_als(U, V, W, n, m, p, max_iters=3000, tol=1e-12):
    """Efficient alternating least squares optimization."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-13)
                U[i * m + j, :] = u
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-13)
                V[j * p + k, :] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-13)
                W[k * n + i, :] = w
        
        # Check convergence via error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        
        if err < tol:
            # Polish
            result = _polish(U, V, W, n, m, p)
            if result is not None:
                return result
            return (U.tolist(), V.tolist(), W.tolist())
        
        # Early stop if no improvement
        if err > prev_err * 1.001 and it > 100:
            break
        prev_err = err
    
    return None

def _try_rank(rank, n, m, p, seed=42):
    """Try to find rank-rank decomposition with diverse initializations."""
    rng = np.random.RandomState(seed)
    
    inits = []
    
    # Init 0: Small random
    inits.append((
        rng.randn(n*m, rank) * 0.05,
        rng.randn(m*p, rank) * 0.05,
        rng.randn(p*n, rank) * 0.05
    ))
    
    # Init 1: Structured basis (cyclic)
    U1 = np.zeros((n*m, rank))
    V1 = np.zeros((m*p, rank))
    W1 = np.zeros((p*n, rank))
    for c in range(rank):
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n*m)) % p
        U1[i_idx * m + j_idx, c] = 1.0
        V1[j_idx * p + k_idx, c] = 1.0
        W1[k_idx * n + i_idx, c] = 1.0
    inits.append((U1, V1, W1))
    
    # Init 2: Random perturbation of basis
    U2 = U1 + rng.randn(n*m, rank) * 0.1
    V2 = V1 + rng.randn(m*p, rank) * 0.1
    W2 = W1 + rng.randn(p*n, rank) * 0.1
    inits.append((U2, V2, W2))
    
    # Init 3: Random with moderate scale
    inits.append((
        rng.randn(n*m, rank) * 0.1,
        rng.randn(m*p, rank) * 0.1,
        rng.randn(p*n, rank) * 0.1
    ))
    
    # Init 4: Sparse random
    U4 = (rng.rand(n*m, rank) > 0.5).astype(float)
    V4 = (rng.rand(m*p, rank) > 0.5).astype(float)
    W4 = (rng.rand(p*n, rank) > 0.5).astype(float)
    inits.append((U4, V4, W4))
    
    # Init 5: Random linear transforms of basis
    P = np.eye(rank) + rng.randn(rank, rank) * 0.3
    Q = np.eye(rank) + rng.randn(rank, rank) * 0.3
    R = np.eye(rank) + rng.randn(rank, rank) * 0.3
    inits.append((U1 @ P, V1 @ Q, W1 @ R))
    
    # Init 6: Anti-symmetric structure
    U6 = np.zeros((n*m, rank))
    V6 = np.zeros((m*p, rank))
    W6 = np.zeros((p*n, rank))
    for c in range(rank):
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n*m)) % p
        U6[i_idx * m + j_idx, c] = 1.0 if c % 2 == 0 else -1.0
        V6[j_idx * p + k_idx, c] = 1.0 if c % 3 == 0 else -1.0
        W6[k_idx * n + i_idx, c] = 1.0 if c % 4 == 0 else -1.0
    inits.append((U6, V6, W6))
    
    # Init 7: Kronecker-like structure
    U7 = np.zeros((n*m, rank))
    V7 = np.zeros((m*p, rank))
    W7 = np.zeros((p*n, rank))
    for c in range(rank):
        a = c % n
        b = (c // n) % m
        cc = (c // (n*m)) % p
        U7[a * m + b, c] = 1.0
        V7[b * p + cc, c] = 1.0
        W7[cc * n + a, c] = 1.0
        # Add cross terms
        if c + 1 < rank:
            a2 = (c + 1) % n
            b2 = ((c + 1) // n) % m
            cc2 = ((c + 1) // (n*m)) % p
            U7[a2 * m + b2, c] += 0.5
            V7[b2 * p + cc2, c] += 0.5
            W7[cc2 * n + a2, c] += 0.5
    inits.append((U7, V7, W7))
    
    for U_init, V_init, W_init in inits:
        if time.time() > _DEADLINE - 5:
            break
        result = _try_als(U_init.tolist(), V_init.tolist(), W_init.tolist(), n, m, p, max_iters=2000, tol=1e-12)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    lower_bound = max(n*m, m*p, n*p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > _DEADLINE - 5:
            break
        
        for seed in range(30):
            if time.time() > _DEADLINE - 5:
                break
            result = _try_rank(rank, n, m, p, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    if best_U is None:
        # Fall back to schoolbook
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
