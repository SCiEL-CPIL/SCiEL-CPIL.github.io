"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Combines structured Hadamard/Winograd-style initialization with iterative refinement.
Falls back to schoolbook (rank 80) if no better decomposition is found in time."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_fast(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=tol)

def _hadamard_matrix(size, dtype=np.float64):
    """Construct a Hadamard matrix of given size (power of 2)."""
    if size == 1:
        return np.ones((1, 1), dtype=dtype)
    H = _hadamard_matrix(size // 2, dtype)
    return np.block([[H, H], [H, -H]])

def _winograd_init(rank, n=4, m=4, p=5, seed=42):
    """
    Initialize U, V, W using structured transforms inspired by Winograd/Hadamard.
    The idea: use block-diagonal or Kronecker-structured patterns to exploit
    algebraic symmetries in the multiplication tensor.
    """
    np_rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Use a combination of structured and random initialization
    # For each rank column, mix Hadamard-like patterns with random perturbations
    
    H4 = _hadamard_matrix(4)  # 4x4 Hadamard
    H2 = _hadamard_matrix(2)  # 2x2 Hadamard
    
    for c in range(rank):
        # Randomly choose a structure type
        struct_type = c % 4
        
        if struct_type == 0:
            # Hadamard-based pattern for U
            pattern = H4.flatten()
            U[:, c] = pattern[:n*m] * np_rng.choice([-1, 1])
            V[:, c] = H4.flatten()[:m*p] * np_rng.choice([-1, 1])
            W[:, c] = H4.flatten()[:p*n] * np_rng.choice([-1, 1])
        elif struct_type == 1:
            # Block structure using H2
            block_U = np.kron(H2, np.eye(2)).flatten()[:n*m]
            block_V = np.kron(H2, np.eye(2)).flatten()[:m*p]
            block_W = np.kron(H2, np.eye(2)).flatten()[:p*n]
            U[:, c] = block_U * np_rng.choice([-1, 1])
            V[:, c] = block_V * np_rng.choice([-1, 1])
            W[:, c] = block_W * np_rng.choice([-1, 1])
        elif struct_type == 2:
            # Sparse random with structured signs
            density = 0.4
            for idx in range(n*m):
                if np_rng.random() < density:
                    U[idx, c] = np_rng.choice([-1, 1, 0.5, -0.5])
            for idx in range(m*p):
                if np_rng.random() < density:
                    V[idx, c] = np_rng.choice([-1, 1, 0.5, -0.5])
            for idx in range(p*n):
                if np_rng.random() < density:
                    W[idx, c] = np_rng.choice([-1, 1, 0.5, -0.5])
        else:
            # Pure random small values
            U[:, c] = np_rng.randn(n*m) * 0.5
            V[:, c] = np_rng.randn(m*p) * 0.5
            W[:, c] = np_rng.randn(p*n) * 0.5
    
    return U, V, W

def _refine(U, V, W, n=4, m=4, p=5, max_iters=200, lr=0.01, deadline=None):
    """
    Iteratively refine U, V, W to minimize the error in the tensor decomposition.
    Uses gradient descent on the smooth objective: sum over (i,j,k) of (T[i,j,k] - 1)^2
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Compute current tensor values
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            uc = U.reshape(n, m, rank)[:, :, c]
            vc = V.reshape(m, p, rank)[:, :, c]
            wc = W.reshape(p, n, rank)[:, :, c]
            wcT = wc.T
            for j in range(m):
                T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wcT
        
        # Compute error
        E = T - 1.0
        max_err = np.max(np.abs(E))
        
        if max_err < 1e-9:
            break
        
        # Gradient computation
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = E[i, j, k]
                    # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                    U_grad[i*m+j, :] += 2 * err * Vr[j, k, :] * Wr[k, i, :]
                    # dT/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                    V_grad[j*p+k, :] += 2 * err * Ur[i, j, :] * Wr[k, i, :]
                    # dT/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                    W_grad[k*n+i, :] += 2 * err * Ur[i, j, :] * Vr[j, k, :]
        
        # Update
        U -= lr * U_grad
        V -= lr * V_grad
        W -= lr * W_grad
    
    return U, V, W

def _try_decomposition(target_rank, seed, n=4, m=4, p=5, deadline=None):
    """Try to find a valid decomposition of the given rank."""
    if deadline is not None and time.time() > deadline - 15:
        return None
    
    # Initialize using Winograd/Hadamard structure
    U, V, W = _winograd_init(target_rank, n, m, p, seed)
    
    # Refine
    U, V, W = _refine(U, V, W, n, m, p, max_iters=150, lr=0.005, deadline=deadline)
    
    # Check if valid
    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-7):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a second refinement pass with smaller learning rate
    U, V, W = _refine(U, V, W, n, m, p, max_iters=100, lr=0.001, deadline=deadline)
    
    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-7):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    best_result = None
    
    # Try ranks below schoolbook, starting from lower ranks for potential big wins
    # but also try near-schoolbook ranks where convergence is easier
    ranks_to_try = [70, 75, 72, 77, 74, 78, 76, 79]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds for each rank
        for seed_offset in range(3):
            if time.time() > deadline - 20:
                break
            seed = target_rank * 1000 + seed_offset * 137
            result = _try_decomposition(target_rank, seed, n, m, p, deadline)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
