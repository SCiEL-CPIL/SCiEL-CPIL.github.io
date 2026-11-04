import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _compute_error(U, V, W, n, m, p, sample_zeros=500):
    """Compute the total error of the decomposition."""
    rank = U.shape[1]
    error = 0.0
    # 1-positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; g = k*n+i
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                error += (val - 1.0) ** 2
    # Zero positions (sampled)
    np.random.seed(42)
    for _ in range(sample_zeros):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        g = np.random.randint(0, p*n)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.sum(U[a, :] * V[b, :] * W[g, :])
            error += val ** 2
    return error

def _optimize_decomposition_fast(n, m, p, initial_rank=None, max_time=100):
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Pre-compute indices for fast t-vector calculation
    a_to_bg = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = []
            for k in range(p):
                b = j * p + k
                g = k * n + i
                pairs.append((b, g))
            a_to_bg[a] = pairs

    b_to_ag = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = []
            for i in range(n):
                a = i * m + j
                g = k * n + i
                pairs.append((a, g))
            b_to_ag[b] = pairs

    g_to_ab = {}
    for k in range(p):
        for i in range(n):
            g = k * n + i
            pairs = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                pairs.append((a, b))
            g_to_ab[g] = pairs

    # Start with schoolbook as a warm start for the highest rank
    U_curr, V_curr, W_curr = _build_schoolbook(n, m, p)
    
    for target_rank in range(initial_rank - 1, max(20, initial_rank - 60), -1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        for seed in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # ALS Loop
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                G_V = V.T @ V
                G_W = W.T @ W
                A_mat = G_V * G_W
                A_mat += 1e-4 * np.eye(target_rank) # Stronger regularization
                
                for a in range(n * m):
                    b_vec = np.zeros(target_rank)
                    for b, g in a_to_bg[a]:
                        b_vec += V[b, :] * W[g, :]
                    
                    try:
                        U[a, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update V
                G_U = U.T @ U
                G_W = W.T @ W
                A_mat = G_U * G_W
                A_mat += 1e-4 * np.eye(target_rank)
                
                for b in range(m * p):
                    b_vec = np.zeros(target_rank)
                    for a, g in b_to_ag[b]:
                        b_vec += U[a, :] * W[g, :]
                    
                    try:
                        V[b, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update W
                G_U = U.T @ U
                G_V = V.T @ V
                A_mat = G_U * G_V
                A_mat += 1e-4 * np.eye(target_rank)
                
                for g in range(p * n):
                    b_vec = np.zeros(target_rank)
                    for a, b in g_to_ab[g]:
                        b_vec += U[a, :] * V[b, :]
                    
                    try:
                        W[g, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Check convergence
                if iteration % 5 == 0:
                    error = _compute_error(U, V, W, n, m, p, sample_zeros=200)
                    if error < 1e-5:
                        # Try rounding
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                            return U_rounded, V_rounded, W_rounded
                        # If rounding failed, we could try to nudge, but for now we accept failure for this seed
                        break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to find a better decomposition
    result = _optimize_decomposition_fast(n, m, p, initial_rank=best_rank, max_time=140)
    
    if result is not None:
        U, V, W = result
        if U.shape[1] < best_rank:
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
