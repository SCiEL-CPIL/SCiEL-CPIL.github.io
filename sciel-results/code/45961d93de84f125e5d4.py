"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    T_expected = np.ones((n, m, p))
    T_computed = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T_computed += U_r[:, :, np.newaxis] * V_r[np.newaxis, :, :] * W_r[np.newaxis, np.newaxis, :]
    return np.allclose(T_computed, T_expected, atol=tol)

def _alternating_minimization(U, V, W, n, m, p, max_iter=2000, tol=1e-10, time_budget=None):
    """Alternating minimization to refine U, V, W with time budget."""
    rank = U.shape[1]
    start = time.time()
    
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
            
        # Fix V, W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(C.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    return None, None, None

        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return None, None, None

        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return None, None, None

        U, V, W = U_new, V_new, W_new
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
            
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None, None, None

def _random_init(n, m, p, rank, seed=42):
    """Initialize U, V, W with random values (dense)."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    # Normalize to avoid numerical issues
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    return U, V, W

def _strassen_style_init(n, m, p, rank, seed=42):
    """Initialize using Strassen-like structure for better convergence."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use block structure: split p into chunks
    # For 3x3x8, try to exploit structure
    # Initialize with small random perturbations around structured values
    
    # Base: use schoolbook-like structure but with shared components
    base_rank = min(rank, 27)  # 3x3x3 = 27
    for r in range(base_rank):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
    
    # Add random components for remaining ranks
    if rank > base_rank:
        U[:, base_rank:rank] = rng.randn(n * m, rank - base_rank) * 0.1
        V[:, base_rank:rank] = rng.randn(m * p, rank - base_rank) * 0.1
        W[:, base_rank:rank] = rng.randn(p * n, rank - base_rank) * 0.1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds suggest rank around 60-65 might be possible
    for target_rank in [64, 63, 62, 61, 60, 59, 58]:
        if time.time() > deadline - 30:
            break
            
        # Try multiple initialization strategies
        for init_type in ['strassen', 'random', 'strassen2']:
            if time.time() > deadline - 30:
                break
                
            for attempt in range(3):
                if time.time() > deadline - 30:
                    break
                    
                seed = 42 + target_rank * 100 + init_type_hash(init_type) * 10 + attempt
                if init_type == 'strassen':
                    U, V, W = _strassen_style_init(n, m, p, target_rank, seed=seed)
                elif init_type == 'strassen2':
                    U, V, W = _strassen_style_init(n, m, p, target_rank, seed=seed + 1000)
                else:
                    U, V, W = _random_init(n, m, p, target_rank, seed=seed)
                
                time_budget = min(deadline - 5, time.time() + 20)
                U_res, V_res, W_res = _alternating_minimization(U, V, W, n, m, p, max_iter=300, time_budget=time_budget)
                
                if U_res is not None and _verify_decomposition(U_res, V_res, W_res, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
            
    return best_result

def init_type_hash(s):
    """Simple hash for string to int."""
    return sum(ord(c) for c in s) % 100
# EVOLVE-BLOCK-END
