"""Rewritten program for matrix multiplication 2x4 * 4x5 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p  # 2*4*5 = 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = 135  # leave 15s margin before 150s kill

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomp_fast(U, V, W, tol=1e-8):
        """Faster verification using numpy einsum."""
        rank = U.shape[1]
        U_idx = U.reshape(n, m, rank)
        V_idx = V.reshape(m, p, rank)
        W_idx = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True

    def polish_to_rational(U, V, W, max_rounds=5):
        """Try to round to rational values (integers or half-integers) while maintaining validity."""
        for _ in range(max_rounds):
            # Try rounding each entry to nearest 0.5
            U_new = np.round(U * 2) / 2
            V_new = np.round(V * 2) / 2
            W_new = np.round(W * 2) / 2
            if verify_decomp_fast(U_new, V_new, W_new, tol=1e-10):
                return U_new, V_new, W_new, True
            # Try rounding to integers
            U_new = np.round(U)
            V_new = np.round(V)
            W_new = np.round(W)
            if verify_decomp_fast(U_new, V_new, W_new, tol=1e-10):
                return U_new, V_new, W_new, True
            break
        return U, V, W, False

    def als_decompose(target_rank, max_iters=500, tol=1e-10, seed=42, init_type='random'):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_type == 'random':
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
        elif init_type == 'structured':
            # Use structured initialization based on Kronecker-like patterns
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            # Fill with a pattern that might help convergence
            for c in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = 1.0 / (c + 1)
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, c] = 1.0 / (c + 1)
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, c] = 1.0 / (c + 1)
        elif init_type == 'sparse':
            # Sparse initialization with random signs
            U = rng.choice([-1, 1], size=(n * m, target_rank)) * 0.5
            V = rng.choice([-1, 1], size=(m * p, target_rank)) * 0.5
            W = rng.choice([-1, 1], size=(p * n, target_rank)) * 0.5
        else:
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
                
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        U[i * m + j, :] = np.linalg.pinv(M) @ np.ones(p)
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        V[j * p + k, :] = np.linalg.pinv(M) @ np.ones(n)
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        W[k * n + i, :] = np.linalg.pinv(M) @ np.ones(m)
            
            # Check convergence
            if verify_decomp_fast(U, V, W, tol=1e-8):
                # Try to polish to rational values
                U_pol, V_pol, W_pol, polished = polish_to_rational(U, V, W)
                if polished:
                    return U_pol, V_pol, W_pol, True
                return U, V, W, True
        
        # Check if we got a valid decomposition
        if verify_decomp_fast(U, V, W, tol=1e-6):
            U_pol, V_pol, W_pol, polished = polish_to_rational(U, V, W)
            if polished:
                return U_pol, V_pol, W_pol, True
            return U, V, W, True
        return U, V, W, False

    # Try various ranks from lower bound up
    # Lower bound is 20, but let's start from 22 for safety
    target_ranks = list(range(22, 40))
    
    init_types = ['random', 'structured', 'sparse']
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
            
        # Try multiple seeds and initialization types
        found = False
        for init_type in init_types:
            if time.time() > deadline:
                break
            if found:
                break
                
            for seed in range(15):
                if time.time() > deadline:
                    break
                if found:
                    break
                    
                U, V, W, valid = als_decompose(target_rank, max_iters=400, tol=1e-10, seed=seed, init_type=init_type)
                if valid:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    found = True
                    break

    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
