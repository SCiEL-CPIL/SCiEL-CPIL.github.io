"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS

    def tensor_residual(U, V, W):
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        target = np.ones((n, m, p))
        return np.sum((reconstructed - target) ** 2)

    def is_half_integer(arr, tol=1e-6):
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)

    def round_to_half(arr):
        return np.round(arr * 2) / 2

    def round_to_int(arr):
        return np.round(arr)

    def verify_solution(U, V, W, tol=1e-8):
        if not (is_half_integer(U, 1e-4) and is_half_integer(V, 1e-4) and is_half_integer(W, 1e-4)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol

    def schoolbook():
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
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    def winograd_init(target_rank, seed=0):
        """Initialize using Winograd-style structure for small matrix mult."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        return U, V, W

    def als_decompose(target_rank, max_iters=150, tol=1e-14, seed=0, init_type="random"):
        if init_type == "winograd":
            U, V, W = winograd_init(target_rank, seed)
        else:
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, target_rank) * 0.3
            V = rng.randn(m * p, target_rank) * 0.3
            W = rng.randn(p * n, target_rank) * 0.3
        
        T = np.ones((n, m, p))
        
        prev_error = np.inf
        for iter_ in range(max_iters):
            if time.time() > deadline - 5:
                return None, None, None, None
                
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((target_rank, p))
                    for k in range(p):
                        A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((target_rank, n))
                    for i in range(n):
                        A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((target_rank, m))
                    for j in range(m):
                        A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
            
            error = tensor_residual(U, V, W)
            
            if error < tol:
                break
                
            if error > prev_error * 1.001 and iter_ > 20:
                break
            prev_error = error
        
        return U, V, W, error

    def polish(U, V, W, max_iters=50, tol=1e-15):
        """High-precision polish pass."""
        T = np.ones((n, m, p))
        rank = U.shape[1]
        
        for iter_ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, m))
                    for j in range(m):
                        A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
            
            error = tensor_residual(U, V, W)
            if error < tol:
                break
        
        return U, V, W, error

    def try_round_and_verify(U, V, W, target_rank):
        """Try various rounding strategies and verify."""
        # Try half-integer rounding
        U_h = round_to_half(U)
        V_h = round_to_half(V)
        W_h = round_to_half(W)
        
        if verify_solution(U_h, V_h, W_h, 1e-6):
            return U_h, V_h, W_h
        
        # Try integer rounding
        U_i = round_to_int(U)
        V_i = round_to_int(V)
        W_i = round_to_int(W)
        
        if verify_solution(U_i, V_i, W_i, 1e-6):
            return U_i, V_i, W_i
        
        # Try mixed: round U to half, V to half, W to half with small perturbation
        for tol_try in [1e-5, 5e-5, 1e-4, 5e-4]:
            U_t = np.round(U / tol_try) * tol_try
            V_t = np.round(V / tol_try) * tol_try
            W_t = np.round(W / tol_try) * tol_try
            if verify_solution(U_t, V_t, W_t, 1e-5):
                return U_t, V_t, W_t
        
        return None, None, None

    def try_rank(target_rank, max_attempts=40):
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            seed = 42 + target_rank * 1000 + attempt
            init_type = "winograd" if attempt % 2 == 0 else "random"
            
            U, V, W, error = als_decompose(target_rank, max_iters=100, tol=1e-12, seed=seed, init_type=init_type)
            
            if U is None:
                return False
            
            if error < 1e-6:
                # Polish
                U_p, V_p, W_p, error_p = polish(U, V, W, max_iters=30, tol=1e-14)
                
                if error_p < 1e-8:
                    U_r, V_r, W_r = try_round_and_verify(U_p, V_p, W_p, target_rank)
                    if U_r is not None:
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        return True
                    elif verify_solution(U_p, V_p, W_p, 1e-6):
                        best = {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
                        best_rank = target_rank
                        return True
        
        return False

    for target_rank in range(13, rank_sb):
        if time.time() > deadline - 20:
            break
        found = try_rank(target_rank, max_attempts=40)
        if found:
            break

    return best
# EVOLVE-BLOCK-END
