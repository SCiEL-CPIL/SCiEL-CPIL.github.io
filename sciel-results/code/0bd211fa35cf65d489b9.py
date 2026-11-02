import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _build_nonzero_and_zero_indices(n, m, p):
    non_zero = []
    non_zero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero.append((a, b, c))
                non_zero_set.add((a, b, c))
    all_indices = [(a, b, c) for a in range(n * m) for b in range(m * p) for c in range(p * n)]
    zero = [idx for idx in all_indices if idx not in non_zero_set]
    return non_zero, zero

def _verify_exact(U, V, W, non_zero, zero, tol=1e-9):
    for a, b, c in non_zero:
        s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        if abs(s - 1.0) > tol:
            return False
    for a, b, c in zero:
        s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        if abs(s) > tol:
            return False
    return True

def _try_rank(rank, n, m, p, non_zero, zero, deadline, seed=0):
    """Try to find rank-r decomposition using alternating least squares."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    # Initialize with random small values
    scale = 0.1
    U = rng.normal(size=(n * m, rank)) * scale
    V = rng.normal(size=(m * p, rank)) * scale
    W = rng.normal(size=(p * n, rank)) * scale
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.005
    max_iter = 3000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute residuals and gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        err = 0.0
        
        # Precompute outer products for speed
        # For each r, we have u_r, v_r, w_r vectors
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Process non-zero entries
        for a, b, c in non_zero:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            e = s - 1.0
            err += e * e
            dU[a, :] += e * V[b, :] * W[c, :]
            dV[b, :] += e * U[a, :] * W[c, :]
            dW[c, :] += e * U[a, :] * V[b, :]
        
        # Process a random sample of zero entries for speed
        sample_size = min(len(zero), 200)
        if sample_size < len(zero):
            sample_idx = rng.choice(len(zero), sample_size, replace=False)
            zero_sample = [zero[i] for i in sample_idx]
        else:
            zero_sample = zero
            
        for a, b, c in zero_sample:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            e = s
            err += e * e
            dU[a, :] += e * V[b, :] * W[c, :]
            dV[b, :] += e * U[a, :] * W[c, :]
            dW[c, :] += e * U[a, :] * V[b, :]
        
        # Normalize gradient by number of samples
        total_samples = len(non_zero) + len(zero_sample)
        dU /= total_samples
        dV /= total_samples
        dW /= total_samples
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # Try to snap to half-integers if error is small
        if err < 1e-4:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify_exact(U_snap, V_snap, W_snap, non_zero, zero):
                return U_snap, V_snap, W_snap, 0.0
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Build indices
    non_zero, zero = _build_nonzero_and_zero_indices(n, m, p)
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound for rank of M_{n,m,p} is max(n*m, m*p, p*n)
    # For 2x3x5: max(6, 15, 10) = 15
    # But we know rank is at most 30 (schoolbook)
    # Let's try ranks from 15 up to 29
    
    for target_rank in range(max(n * m, m * p, p * n), rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_rank(target_rank, n, m, p, non_zero, zero, deadline, seed)
            if result is not None:
                U, V, W, final_err = result
                # Final verification with all zero entries
                if _verify_exact(U, V, W, non_zero, zero, tol=1e-9):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: return schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
