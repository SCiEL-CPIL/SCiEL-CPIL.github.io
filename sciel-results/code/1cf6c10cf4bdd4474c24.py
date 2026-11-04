import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110.0

    # Try to find a lower rank decomposition
    target_ranks = [25, 26, 27, 28, 29, 30, 35, 40, 44]
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 5:
            break
        try:
            result = search_rank(n, m, p, target_rank, deadline)
            if result is not None:
                return result
        except Exception:
            pass

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def search_rank(n, m, p, rank, deadline):
    # Initialize with small random half-integers
    U = np.random.randint(0, 3, size=(n * m, rank)) * 0.5
    V = np.random.randint(0, 3, size=(m * p, rank)) * 0.5
    W = np.random.randint(0, 3, size=(p * n, rank)) * 0.5

    # Precompute target tensor values
    # T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Precompute indices for efficient error calculation
    # We will compute the error as Frobenius norm of (DecomposedTensor - TargetTensor)
    
    # Let's use a simple gradient-free local search (Nelder-Mead style or random perturbation)
    # Given the high dimensionality (3 matrices * rank * dim), this is hard.
    # But the directive suggests: "initializing the factor matrices with small random integer or half-integer values and iteratively applying gradient-free local search... to minimize the Frobenius norm... until exact integer recovery is achieved."
    
    # Since exact integer recovery is very hard with generic search, we will try to minimize the error and then snap to half-integers.
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Time budget per rank attempt
    max_iters = 2000
    step_size = 0.5
    
    for iter in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        # Compute current error
        error = compute_error(U, V, W, n, m, p)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-10:
            # Found a solution, now snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
            if snap_error < 1e-10:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            else:
                # If snap doesn't work, continue searching or accept float if allowed? 
                # Task says "integer/half-integer entries". So we must snap.
                # If snap fails, we can't return this.
                pass

        # Random perturbation
        # Pick a random matrix and a random entry
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            i, j = np.random.randint(0, n*m), np.random.randint(0, rank)
            U[i, j] += np.random.choice([-step_size, step_size])
        elif mat_idx == 1:
            i, j = np.random.randint(0, m*p), np.random.randint(0, rank)
            V[i, j] += np.random.choice([-step_size, step_size])
        else:
            i, j = np.random.randint(0, p*n), np.random.randint(0, rank)
            W[i, j] += np.random.choice([-step_size, step_size])

    # If we found a good float solution but didn't snap, try snapping the best
    if best_error < 1e-6:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
        if snap_error < 1e-10:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    return None

def compute_error(U, V, W, n, m, p):
    # Compute sum_{i,j,k} (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1)^2
    # Efficient way:
    # Let A = U (n*m x r), B = V (m*p x r), C = W (p*n x r)
    # We want to compute the tensor T[i, j, k] = sum_r A[i*m+j, r] B[j*p+k, r] C[k*n+i, r]
    
    # This is a bit tricky to vectorize directly as a 3D tensor without loops or einsum.
    # Using einsum:
    # U is (n*m, r). Reshape to (n, m, r).
    # V is (m*p, r). Reshape to (m, p, r).
    # W is (p*n, r). Reshape to (p, n, r).
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # Note the index swap in W3: W[k, i, r]
    
    # We can compute this via einsum
    # T = np.einsum('ijr,jkr,kin->ikn', U3, V3, W3) # This gives T[i, k, n]? No.
    # Let's be careful.
    # U3 indices: i, j, r
    # V3 indices: j, k, r
    # W3 indices: k, i, r (since W is p*n, index k*n+i, so row k, col i)
    
    # We want T[i, j, k]
    # T[i, j, k] = sum_r U3[i, j, r] V3[j, k, r] W3[k, i, r]
    
    # einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Target is 1 everywhere
    error = np.sum((T - 1.0)**2)
    return error
