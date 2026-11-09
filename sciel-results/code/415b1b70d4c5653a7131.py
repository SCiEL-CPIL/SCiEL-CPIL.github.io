import time
import itertools
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-6):
    # Reconstruct tensor from factors
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is expensive to check fully, but we can check on random inputs or specific structure.
    # For exactness, we should check the tensor entries.
    # The tensor for matrix multiplication is T[i, j, k] = 1 if indices match, 0 otherwise?
    # No, the tensor is the structure of the multiplication.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # T is a 3-tensor. T[i, j, k] = 1.
    # The decomposition is T = sum_r u_r v_r w_r.
    # u_r is vector of size nm, v_r of size mp, w_r of size pn.
    # (u_r v_r w_r)[i, j, k] = u_r[i, j] v_r[j, k] w_r[k, i]
    # We want sum_r u_r[i, j] v_r[j, k] w_r[k, i] = 1 for all i, j, k.
    
    U_flat = U.reshape(-1, U.shape[1])
    V_flat = V.reshape(-1, V.shape[1])
    W_flat = W.reshape(-1, W.shape[1])
    
    # Check a subset of entries to save time if needed, but 3*4*8=96 is small.
    # We check all 96 entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, target_rank, max_iters=500, time_limit=20.0):
    """
    Attempts to find a decomposition of rank target_rank for M_{n,m,p}.
    Uses least squares to minimize the error of the tensor reconstruction.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize with random factors
    # We use a "warm start" from schoolbook if possible, but target_rank < nm*mp*p
    # So we just use random init with small variance.
    
    best_error = float('inf')
    best_solution = None
    
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        def residuals(params):
            U_r = params[0:n*target_rank].reshape(n, target_rank)
            V_r = params[n*target_rank:n*m*target_rank].reshape(m, target_rank)
            W_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(p, target_rank)
            
            # This mapping is tricky. U is nm x rank.
            # Let's just flatten U, V, W directly.
            pass
        
        # Better: flatten U, V, W into a single vector x
        # x = [U.flatten(), V.flatten(), W.flatten()]
        # U = x[0:nm*R].reshape(nm, R)
        # V = x[nm*R:nm*R+mp*R].reshape(mp, R)
        # W = x[nm*R+mp*R:].reshape(pn, R)
        
        def objective(x):
            U_r = x[0:n*target_rank*target_rank].reshape(n, target_rank, target_rank) # No, U is nm x R
            # Correct slicing:
            U_flat_len = nm * target_rank
            V_flat_len = mp * target_rank
            W_flat_len = pn * target_rank
            
            U_r = x[0:U_flat_len].reshape(nm, target_rank)
            V_r = x[U_flat_len:U_flat_len+V_flat_len].reshape(mp, target_rank)
            W_r = x[U_flat_len+V_flat_len:].reshape(pn, target_rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                        res.append(val - 1.0)
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(objective, x0, method='lm', max_nfev=max_iters)
            if result.cost < best_error:
                best_error = result.cost
                if best_error < 1e-8:
                    best_solution = result.x
                    break
        except Exception:
            continue
            
    if best_solution is not None:
        U_flat_len = nm * target_rank
        V_flat_len = mp * target_rank
        U_out = best_solution[0:U_flat_len].reshape(nm, target_rank)
        V_out = best_solution[U_flat_len:U_flat_len+V_flat_len].reshape(mp, target_rank)
        W_out = best_solution[U_flat_len+V_flat_len:].reshape(pn, target_rank)
        
        if _verify(U_out, V_out, W_out, n, m, p):
            return {"U": U_out, "V": V_out, "W": W_out}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Try to improve on 3x4x4 block
    # If we find rank R for 3x4x4, we can use 2*R for 3x4x8.
    # Schoolbook 3x4x4 is 48.
    # Try ranks 47, 46, 45
    
    best_rank_344 = 48
    best_decomp_344 = _schoolbook(3, 4, 4)
    
    for r in [47, 46, 45]:
        if time.time() > deadline:
            break
        print(f"Trying rank {r} for 3x4x4...")
        decomp = _try_find_decomposition(3, 4, 4, r, time_limit=30.0)
        if decomp is not None:
            best_rank_344 = r
            best_decomp_344 = decomp
            print(f"Found valid decomposition for 3x4x4 with rank {r}")
            break
        else:
            print(f"Failed to find rank {r} for 3x4x4")
            
    # Construct 3x4x8 from 3x4x4
    # B (4x8) = [B1 (4x4), B2 (4x4)]
    # C (3x8) = [C1 (3x4), C2 (3x4)]
    # C1 = A B1, C2 = A B2
    # Rank 3x4x8 = Rank 3x4x4 + Rank 3x4x4 = 2 * best_rank_344
    
    R = best_rank_344
    if R < 48:
        U344 = best_decomp_344["U"]
        V344 = best_decomp_344["V"]
        W344 = best_decomp_344["W"]
        
        # U is 12 x R. For 3x4x8, U is 12 x 2R.
        # U8 = [U344, U344]
        U8 = np.hstack([U344, U344])
        
        # V is 32 x 2R.
        # V344 is 16 x R.
        # For B1 (first 4 cols of B), V corresponds to indices j*p+k where p=4?
        # No, the tensor V for 3x4x4 has indices j (0..3) and k (0..3). Size 16.
        # For 3x4x8, V has indices j (0..3) and k (0..7). Size 32.
        # The first R terms correspond to B1 (k=0..3).
        # The next R terms correspond to B2 (k=4..7).
        
        V8 = np.zeros((32, 2*R))
        # First half: V344 mapped to k=0..3
        for j in range(4):
            for k in range(4):
                V8[j*8+k, :R] = V344[j*4+k, :]
        
        # Second half: V344 mapped to k=4..7
        for j in range(4):
            for k in range(4):
                V8[j*8+(k+4), R:] = V344[j*4+k, :]
                
        # W is 24 x 2R.
        # W344 is 12 x R. Indices k (0..3) and i (0..2).
        # For 3x4x8, W has indices k (0..7) and i (0..2). Size 24.
        # First half: W344 mapped to k=0..3.
        for k in range(4):
            for i in range(3):
                W8[k*3+i, :R] = W344[k*3+i, :]
                
        # Second half: W344 mapped to k=4..7.
        for k in range(4):
            for i in range(3):
                W8[(k+4)*3+i, R:] = W344[k*3+i, :]
                
        # Verify the full 3x4x8 decomposition
        if _verify(U8, V8, W8, 3, 4, 8):
            return {"U": U8, "V": V8, "W": W8}
        else:
            print("Verification failed for constructed 3x4x8 decomposition.")
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
