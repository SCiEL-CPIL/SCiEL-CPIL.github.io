"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def schoolbook():
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

    def verify_fast(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    def snap_to_valid(X):
        # Snap to nearest 0, 0.5, 1, -0.5, -1
        # Actually, the set is {0, ±1, ±0.5}
        # We can just round to nearest 0.5 and check if it's in the set
        # But 1.5 is not allowed.
        # So we round to nearest 0.5, then if it's > 1 or < -1, clamp? 
        # No, the problem says entries are integers or half-integers.
        # It doesn't explicitly restrict magnitude, but typically small.
        # However, the set {0, ±1, ±0.5} is implied by "integer/half-integer" 
        # in many contexts of this task, but let's assume any integer or half-integer is allowed.
        # Wait, the prompt says "integer/half-integer entries". 
        # The inspiration 3 mentions "coefficients must be integers or half-integers".
        # It does NOT say they must be in {-1, -0.5, 0, 0.5, 1}.
        # So we can snap to any multiple of 0.5.
        
        X_snapped = np.round(X * 2) / 2
        return X_snapped

    def als_decompose(r, max_iters=50):
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Precompute indices for efficiency? 
        # The loops are small enough that direct indexing is fine.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    # Construct M (p x r)
                    # M[k, c] = V[j*p + k, c] * W[k*n + i, c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve M u = 1
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[idx, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    # Construct M (n x r)
                    # M[i, c] = U[i*m + j, c] * W[k*n + i, c]
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    # Solve M v = 1
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[idx, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    # Construct M (m x r)
                    # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve M w = 1
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[idx, :] = w
        
        # Snap
        U_snapped = snap_to_valid(U)
        V_snapped = snap_to_valid(V)
        W_snapped = snap_to_valid(W)
        
        # Convert to lists
        U_list = U_snapped.tolist()
        V_list = V_snapped.tolist()
        W_list = W_snapped.tolist()
        
        return U_list, V_list, W_list

    best_result = schoolbook()
    best_rank = n * m * p

    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U, V, W = als_decompose(r, max_iters=20)
            
            if verify_integer(U, V, W) and verify_fast(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                found = True
                break
        
        if found:
            # If we found a solution, we can try to go lower? 
            # No, we are iterating from low to high. 
            # If we found one at r, it's the minimal we found so far.
            # We can break if we are confident, but let's just return.
            # Actually, we should check if we can go even lower? 
            # The loop starts from lower_bound. So if we find it at 28, it's minimal.
            # If we find it at 30, maybe 29 works? 
            # But we already tried 29. 
            # So we can break.
            break

    return best_result
# EVOLVE-BLOCK-END
