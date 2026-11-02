"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for c in range(rank):
        u = np.array(U[:, c]).reshape(n, m)
        v = np.array(V[:, c]).reshape(m, p)
        w = np.array(W[:, c]).reshape(p, n)
        # Check that the outer product u ⊗ v ⊗ w equals the corresponding slice
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix multiplication, T[i,j,k] = 1 if i,j,k are valid indices
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # Wait, that's not right. Let me think again.
        # The tensor for matrix multiplication A(2x3) * B(3x4) = C(2x4)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,4)
        # And we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k] = 1
        pass
    
    # Better: build the full tensor and check
    T_target = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u = np.array(U[:, c]).reshape(n, m)
        v = np.array(V[:, c]).reshape(m, p)
        w = np.array(W[:, c]).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _try_random_decomposition(n, m, p, max_rank, num_trials=100, seed=42):
    """Try to find a low-rank decomposition using random initialization and refinement."""
    rng = np.random.RandomState(seed)
    best_rank = max_rank
    best_decomp = None
    
    # Start with schoolbook as baseline
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
    best_decomp = (U, V, W)
    best_rank = rank
    
    # Try various ranks from low to high
    for target_rank in range(1, min(max_rank + 1, n * m * p)):
        if target_rank >= best_rank:
            break
            
        for trial in range(num_trials):
            # Random initialization
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Simple iterative refinement: try to minimize ||T_recon - T_target||_F
            # Using a simple gradient-free approach: randomly perturb and keep if better
            current_error = None
            U_cur = U_rand.copy()
            V_cur = V_rand.copy()
            W_cur = W_rand.copy()
            
            # Compute initial error
            T_recon = np.zeros((n, m, p))
            for c in range(target_rank):
                u = U_cur[:, c].reshape(n, m)
                v = V_cur[:, c].reshape(m, p)
                w = W_cur[:, c].reshape(p, n)
                T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
            T_target = np.ones((n, m, p))
            current_error = np.linalg.norm(T_recon - T_target)
            
            # Try a few iterations of random perturbation
            for it in range(20):
                # Pick a random component to perturb
                comp = rng.randint(3)  # 0=U, 1=V, 2=W
                idx1 = rng.randint(0, [n*m, m*p, p*n][comp])
                idx2 = rng.randint(0, target_rank)
                
                old_val = [U_cur, V_cur, W_cur][comp][idx1, idx2]
                new_val = old_val + rng.randn() * 0.1
                [U_cur, V_cur, W_cur][comp][idx1, idx2] = new_val
                
                T_recon = np.zeros((n, m, p))
                for c in range(target_rank):
                    u = U_cur[:, c].reshape(n, m)
                    v = V_cur[:, c].reshape(m, p)
                    w = W_cur[:, c].reshape(p, n)
                    T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
                new_error = np.linalg.norm(T_recon - T_target)
                
                if new_error < current_error:
                    current_error = new_error
                else:
                    [U_cur, V_cur, W_cur][comp][idx1, idx2] = old_val
            
            # Check if we got close to exact
            if current_error < 1e-4:
                # Try to snap to exact values
                # For now, just record this as a candidate
                U_list = [[U_cur[i, c] for c in range(target_rank)] for i in range(n * m)]
                V_list = [[V_cur[i, c] for c in range(target_rank)] for i in range(m * p)]
                W_list = [[W_cur[i, c] for c in range(target_rank)] for i in range(p * n)]
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_decomp = (U_list, V_list, W_list)
                    break
        
        if best_rank <= target_rank:
            break
    
    return best_rank, best_decomp

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Try ranks from 12 upward
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            
            rng = np.random.RandomState(42 + trial)
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Simple check: can we get close with these?
            # Use a crude test: compute the tensor and see if it's close to all-ones
            T_recon = np.zeros((n, m, p))
            for c in range(target_rank):
                u = U_rand[:, c].reshape(n, m)
                v = V_rand[:, c].reshape(m, p)
                w = W_rand[:, c].reshape(p, n)
                T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
            
            error = np.linalg.norm(T_recon - np.ones((n, m, p)))
            
            if error < 0.1:
                # Promising! Try to refine
                U_cur = U_rand.copy()
                V_cur = V_rand.copy()
                W_cur = W_rand.copy()
                
                for it in range(100):
                    if time.time() > deadline - 2:
                        break
                    
                    comp = trial % 3
                    idx1 = (it * 7 + trial) % [n*m, m*p, p*n][comp]
                    idx2 = (it * 13 + trial) % target_rank
                    
                    old_val = [U_cur, V_cur, W_cur][comp][idx1, idx2]
                    new_val = old_val + rng.randn() * 0.05
                    [U_cur, V_cur, W_cur][comp][idx1, idx2] = new_val
                    
                    T_recon = np.zeros((n, m, p))
                    for c in range(target_rank):
                        u = U_cur[:, c].reshape(n, m)
                        v = V_cur[:, c].reshape(m, p)
                        w = W_cur[:, c].reshape(p, n)
                        T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
                    new_error = np.linalg.norm(T_recon - np.ones((n, m, p)))
                    
                    if new_error < error:
                        error = new_error
                    else:
                        [U_cur, V_cur, W_cur][comp][idx1, idx2] = old_val
                
                if error < 1e-4:
                    U_list = [[U_cur[i, c] for c in range(target_rank)] for i in range(n * m)]
                    V_list = [[V_cur[i, c] for c in range(target_rank)] for i in range(m * p)]
                    W_list = [[W_cur[i, c] for c in range(target_rank)] for i in range(p * n)]
                    
                    if _validate(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        break
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
