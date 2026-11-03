"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition exactly reproduces T."""
        recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(recon, T, atol=1e-10)
    
    def peel_greedy(seed, max_iters=60):
        """Greedy rank-1 peeling with better initialization."""
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if np.allclose(residual, 0, atol=1e-12):
                break
            
            # Try to find a good rank-1 component
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            n_trials = 300
            for t in range(n_trials):
                # Random u and v with some structure
                if t % 3 == 0:
                    # Try sparse-ish patterns
                    u = rng.choice([-1.0, 0.0, 1.0], size=n * m)
                    v = rng.choice([-1.0, 0.0, 1.0], size=m * p)
                elif t % 3 == 1:
                    # Random Gaussian
                    u = rng.randn(n * m)
                    v = rng.randn(m * p)
                else:
                    # Block-structured
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    for i in range(n):
                        for j in range(m):
                            u[i * m + j] = rng.choice([-1.0, 1.0])
                    for j in range(m):
                        for k in range(p):
                            v[j * p + k] = rng.choice([-1.0, 1.0])
                
                # Reshape for computation
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                # Compute optimal w
                num = np.zeros((n, p))
                den = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        uj = u_mat[i, :]
                        vk = v_mat[:, k]
                        ujvk = uj * vk
                        num[i, k] = np.dot(residual[i, :, k], ujvk)
                        den[i, k] = np.dot(ujvk, ujvk)
                
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                for i in range(n):
                    for k in range(p):
                        if mask[i, k]:
                            w_flat[k * n + i] = num[i, k] / den[i, k]
                
                # Score: how much does this reduce the residual norm?
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_flat[k * n + i]
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            # Check if this peel actually helps
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] = best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]
            
            new_residual = residual - recon
            old_norm = np.sum(residual * residual)
            new_norm = np.sum(new_residual * new_residual)
            
            if new_norm >= old_norm - 1e-12:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        if verify_decomp(U, V, W, rank):
            return U, V, W, rank
        else:
            return None
    
    # Try multiple seeds
    for seed in range(30):
        if time.time() > deadline - 15.0:
            break
        result = peel_greedy(seed)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
                print(f"Found rank {rank} with seed {seed}")
    
    # Try a structured approach: exploit the fact that T is all-ones
    # The tensor T[i,j,k] = 1 for all i,j,k
    # This means sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    # 
    # One simple decomposition: use rank = n*p terms
    # For each (i,k), set U[i*m+j, r(i,k)] = 1/m, V[j*p+k, r(i,k)] = 1, W[k*n+i, r(i,k)] = m
    # Then sum_j (1/m) * 1 * m = 1
    # Wait, that's not right because V depends on j too.
    
    # Alternative: rank = n*m*p is schoolbook. Let's try rank = n*p.
    # For each (i,k), we need sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # If we set W[k*n+i, r(i,k)] = 1, then we need sum_j U[i*m+j,r] * V[j*p+k,r] = 1
    # This is like saying the (i,k) block of U^T V has sum = 1.
    
    # Try a different structured approach
    def structured_decomposition():
        """Try rank = n*p decomposition."""
        rank = n * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for i in range(n):
            for k in range(p):
                r = i * p + k
                # Set W[k*n + i, r] = 1
                W[k * n + i, r] = 1.0
                # We need sum_j U[i*m+j, r] * V[j*p+k, r] = 1
                # Simple choice: U[i*m+j, r] = 1/sqrt(m) for all j, V[j*p+k, r] = 1/sqrt(m) for all j
                # Then sum_j (1/sqrt(m)) * (1/sqrt(m)) = m * (1/m) = 1
                for j in range(m):
                    U[i * m + j, r] = 1.0 / np.sqrt(m)
                    V[j * p + k, r] = 1.0 / np.sqrt(m)
        
        if verify_decomp(U, V, W, rank):
            return U, V, W, rank
        return None
    
    result = structured_decomposition()
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_decomp = (U, V, W)
            print(f"Found structured rank {rank}")
    
    # Try another structured approach: rank = m
    # For each j, set U[i*m+j, r(j)] = 1, V[j*p+k, r(j)] = 1, W[k*n+i, r(j)] = 1/m
    # Then sum_j 1 * 1 * (1/m) = 1
    def structured_decomp_m():
        rank = m
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for j in range(m):
            r = j
            for i in range(n):
                U[i * m + j, r] = 1.0
            for k in range(p):
                V[j * p + k, r] = 1.0
            for i in range(n):
                for k in range(p):
                    W[k * n + i, r] = 1.0 / m
        
        if verify_decomp(U, V, W, rank):
            return U, V, W, rank
        return None
    
    result = structured_decomp_m()
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_decomp = (U, V, W)
            print(f"Found structured m-rank {rank}")
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

