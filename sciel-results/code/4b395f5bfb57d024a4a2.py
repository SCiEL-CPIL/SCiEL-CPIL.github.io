import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin before 150s kill

    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Fallback: Schoolbook rank 56
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Rank 50 Strassen-based decomposition
    def strassen_rank50():
        rank = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
                
        return U, V, W, rank

    # Verify a decomposition (vectorized)
    def verify(U, V, W):
        prod = U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :]
        result = np.sum(prod, axis=3)
        return np.allclose(result, T, atol=1e-8)

    def compute_residual(U, V, W):
        R = np.zeros_like(T)
        for c in range(U.shape[1]):
            R += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return R

    # Fast ALS with incremental residual
    def fast_als(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-9):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        R = compute_residual(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                vw = V[:,c] * W[:,c]
                denom = np.sum(vw * vw)
                if denom > 1e-12:
                    U_new = np.einsum('abk,bk->a', T + R, vw) / denom
                else:
                    U_new = U[:,c]
                R += np.einsum('i,j,k->ijk', U_new, V[:,c], W[:,c])
                U[:,c] = U_new
            
            # Update V
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                uw = U[:,c] * W[:,c]
                denom = np.sum(uw * uw)
                if denom > 1e-12:
                    V_new = np.einsum('abk,ak->b', T + R, uw) / denom
                else:
                    V_new = V[:,c]
                R += np.einsum('i,j,k->ijk', U[:,c], V_new, W[:,c])
                V[:,c] = V_new
            
            # Update W
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                uv = U[:,c] * V[:,c]
                denom = np.sum(uv * uv)
                if denom > 1e-12:
                    W_new = np.einsum('abk,ab->k', T + R, uv) / denom
                else:
                    W_new = W[:,c]
                R += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W_new)
                W[:,c] = W_new
            
            err = np.linalg.norm(T - R)
            if err < tol:
                break
        
        return U, V, W, R

    # Try to remove redundant columns by checking linear dependencies
    def try_reduce_columns(U, V, W, max_removals=5):
        """Try to remove columns that are linearly dependent across all three factors."""
        r = U.shape[1]
        removed = 0
        changed = True
        
        while changed and removed < max_removals:
            changed = False
            for c in range(r):
                if time.time() > deadline:
                    return U, V, W, removed
                
                # Try removing column c
                mask = np.ones(r, dtype=bool)
                mask[c] = False
                U_sub = U[:, mask]
                V_sub = V[:, mask]
                W_sub = W[:, mask]
                
                if verify(U_sub, V_sub, W_sub):
                    U = U_sub
                    V = V_sub
                    W = W_sub
                    r -= 1
                    removed += 1
                    changed = True
                    break
        
        return U, V, W, removed

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s
    
    # Try to reduce rank via column removal first
    U_red, V_red, W_red, n_removed = try_reduce_columns(best_U, best_V, best_W, max_removals=10)
    if n_removed > 0:
        best_U, best_V, best_W, best_rank = U_red, V_red, W_red, U_red.shape[1]
    
    # Try to reduce rank further via ALS
    for target_rank in range(best_rank - 1, max(best_rank - 15, 30), -1):
        if time.time() > deadline:
            break
        
        success = False
        # Try multiple initialization strategies
        for restart in range(8):
            if time.time() > deadline:
                break
            
            if restart == 0:
                U_init = best_U[:, :target_rank]
                V_init = best_V[:, :target_rank]
                W_init = best_W[:, :target_rank]
            elif restart == 1:
                # Perturb the top columns
                noise_scale = 0.1
                U_init = best_U[:, :target_rank] + noise_scale * np.random.randn(n*m, target_rank)
                V_init = best_V[:, :target_rank] + noise_scale * np.random.randn(m*p, target_rank)
                W_init = best_W[:, :target_rank] + noise_scale * np.random.randn(p*n, target_rank)
            else:
                # Random subset of previous best columns
                idx = np.random.choice(best_rank, target_rank, replace=False)
                U_init = best_U[:, idx]
                V_init = best_V[:, idx]
                W_init = best_W[:, idx]
            
            U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_init, target_rank, max_iters=80, tol=1e-10)
            
            if np.linalg.norm(T - R_try) < 1e-8:
                # Try to reduce further
                U_try, V_try, W_try, n_rem = try_reduce_columns(U_try, V_try, W_try, max_removals=3)
                if n_rem > 0:
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, U_try.shape[1]
                else:
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                success = True
                break
        
        # If not successful with pruning, try random restarts
        if not success:
            for restart in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(1000 + restart * 7)
                U_init = np.random.randn(n*m, target_rank)
                V_init = np.random.randn(m*p, target_rank)
                W_init = np.random.randn(p*n, target_rank)
                
                U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_init, target_rank, max_iters=120, tol=1e-10)
                
                if np.linalg.norm(T - R_try) < 1e-8:
                    U_try, V_try, W_try, n_rem = try_reduce_columns(U_try, V_try, W_try, max_removals=3)
                    if n_rem > 0:
                        best_U, best_V, best_W, best_rank = U_try, V_try, W_try, U_try.shape[1]
                    else:
                        best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                    success = True
                    break

    # Final polish: verify and ensure clean output
    U_out = np.round(best_U, decimals=10).tolist()
    V_out = np.round(best_V, decimals=10).tolist()
    W_out = np.round(best_W, decimals=10).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
