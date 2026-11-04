import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin before 150s kill

    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a, b, c correspond to a valid multiplication term
    # a = i*m + j, b = j*p + k, c = k*n + i
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
            # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3: A11(B12-B22) -> C12 += M3, C22 += M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4: A22(B21-B11) -> C11 += M4, C21 += M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A21-A11)(B11+B12) -> C22 += M6
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A12-A22)(B21+B22) -> C11 += M7
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        # Part 1: 2x4x4 (k=0..3) - 4 Strassen blocks
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        
        # Part 2: 2x4x2 (k=4..5) - 2 Strassen blocks
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
                
        return U, V, W, rank

    # Verify a decomposition (vectorized)
    def verify(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reshape to (nm, 1, 1, r), (1, mp, 1, r), (1, 1, pn, r)
        # Product: (nm, mp, pn, r)
        # Sum over r: (nm, mp, pn)
        prod = U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :]
        result = np.sum(prod, axis=3)
        return np.allclose(result, T, atol=1e-8)

    # Fast ALS with incremental residual
    def fast_als(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-9):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Compute initial residual R = sum(U_c V_c W_c)
        R = np.zeros_like(T)
        for c in range(target_rank):
            R += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for c in range(target_rank):
                # Remove current contribution
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                
                vw = V[:,c] * W[:,c]
                denom = np.sum(vw * vw)
                if denom > 1e-12:
                    U_new = np.einsum('abk,bk->a', T + R, vw) / denom
                else:
                    U_new = U[:,c]
                
                # Add new contribution
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
            
            # Check convergence
            err = np.linalg.norm(T - R)
            if err < tol:
                break
        
        return U, V, W, R

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank
    # Strategy: Try ranks from best_rank-1 down to best_rank-10
    for target_rank in range(best_rank - 1, max(best_rank - 10, 30), -1):
        if time.time() > deadline:
            break
        
        success = False
        # Try multiple initialization strategies
        for restart in range(5):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Use top columns of previous best
                U_init = best_U[:, :target_rank]
                V_init = best_V[:, :target_rank]
                W_init = best_W[:, :target_rank]
            else:
                # Random subset of previous best columns
                idx = np.random.choice(best_rank, target_rank, replace=False)
                U_init = best_U[:, idx]
                V_init = best_V[:, idx]
                W_init = best_W[:, idx]
            
            U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_init, target_rank, max_iters=50, tol=1e-10)
            
            if np.linalg.norm(T - R_try) < 1e-8:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                success = True
                break
        
        # If not successful with pruning, try random restarts
        if not success:
            for restart in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(1000 + restart)
                U_init = np.random.randn(n*m, target_rank)
                V_init = np.random.randn(m*p, target_rank)
                W_init = np.random.randn(p*n, target_rank)
                
                U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-10)
                
                if np.linalg.norm(T - R_try) < 1e-8:
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                    success = True
                    break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
