import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=tol)
    
    # Rank 50 Strassen decomposition
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
        
        # 6 Strassen blocks for k=0..5
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)
        
        # 8 direct terms for k=6
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
        
        return U, V, W, rank
    
    U, V, W, best_rank = strassen_rank50()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        best_rank = n * m * p
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Try to reduce rank by removing linearly dependent columns
    for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
        if time.time() > deadline:
            break
        
        if target_rank == best_rank - 1:
            for col_to_remove in range(best_rank):
                if time.time() > deadline:
                    break
                U_test = np.delete(U, col_to_remove, axis=1)
                V_test = np.delete(V, col_to_remove, axis=1)
                W_test = np.delete(W, col_to_remove, axis=1)
                if verify(U_test, V_test, W_test):
                    U, V, W, best_rank = U_test, V_test, W_test, target_rank
                    break
            continue
        
        # Try random ALS for lower ranks
        for restart in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart)
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    vw = V_try[:,c] * W_try[:,c]
                    denom = np.sum(vw * vw)
                    if abs(denom) > 1e-12:
                        U_try[:,c] = np.einsum('abk,bk->a', residual, vw) / denom
                
                # Update V
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    uw = U_try[:,c] * W_try[:,c]
                    denom = np.sum(uw * uw)
                    if abs(denom) > 1e-12:
                        V_try[:,c] = np.einsum('abk,ak->b', residual, uw) / denom
                
                # Update W
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    uv = U_try[:,c] * V_try[:,c]
                    denom = np.sum(uv * uv)
                    if abs(denom) > 1e-12:
                        W_try[:,c] = np.einsum('abk,ab->k', residual, uv) / denom
                
                if verify(U_try, V_try, W_try, tol=1e-8):
                    break
            
            if verify(U_try, V_try, W_try):
                U, V, W, best_rank = U_try, V_try, W_try, target_rank
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
