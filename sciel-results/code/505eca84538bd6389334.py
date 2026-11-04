import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s margin before 150s kill

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

    # Fast verification using einsum
    def verify_fast(U, V, W):
        # Compute sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] and compare to T
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(result, T, atol=1e-10)

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify_fast(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Fast column elimination: try removing one column at a time
    # Use fast verification
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for col_to_remove in range(best_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(best_U, col_to_remove, axis=1)
            V_test = np.delete(best_V, col_to_remove, axis=1)
            W_test = np.delete(best_W, col_to_remove, axis=1)
            if verify_fast(U_test, V_test, W_test):
                best_U, best_V, best_W, best_rank = U_test, V_test, W_test, best_rank - 1
                improved = True
                break

    # Try to find lower-rank decompositions using a smarter approach:
    # For each target rank, try to solve the linear system directly
    # The tensor decomposition T = sum_c U_c ⊗ V_c ⊗ W_c
    # is a system of linear equations in the entries of U, V, W
    
    # Alternative: use the fact that we can fix one matrix and solve for others
    # Let's try a structured approach: fix W to have a specific structure
    
    for target_rank in range(best_rank - 1, max(best_rank - 15, 30), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with faster ALS
        for restart in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(500 + restart * 13)
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Fast ALS: vectorized residual computation
            converged = False
            for iteration in range(30):
                if time.time() > deadline:
                    break
                
                # Compute current approximation
                approx = np.einsum('ac,bc,dc->abd', U_try, V_try, W_try)
                residual = T - approx
                
                # Update each column of U, V, W
                for c in range(target_rank):
                    # Update U[:,c]
                    # residual = U[:,c] ⊗ (V[:,c] ⊗ W[:,c]) + other terms
                    # So U[:,c] = <residual, V[:,c] ⊗ W[:,c]> / ||V[:,c] ⊗ W[:,c]||²
                    vw = np.outer(V_try[:,c], W_try[:,c])  # shape (m*p, p*n)
                    denom = np.sum(vw * vw)
                    if abs(denom) > 1e-12:
                        # U[:,c] = sum over b,d of residual[a,b,d] * V[b,c] * W[d,c]
                        U_try[:,c] = np.einsum('abd,b,d->a', residual, V_try[:,c], W_try[:,c]) / denom
                    
                    # Update V[:,c]
                    uw = np.outer(U_try[:,c], W_try[:,c])  # shape (n*m, p*n)
                    denom = np.sum(uw * uw)
                    if abs(denom) > 1e-12:
                        V_try[:,c] = np.einsum('abd,a,d->b', residual, U_try[:,c], W_try[:,c]) / denom
                    
                    # Update W[:,c]
                    uv = np.outer(U_try[:,c], V_try[:,c])  # shape (n*m, m*p)
                    denom = np.sum(uv * uv)
                    if abs(denom) > 1e-12:
                        W_try[:,c] = np.einsum('abd,a,b->d', residual, U_try[:,c], V_try[:,c]) / denom
                
                # Check convergence
                approx = np.einsum('ac,bc,dc->abd', U_try, V_try, W_try)
                if np.linalg.norm(approx - T) < 1e-8:
                    converged = True
                    break
            
            if converged:
                if verify_fast(U_try, V_try, W_try):
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                    break

    # Final: try one more round of column elimination on the best result
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for col_to_remove in range(best_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(best_U, col_to_remove, axis=1)
            V_test = np.delete(best_V, col_to_remove, axis=1)
            W_test = np.delete(best_W, col_to_remove, axis=1)
            if verify_fast(U_test, V_test, W_test):
                best_U, best_V, best_W, best_rank = U_test, V_test, W_test, best_rank - 1
                improved = True
                break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
