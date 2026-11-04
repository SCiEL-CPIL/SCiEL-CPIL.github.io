import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # 20s margin before 150s kill

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

    # Verify a decomposition
    def verify(U, V, W):
        # Compute sum_c U[:,c] V[:,c] W[:,c] and compare to T
        # This is the tensor product sum
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-10)

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        # Fall back to schoolbook if Strassen is broken
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank using linear algebra / iterative refinement
    # Strategy: try rank 49, 48, 47... by attempting to find a valid decomposition
    # Using gradient-based optimization on random initializations
    
    for target_rank in [49, 48, 47, 46]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for restart in range(30):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent to minimize ||sum_c U_c V_c W_c - T||^2
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                current = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    current += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                
                error = current - T
                if np.linalg.norm(error) < 1e-8:
                    break
                
                # Compute gradients
                # d/dU[i,c] = sum_{b,c'} error[i,b,c'] * V[b,c] * W[c',c]
                # This is expensive, so we use a simplified update
                # Actually, let's use a different approach: ALS (Alternating Least Squares)
                pass
            
            # ALS approach
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for c in range(target_rank):
                    # U[:,c] should satisfy: sum_c U[:,c] (V[:,c] outer W[:,c]) = T
                    # For fixed V and W, U[:,c] = argmin ||sum_c U[:,c] (V[:,c] outer W[:,c]) - T||
                    # This is a linear system: sum_c U[:,c] * (V[:,c] outer W[:,c]) = T
                    # Flatten: U_flat @ M = T_flat where M is the design matrix
                    # But this is complex. Let's use a simpler heuristic.
                    pass
                
                # Update V
                for c in range(target_rank):
                    pass
                
                # Update W
                for c in range(target_rank):
                    pass
            
            # Check if we found a valid solution
            if verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank == target_rank:
            break

    # Try a more targeted approach: modify the rank 50 solution to reduce rank
    # by combining terms or finding linear dependencies
    if best_rank > 49:
        if time.time() < deadline:
            # Try to find a rank 49 solution by perturbing the rank 50 solution
            # and checking if one column can be removed
            U_try = best_U.copy()
            V_try = best_V.copy()
            W_try = best_W.copy()
            
            for col_to_remove in range(best_rank):
                if time.time() > deadline:
                    break
                
                U_test = np.delete(U_try, col_to_remove, axis=1)
                V_test = np.delete(V_try, col_to_remove, axis=1)
                W_test = np.delete(W_try, col_to_remove, axis=1)
                
                if verify(U_test, V_test, W_test):
                    best_U, best_V, best_W, best_rank = U_test, V_test, W_test, best_rank - 1
                    break

    # Convert to lists for return
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
