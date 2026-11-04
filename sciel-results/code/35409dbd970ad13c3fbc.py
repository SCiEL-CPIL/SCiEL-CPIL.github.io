import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135  # 15s margin before 150s kill

    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Verify a decomposition
    def verify(U, V, W):
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-9)

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

        # 6 Strassen blocks for 2x4x6 (columns 0-5)
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)

        # Schoolbook for column 6
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1

        return U, V, W, rank

    # Alternative: 2x4x7 split as 2x4x4 + 2x4x2 + 2x4x1
    # 2x4x4: 2 Strassen blocks = 14
    # 2x4x2: 1 Strassen block = 7  
    # 2x4x1: 2 schoolbook = 2
    # Total: 23? No, that's wrong because the Strassen blocks overlap in structure
    # Actually for 2x4x4: columns 0-3, we need 2 Strassen blocks (j=0,1 and j=2,3)
    # For 2x4x2: columns 4-5, we need 1 Strassen block
    # For 2x4x1: column 6, we need 2 schoolbook
    # But wait - the Strassen for 2x2x2 uses both rows of A, so for 2x4x4:
    #   Block 1: i=0,1; j=0,1; k=0,1 -> rank 7
    #   Block 2: i=0,1; j=2,3; k=0,1 -> rank 7
    #   Block 3: i=0,1; j=0,1; k=2,3 -> rank 7
    #   Block 4: i=0,1; j=2,3; k=2,3 -> rank 7
    # That's 28 for 2x4x4, plus 7 for 2x4x2, plus 2 for 2x4x1 = 37
    # But that's more than 50? No, 28+7+2=37 < 50. Let me recheck.
    # Actually the current code uses 6 Strassen blocks = 42 + 8 = 50
    # 6 blocks: (0,0,0),(0,0,2),(0,2,0),(0,2,2) = 4 for 2x4x4
    #            (0,0,4),(0,2,4) = 2 for 2x4x2
    # That's 4*7 + 2*7 + 8 = 28+14+8 = 50. Correct.
    # But we only need 4 Strassen blocks for 2x4x4 (covering all 4 columns in pairs)
    # and 2 for 2x4x2. That's what we have.

    # Try Winograd-style: F(4,4,4) has rank 18 for 4x4x4
    # For 2x4x4, we can use a restricted version
    # F(4,2,4) has rank 8 for 4x2x4
    # For 2x4x4: think of it as (2x4) x (4x4), using F(4,4,4) restricted to 2 rows
    # This gives rank 18 for the 4x4 part but we only use 2 rows of A
    # Actually, let's use the known optimal: rank(2x4x4) = 16 (by Winograd F(4,4,4) on 4x4x4,
    # but we only need 2 rows so it might be less)
    # Known: rank(2x4x4) <= 16, and it's conjectured to be 16
    # If we use rank 16 for 2x4x4, rank 7 for 2x4x2, rank 2 for 2x4x1: total = 25
    # But we need to verify this is valid.

    # Let's try a direct Winograd F(4,4,4) approach for the 4x4x4 part
    # F(4,4,4): 18 multiplications for 4x4x4
    # The transform: A' = M_A @ A, B' = M_B @ B, C = M_C @ (A' * B')
    # For 2x4x4: we use only 2 rows of A, so A' has 2*4=8 entries transformed
    # This is complex. Let me use a simpler approach.

    # Best bet: use the rank-50 solution and try to reduce via linear algebra
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        # Fallback schoolbook
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        rank_s = rank

    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank: check if any column is redundant
    # For each column, check if removing it still gives a valid decomposition
    # (This only works if there are exact zeros, which there aren't in Strassen)
    
    # More powerful: use a Gauss-Newton / Levenberg-Marquardt approach
    # to find a lower-rank solution
    # Start from the rank-50 solution, try to project to rank 49
    
    # Alternative: use the fact that the tensor has a specific structure
    # and try to find a rank-48 or lower decomposition using random initialization
    # with ALS (Alternating Least Squares)
    
    # ALS for tensor rank decomposition
    def als_decompose(target_rank, n_iters=100, n_restarts=5):
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            np.random.seed(1000 + restart)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            for it in range(n_iters):
                if time.time() > deadline:
                    return None, None, None, None
                
                # Update U: for each column c, U[:,c] = argmin ||T - sum_c U[:,c] (V[:,c] outer W[:,c])||
                # Flatten T to 2D: T_flat[i, (j,k)] = T[i,j,k]
                # For fixed V,W: T = sum_c U[:,c] (V[:,c] outer W[:,c])
                # T_flat @ ... = ...
                # This is: for each i, T[i,:,:] = sum_c U[i,c] * (V[:,c] outer W[:,c])
                # Flatten (j,k) into single index: T2d[i, j*p*n + k] 
                # Design matrix: M[i, c] = ... this is complex
                
                # Simpler: use gradient descent
                # Compute current tensor
                current = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    current += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                
                error = current - T
                err_norm = np.linalg.norm(error)
                if err_norm < 1e-8:
                    return U, V, W, target_rank
                
                # Gradient computation
                # dL/dU[i,c] = sum_{j,k} error[i,j,k] * V[j,c] * W[k,c]
                # dL/dV[j,c] = sum_{i,k} error[i,j,k] * U[i,c] * W[k,c]
                # dL/dW[k,c] = sum_{i,j} error[i,j,k] * U[i,c] * V[j,c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(target_rank):
                    dU[:,c] = np.einsum('ijk,j,k->i', error, V[:,c], W[:,c])
                    dV[:,c] = np.einsum('ijk,i,k->j', error, U[:,c], W[:,c])
                    dW[:,c] = np.einsum('ijk,i,j->k', error, U[:,c], V[:,c])
                
                # Adaptive learning rate
                lr = 0.001
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Normalize to prevent drift
                for c in range(target_rank):
                    u_norm = np.linalg.norm(U[:,c])
                    v_norm = np.linalg.norm(V[:,c])
                    w_norm = np.linalg.norm(W[:,c])
                    if u_norm > 0 and v_norm > 0 and w_norm > 0:
                        # Keep product of norms constant
                        target_norm = (u_norm * v_norm * w_norm) ** (1/3)
                        U[:,c] *= target_norm / u_norm
                        V[:,c] *= target_norm / v_norm
                        W[:,c] *= target_norm / w_norm
                        if np.linalg.norm(U[:,c]) < 1e-10:
                            U[:,c] += 1e-10 * np.random.randn(n*m)
                        if np.linalg.norm(V[:,c]) < 1e-10:
                            V[:,c] += 1e-10 * np.random.randn(m*p)
                        if np.linalg.norm(W[:,c]) < 1e-10:
                            W[:,c] += 1e-10 * np.random.randn(p*n)
        
        return None, None, None, None

    # Try rank reduction from 49 down to 45
    for target_rank in range(49, 44, -1):
        if time.time() > deadline:
            break
        
        U_new, V_new, W_new, r = als_decompose(target_rank, n_iters=80, n_restarts=3)
        if U_new is not None and verify(U_new, V_new, W_new):
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, target_rank
            break

    # Final: try to remove redundant columns from best solution
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        for c in range(best_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(best_U, c, axis=1)
            V_test = np.delete(best_V, c, axis=1)
            W_test = np.delete(best_W, c, axis=1)
            if verify(U_test, V_test, W_test):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                improved = True
                break
        if not improved:
            break

    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
