import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Tensor T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # In matrix multiplication C = A @ B, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor has 1s at (i,j,k) for all valid i,j,k.
    T = np.ones((n, m, p))
    
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    return np.allclose(T, T_recon, atol=1e-9)

def _strassen_rank14():
    """
    Construct the rank-14 decomposition using two independent Strassen 2x2x2 blocks.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    def add_strassen_block(start_r, b_col0, b_col1, c_col0, c_col1):
        # V indices: B[j, k] -> j*p + k
        v00 = 0 * p + b_col0   # B[0, b_col0]
        v01 = 0 * p + b_col1   # B[0, b_col1]
        v10 = 1 * p + b_col0   # B[1, b_col0]
        v11 = 1 * p + b_col1   # B[1, b_col1]
        
        # W indices: C[i, k] -> k*n + i
        # C[0, c_col0] -> k=c_col0, i=0 -> c_col0*n + 0
        w00 = c_col0 * n + 0
        w01 = c_col1 * n + 0
        # C[1, c_col0] -> k=c_col0, i=1 -> c_col0*n + 1
        w10 = c_col0 * n + 1
        w11 = c_col1 * n + 1
        
        # U indices: A[i, j] -> i*m + j
        u00 = 0 * m + 0  # A[0,0]
        u01 = 0 * m + 1  # A[0,1]
        u10 = 1 * m + 0  # A[1,0]
        u11 = 1 * m + 1  # A[1,1]
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        r = start_r
        U[u00, r] = 1; U[u11, r] = 1
        V[v00, r] = 1; V[v11, r] = 1
        W[w00, r] = 1; W[w11, r] = 1  # C[0,0] += m1, C[1,1] += m1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,0]
        r = start_r + 1
        U[u10, r] = 1; U[u11, r] = 1
        V[v00, r] = 1
        W[w10, r] = 1; W[w11, r] = -1  # C[1,0] += m2, C[1,1] -= m2
        
        # m3 = A[0,0] * (B[1,0]-B[1,1])
        r = start_r + 2
        U[u00, r] = 1
        V[v10, r] = 1; V[v11, r] = -1
        W[w01, r] = 1; W[w11, r] = 1  # C[0,1] += m3, C[1,1] += m3
        
        # m4 = A[1,1] * (B[0,1]-B[0,0])
        r = start_r + 3
        U[u11, r] = 1
        V[v01, r] = 1; V[v00, r] = -1
        W[w00, r] = 1; W[w10, r] = 1  # C[0,0] += m4, C[1,0] += m4
        
        # m5 = (A[0,0]+A[0,1]) * B[1,1]
        r = start_r + 4
        U[u00, r] = 1; U[u01, r] = 1
        V[v11, r] = 1
        W[w01, r] = 1; W[w00, r] = -1  # C[0,1] += m5, C[0,0] -= m5
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        r = start_r + 5
        U[u10, r] = 1; U[u00, r] = -1
        V[v00, r] = 1; V[v01, r] = 1
        W[w11, r] = 1  # C[1,1] += m6
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        r = start_r + 6
        U[u01, r] = 1; U[u11, r] = -1
        V[v10, r] = 1; V[v11, r] = 1
        W[w00, r] = 1  # C[0,0] += m7

    # Left half: B columns 0,1 -> C columns 0,1
    add_strassen_block(0, b_col0=0, b_col1=1, c_col0=0, c_col1=1)
    
    # Right half: B columns 2,3 -> C columns 2,3
    add_strassen_block(7, b_col0=2, b_col1=3, c_col0=2, c_col1=3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank13(deadline):
    """
    Try to find a rank-13 decomposition using alternating minimization.
    We start from the rank-14 Strassen solution and try to merge two factors
    by optimizing a rank-13 parameterization.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    
    # Get the rank-14 solution as starting point
    sol14 = _strassen_rank14()
    U14 = np.array(sol14['U'], dtype=float)
    V14 = np.array(sol14['V'], dtype=float)
    W14 = np.array(sol14['W'], dtype=float)
    
    # For rank 13, we need to find U (4x13), V (8x13), W (8x13)
    # such that sum_r U[:,r] * V[:,r] * W[:,r] = T (the multiplication tensor)
    
    # Strategy: Use alternating minimization with soft rank penalty
    # Start by randomly selecting 13 of the 14 Strassen factors and optimizing
    
    np.random.seed(42)
    
    # Try multiple random initializations
    best_solution = None
    best_error = float('inf')
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
            
        # Choose which factor to drop (try dropping each one)
        drop_idx = trial % 14
        
        # Initialize with 13 of the 14 Strassen factors
        keep = [i for i in range(14) if i != drop_idx]
        U = U14[:, keep].copy()
        V = V14[:, keep].copy()
        W = W14[:, keep].copy()
        
        # Add small random perturbation to escape local minima
        if trial > 0:
            U += np.random.randn(*U.shape) * 0.01
            V += np.random.randn(*V.shape) * 0.01
            W += np.random.randn(*W.shape) * 0.01
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline - 30:
                break
                
            # Update U: For each column r, solve for U[:,r] given V[:,r] and W[:,r]
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed r, U[:,r] appears linearly in each equation
            
            # Residual: T - sum_{r' != r} U[:,r'] * V[:,r'] * W[:,r']
            for r in range(target_rank):
                # Compute residual excluding factor r
                residual = np.ones((n, m, p))
                for r2 in range(target_rank):
                    if r2 == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i,j,k] -= U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                
                # Solve for U[:,r]: residual[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is a linear system in U[:,r]
                # For each (i,j), we have: residual[i,j,k] / (V[j*p+k, r] * W[k*n+i, r]) = U[i*m+j, r]
                # But this must hold for all k, so we need consistency
                
                # Use least squares approach
                # For each (i,j), the equation is: sum_k (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) * delta_k = residual[i,j,k]
                # This is underdetermined, so use the constraint that U[i*m+j,r] should be consistent
                
                # Actually, let's use a different approach: gradient descent
                # Or use the fact that for each (i,j), U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) should equal residual[i,j,k]
                # So U[i*m+j,r] = residual[i,j,k] / (V[j*p+k,r] * W[k*n+i,r]) for any k where denominator != 0
                
                for i in range(n):
                    for j in range(m):
                        # Find k where V[j*p+k,r] * W[k*n+i,r] is non-zero
                        found = False
                        for k in range(p):
                            denom = V[j*p+k, r] * W[k*n+i, r]
                            if abs(denom) > 1e-10:
                                U[i*m+j, r] = residual[i,j,k] / denom
                                found = True
                                break
                        if not found:
                            # Keep current value
                            pass
            
            # Update V similarly
            for r in range(target_rank):
                residual = np.ones((n, m, p))
                for r2 in range(target_rank):
                    if r2 == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i,j,k] -= U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                
                for j in range(m):
                    for k in range(p):
                        found = False
                        for i in range(n):
                            denom = U[i*m+j, r] * W[k*n+i, r]
                            if abs(denom) > 1e-10:
                                V[j*p+k, r] = residual[i,j,k] / denom
                                found = True
                                break
                        if not found:
                            pass
            
            # Update W similarly
            for r in range(target_rank):
                residual = np.ones((n, m, p))
                for r2 in range(target_rank):
                    if r2 == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i,j,k] -= U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                
                for k in range(p):
                    for i in range(n):
                        found = False
                        for j in range(m):
                            denom = U[i*m+j, r] * V[j*p+k, r]
                            if abs(denom) > 1e-10:
                                W[k*n+i, r] = residual[i,j,k] / denom
                                found = True
                                break
                        if not found:
                            pass
            
            # Check convergence
            T_recon = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            error = np.linalg.norm(T_recon - np.ones((n,m,p)))
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-9:
                break
        
        if best_error < 1e-9:
            break
    
    if best_solution is not None and best_error < 1e-9:
        U, V, W = best_solution
        # Round to nearest half-integer if possible
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded solution
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        # Try without rounding
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p = 16."""
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Construct the rank-14 Strassen solution
    best = _strassen_rank14()
    
    # Verify correctness
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if something went wrong
        return _schoolbook()
    
    # Try to find rank-13 solution
    if time.time() < deadline - 35:
        rank13 = _try_rank13(deadline)
        if rank13 is not None:
            if _verify(rank13["U"], rank13["V"], rank13["W"], n, m, p):
                return rank13
    
    return best
