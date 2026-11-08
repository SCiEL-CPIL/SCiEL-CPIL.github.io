"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s kill
    
    def make_schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_float(U_np, V_np, W_np, R, tol=1e-9):
        """Quick float check"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(R, max_iters=200, n_restarts=3, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row (i,j), U[i*m+j,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # Fix V,W. For fixed j, the equation for each i,k is:
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                # This is a linear system: U[i*m+j,:] @ M_{i,j} = 1 (vector of length p)
                # where M_{i,j}[k,r] = V[j*p+k,r] * W[k*n+i,r]
                
                for i in range(n):
                    for j in range(m):
                        # Build matrix M of shape (p, R): M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve U[i*m+j,:] such that M^T @ U[i*m+j,:] = 1 (length p)
                        # i.e., U[i*m+j,:] @ M = 1
                        # Use least squares: U[i*m+j,:] = argmin ||U[i*m+j,:] @ M - 1||^2
                        # This is (M^T M) U^T = M^T * 1
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row (j,k), V[j*p+k,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                # Fix U,W. For fixed j,k:
                # sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
                # V[j*p+k,:] @ N = 1 where N[i,r] = U[i*m+j,r] * W[k*n+i,r]
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row (k,i), W[k*n+i,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                # Fix U,V. For fixed k,i:
                # sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
                # W[k*n+i,:] @ P = 1 where P[j,r] = U[i*m+j,r] * V[j*p+k,r]
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if verify_float(U, V, W, R, tol=1e-8):
                    # Convert to exact rationals/ints
                    U_list = [[0]*R for _ in range(n*m)]
                    V_list = [[0]*R for _ in range(m*p)]
                    W_list = [[0]*R for _ in range(p*n)]
                    for idx in range(n*m):
                        for r in range(R):
                            U_list[idx][r] = int(round(U[idx, r] * 2)) / 2 if abs(U[idx,r]*2 - round(U[idx,r]*2)) < 1e-6 else U[idx, r]
                    for idx in range(m*p):
                        for r in range(R):
                            V_list[idx][r] = int(round(V[idx, r] * 2)) / 2 if abs(V[idx,r]*2 - round(V[idx,r]*2)) < 1e-6 else V[idx, r]
                    for idx in range(p*n):
                        for r in range(R):
                            W_list[idx][r] = int(round(W[idx, r] * 2)) / 2 if abs(W[idx,r]*2 - round(W[idx,r]*2)) < 1e-6 else W[idx, r]
                    
                    if verify_exact(U_list, V_list, W_list, R):
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try with different scaling
            U = U / (np.linalg.norm(U) + 1e-10)
            V = V / (np.linalg.norm(V) + 1e-10)
            W = W / (np.linalg.norm(W) + 1e-10)
        
        return None
    
    # Try ranks from low to high
    # Known lower bound: for m=4, rank >= m = 4 (trivial), but for matrix mult rank >= max(n*m, m*p, p*n)/something
    # Actually rank must be at least max(n*m, m*p, p*n) / min(n*m, m*p, p*n) * min... 
    # Simple bound: rank >= m = 4. Realistically for 2x4x5, try 10-20.
    for R in range(10, 25):
        if time.time() > deadline:
            break
        result = try_rank(R, max_iters=100, n_restarts=2, seed_base=R*100)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
