import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Equation: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from low to high
    best_result = None
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    # Use alternating least squares / gradient approach
    
    for target_rank in range(24, 33):  # Try from 24 up to 32
        if time.time() > deadline - 5:
            break
        
        success = False
        for trial in range(50):
            if time.time() > deadline - 2:
                break
            
            rank = target_rank
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each (i,j), minimize sum over (k) of (target - V*W)^2 * U^2
                # Actually, the equation is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # For fixed V,W, and fixed (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is a linear system in U[i*m+j,:]
                
                # Let's do it properly: for each (i,j), we have p equations:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1, for k=0..p-1
                # Let M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then M @ U[i*m+j,:] = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k,:] = V[j*p+k,:] * W[k*n+i,:]
                        # Solve M @ x = ones(p)
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[i*m+j,:] = x
                        except:
                            pass
                
                # Similarly update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i,:] = U[i*m+j,:] * W[k*n+i,:]
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[j*p+k,:] = x
                        except:
                            pass
                
                # Similarly update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j,:] = U[i*m+j,:] * V[j*p+k,:]
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k*n+i,:] = x
                        except:
                            pass
                
                # Check residual
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] for r in range(rank))
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-8:
                    success = True
                    break
            
            if success:
                break
        
        if success:
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_rounded[i*m+j,r] * V_rounded[j*p+k,r] * W_rounded[k*n+i,r] for r in range(rank))
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-9:
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                break
            else:
                # Try integer rounding too
                U_int = np.round(U).astype(int)
                V_int = np.round(V).astype(int)
                W_int = np.round(W).astype(int)
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_int[i*m+j,r] * V_int[j*p+k,r] * W_int[k*n+i,r] for r in range(rank))
                            max_err = max(max_err, abs(val - 1))
                if max_err < 1e-9:
                    best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    break
    
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
