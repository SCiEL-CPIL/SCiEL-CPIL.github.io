"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        # Build the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_list(arr):
        """Convert numpy array to list of lists."""
        return [[x for x in row] for row in arr]
    
    # Try to find a better decomposition
    # Strategy: try ranks from 62 down, but that's too slow.
    # Instead, try a few specific ranks and use optimization.
    
    # For now, try a simple approach: see if we can find rank 62
    # This is unlikely to be easy, so let's try a few things.
    
    # Approach: use the fact that we can try to merge some terms
    # For the schoolbook, each term is (i,j,k). Can we combine some?
    
    # Let's try a numerical approach for a slightly reduced rank
    # Try rank = 62, 61, 60 with a simple optimization
    
    best_result = None
    best_rank = n * m * p
    
    # Try a few candidate ranks
    for target_rank in [62, 61, 60, 59, 58]:
        if time.time() > deadline - 20:
            break
        
        # Initialize with schoolbook and zero out some entries
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize with first target_rank schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
                    else:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        
        # Try alternating minimization
        # This is a 3-way tensor decomposition, so we can alternate
        # fixing two and solving for the third
        
        success = False
        for iteration in range(100):
            if time.time() > deadline - 10:
                break
            
            # Solve for U: for each (i,j), U[i*m+j, :] = (V[j*p+k,:] * W[k*n+i,:])^T * 1
            # Actually, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # For fixed V,W: U[i*m+j,c] appears in the equation for (i,j,k)
            # This is a linear system in U if V,W are fixed
            
            # Let's set up the linear system for U
            # For each (i,j), we have variables U[i*m+j, 0..target_rank-1]
            # The constraint is: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # This is p equations in target_rank unknowns (if target_rank > p, underdetermined)
            
            # For each (i,j):
            for i in range(n):
                for j in range(m):
                    # Build A matrix: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        for c in range(target_rank):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve A @ u = 1 for u = U[i*m+j, :]
                    try:
                        u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Similarly for V
            for j in range(m):
                for k in range(p):
                    # V[j*p+k, c] appears in equation for (i,j,k)
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # For fixed U,W: V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        for c in range(target_rank):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        v, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Similarly for W
            for k in range(p):
                for i in range(n):
                    # W[k*n+i, c] appears in equation for (i,j,k)
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # For fixed U,V: W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-6:
                success = True
                break
        
        if success:
            # Convert to exact rationals if possible
            U_list = to_list(U)
            V_list = to_list(V)
            W_list = to_list(W)
            
            # Verify exactly
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
            else:
                # Try to round to nearest rationals
                from fractions import Fraction
                def round_to_rational(x, max_denom=100):
                    for d in range(1, max_denom+1):
                        num = round(x * d)
                        if abs(x - num/d) < 1e-8:
                            return num / d
                    return x
                
                U_list = [[round_to_rational(x) for x in row] for row in U_list]
                V_list = [[round_to_rational(x) for x in row] for row in V_list]
                W_list = [[round_to_rational(x) for x in row] for row in W_list]
                
                if verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
