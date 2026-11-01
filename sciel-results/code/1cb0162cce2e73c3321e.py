import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target: Rank 11
    target_rank = 11
    
    # Time guard
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check if U, V, W are integer or half-integer
        # Multiply by 2 to check for integers
        U2 = np.array(U) * 2
        V2 = np.array(V) * 2
        W2 = np.array(W) * 2
        
        if not np.allclose(U2, np.round(U2)): return False
        if not np.allclose(V2, np.round(V2)): return False
        if not np.allclose(W2, np.round(W2)): return False
        
        # Check tensor identity
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Must equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Attempt to find rank 11
    # Strategy: Random sparse V, W. Solve for U.
    # U is linear if V, W fixed.
    # Eq: sum_t U[i*m+j, t] * (V[j*p+k, t] * W[k*n+i, t]) = 1
    
    def try_find_rank_11():
        if time.time() > deadline:
            return None
            
        for _ in range(1000): # Try 1000 times
            if time.time() > deadline:
                return None
                
            # Generate random sparse V and W
            # Sizes: V (m*p, 11) = (6, 11), W (p*n, 11) = (6, 11)
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Fill with small integers
            for idx in range(m*p):
                for t in range(target_rank):
                    if random.random() < 0.3: # Sparsity
                        V[idx, t] = random.choice([-1, 1])
                        
            for idx in range(p*n):
                for t in range(target_rank):
                    if random.random() < 0.3:
                        W[idx, t] = random.choice([-1, 1])
            
            # Build linear system for U
            # Unknowns: U[i*m+j, t] for i in n, j in m, t in rank
            # Total unknowns: n*m*rank = 4*11 = 44
            # Equations: n*m*p = 12
            # A * x = b
            # x is vector of U values
            # For each (i,j,k), eq is:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            
            A = np.zeros((n*m*p, n*m*target_rank))
            b = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(target_rank):
                            # Coefficient for U[i*m+j, t]
                            coeff = V[j*p+k, t] * W[k*n+i, t]
                            # Variable index: (i*m+j) * target_rank + t
                            var_idx = (i*m+j) * target_rank + t
                            A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares
            # Since underdetermined, use pseudo-inverse or linalg.lstsq
            try:
                U_flat, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Reshape U
            U = U_flat.reshape((n*m, target_rank))
            
            # Check if U is close to integer/half-integer
            # Multiply by 2
            U2 = U * 2
            U_rounded = np.round(U2)
            
            if np.allclose(U2, U_rounded, atol=1e-6):
                # Convert back to float (half-integers)
                U_candidate = U_rounded / 2.0
                V_candidate = V
                W_candidate = W
                
                # Verify
                if is_valid(U_candidate, V_candidate, W_candidate, target_rank):
                    return {
                        "U": U_candidate.tolist(),
                        "V": V_candidate.tolist(),
                        "W": W_candidate.tolist()
                    }
            
            # Also try rounding U directly to integers?
            # The problem allows half-integers.
            
    result = try_find_rank_11()
    
    if result:
        return result
        
    # Fallback to Rank 12 (Schoolbook)
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
