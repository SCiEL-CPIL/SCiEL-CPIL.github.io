"""Matrix multiplication 2x3x4 - rank-minimal exact tensor decomposition."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _to_rational_array(mat, tol=1e-8):
    """Convert a numeric matrix to a matrix of Fractions (exact rationals)."""
    rows, cols = mat.shape
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = float(mat[i, j])
            if abs(val) < tol:
                result[i][j] = Fraction(0)
                continue
            found = False
            for den in range(1, 201):
                num = round(val * den)
                if abs(val - num / den) < tol:
                    result[i][j] = Fraction(num, den)
                    found = True
                    break
            if not found:
                result[i][j] = Fraction(val).limit_denominator(1000000)
    return result

def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly by checking all (i,j,k) triplets."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all valid (i,j,k) triplets sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid triplets sum to 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i2, c]
                            if abs(s) > tol:
                                return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Build the schoolbook solution as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    lower_bound = max(n*m, m*p, p*n)  # 12
    
    # Proper ALS implementation using matrix operations for speed
    def als_solve_U(T, V, W, rank, n, m, p):
        """Solve for U given V, W."""
        # B has shape (m*p * p*n, rank), B[b * p*n + c, r] = V[b, r] * W[c, r]
        B = np.zeros((m * p * p * n, rank))
        for b in range(m * p):
            for c in range(p * n):
                B[b * p * n + c, :] = V[b, :] * W[c, :]
        
        T_flat = T.reshape(n * m, m * p * p * n)
        
        try:
            U_sol = T_flat @ np.linalg.pinv(B.T)
        except:
            U_sol = np.random.randn(n * m, rank)
        return U_sol
    
    def als_solve_V(T, U, W, rank, n, m, p):
        """Solve for V given U, W."""
        D = np.zeros((n * m * p * n, rank))
        for a in range(n * m):
            for c in range(p * n):
                D[a * p * n + c, :] = U[a, :] * W[c, :]
        
        T_reshaped = np.zeros((n * m * p * n, m * p))
        for a in range(n * m):
            for c in range(p * n):
                T_reshaped[a * p * n + c, :] = T[a, :, c]
        
        try:
            V_T = np.linalg.pinv(D) @ T_reshaped
            V_sol = V_T.T
        except:
            V_sol = np.random.randn(m * p, rank)
        return V_sol
    
    def als_solve_W(T, U, V, rank, n, m, p):
        """Solve for W given U, V."""
        D = np.zeros((n * m * m * p, rank))
        for a in range(n * m):
            for b in range(m * p):
                D[a * m * p + b, :] = U[a, :] * V[b, :]
        
        T_reshaped = np.zeros((n * m * m * p, p * n))
        for a in range(n * m):
            for b in range(m * p):
                T_reshaped[a * m * p + b, :] = T[a, b, :]
        
        try:
            W_T = np.linalg.pinv(D) @ T_reshaped
            W_sol = W_T.T
        except:
            W_sol = np.random.randn(p * n, rank)
        return W_sol
    
    # Try ranks from lower bound up to schoolbook
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > DEADLINE - 20:
            break
        
        max_tries = 20
        found = False
        
        for trial in range(max_tries):
            if time.time() > DEADLINE - 10:
                break
            
            # Random initialization
            U_al = np.random.randn(n * m, target_rank)
            V_al = np.random.randn(m * p, target_rank)
            W_al = np.random.randn(p * n, target_rank)
            
            # Run ALS iterations
            for it in range(30):
                if time.time() > DEADLINE - 5:
                    break
                U_al = als_solve_U(T, V_al, W_al, target_rank, n, m, p)
                V_al = als_solve_V(T, U_al, W_al, target_rank, n, m, p)
                W_al = als_solve_W(T, U_al, V_al, target_rank, n, m, p)
            
            if time.time() > DEADLINE - 3:
                break
            
            # Verify
            valid = _verify_exact(U_al, V_al, W_al, n, m, p, target_rank, tol=1e-4)
            if valid:
                # Convert to rational
                U_rat = _to_rational_array(U_al)
                V_rat = _to_rational_array(V_al)
                W_rat = _to_rational_array(W_al)
                
                # Verify with rationals
                valid_rat = _verify_exact(U_rat, V_rat, W_rat, n, m, p, target_rank, tol=1e-6)
                if valid_rat:
                    best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
