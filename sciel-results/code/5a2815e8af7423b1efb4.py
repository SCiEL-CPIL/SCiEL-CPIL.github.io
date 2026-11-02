import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 110.0

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_sb = [[0]*schoolbook_rank for _ in range(n*m)]
    V_sb = [[0]*schoolbook_rank for _ in range(m*p)]
    W_sb = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank solutions
    # We search for ranks from schoolbook_rank - 1 down to 30
    for rank in range(schoolbook_rank - 1, 29, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try a few random instances for this rank
        found = False
        for _ in range(100): # 100 random trials
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random integer U and V
            # Use small integers to keep numbers manageable
            U_rand = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V_rand = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            
            # Build linear system A x = b for W
            # Variables x are w_{k,i,c}
            # Index mapping: var_idx = (k*n + i)*rank + c
            num_vars = p * n * rank
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            coeff = U_rand[i*m+j][c] * V_rand[j*p+k][c]
                            if coeff != 0:
                                var_idx = (k*n + i)*rank + c
                                A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve A x = b
            # Check if b is in col(A)
            # We can use numpy.linalg.lstsq and check residual
            try:
                x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                # Check residual
                if len(residuals) > 0 and residuals[0] < 1e-9:
                    # Solution exists
                    # Convert x to W
                    W_rand = [[0]*rank for _ in range(p*n)]
                    for k in range(p):
                        for i in range(n):
                            for c in range(rank):
                                val = x[(k*n + i)*rank + c]
                                # Round to nearest fraction?
                                # The solution x is real. We need rational.
                                # If the system has a rational solution, numpy should find a close one.
                                # Let's try to find a rational approximation.
                                # For now, just check if integer rounding works?
                                # Better: use Fraction if possible, but numpy returns float.
                                # Let's assume the solution is close to a rational with small denominator.
                                # For exactness, we should verify.
                                W_rand[k*n+i][c] = int(round(val))
                    
                    # Verify the rounded solution
                    if _verify(U_rand, V_rand, W_rand):
                        best_rank = rank
                        best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                        found = True
                        break
            except:
                pass
        
        if found:
            # Continue to lower ranks
            continue
    
    return best_result
