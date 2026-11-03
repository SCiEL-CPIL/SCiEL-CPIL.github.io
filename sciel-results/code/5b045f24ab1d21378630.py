import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 63
    best_result = None
    
    # Start with schoolbook
    U_school = [[0]*63 for _ in range(n*m)]
    V_school = [[0]*63 for _ in range(m*p)]
    W_school = [[0]*63 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try to find lower rank
    # We will iterate r from 62 down
    for r in range(62, 30, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
                
            # Random U, V with small integers
            # To make it easier, maybe use sparse matrices?
            # Let's try dense random integers in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(n*m, r))
            V_rand = np.random.randint(-1, 2, size=(m*p, r))
            
            # Build linear system for W
            # Variables: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
            # Total variables: p*n*r
            # Equations: for each i, j, k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
            
            # We can rearrange:
            # For each (i, k), we have m equations (one for each j).
            # sum_c (U[i,j,c] V[j,k,c]) W[k,i,c] = 1
            # This is a linear system in W[k,i,c].
            
            # Let's solve for each (i, k) pair?
            # No, W[k,i,c] is shared across j?
            # Wait, W[k,i,c] depends on k and i, but not j.
            # So for a fixed (i, k), the variable is W[k,i,c].
            # The equations are for j=0..m-1.
            # sum_c A_{j,c} W[k,i,c] = 1, where A_{j,c} = U[i,j,c] V[j,k,c].
            # This is a system of m equations with r variables W[k,i,c].
            # If r >= m, we can solve it.
            
            W_sol = np.zeros((p*n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build matrix A of size m x r
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                    
                    # Solve A x = 1
                    # x is W[k*n+i, :]
                    try:
                        # Use least squares if underdetermined
                        x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check residual
                        if np.linalg.norm(A @ x - np.ones(m)) > 1e-6:
                            valid = False
                            break
                        W_sol[k*n+i, :] = x
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # We have a candidate W_sol.
            # Check if it leads to exact decomposition with rational entries.
            # The entries of W_sol are likely irrational or floating point.
            # We need to check if there exists a rational solution.
            # Since the system is underdetermined, we can try to find a rational solution.
            # However, if the random U, V were integers, the solution W might be rational.
            # Let's check if W_sol is close to half-integers.
            
            # Round to nearest half-integer
            W_rounded = np.round(W_sol * 2) / 2
            
            # Verify exact decomposition with rounded values
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # We use U_rand, V_rand (integers) and W_rounded (half-integers)
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rounded[k*n+i, c]
                        if abs(s - 1) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
            
            if is_exact:
                # Found a solution!
                # Convert to lists of integers/half-integers
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                # Ensure types are correct (floats for half-integers are ok if exact?)
                # The prompt says "integer/half-integer entries".
                # Returning floats that are exactly 0.5, 1.0 etc is usually fine.
                return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
