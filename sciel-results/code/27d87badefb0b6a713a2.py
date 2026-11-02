"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
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
    
    schoolbook = make_schoolbook()
    
    # Try the slice method: fix U to a known low-rank decomposition of the 3x3x3 core,
    # then solve linearly for V and W extensions
    
    # Strassen's rank-7 decomposition for 3x3x3 matrix multiplication
    # This gives us 7 rank-1 terms for the 3x3x3 core
    # We extend this to 3x3x5 by adding terms for the extra 2 columns of B
    
    # First, let's use the known rank-7 Strassen decomposition for 3x3x3
    # U_strassen: 9x7, V_strassen: 9x7, W_strassen: 9x7
    # For 3x3x5, we need U: 9xR, V: 15xR, W: 15xR
    
    # The tensor for 3x3x5 is: T[i*m+j, j*p+k, k*n+i] = 1
    # where i in [0,3), j in [0,3), k in [0,5)
    
    # Let's try to construct a decomposition using the slice method
    
    # Define the 3x3x3 Strassen factors (rank 7)
    # Using standard Strassen formulas for 3x3:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 3x3 matrices, Strassen's algorithm has rank 7 (or 8 in some formulations)
    # Let's use a concrete rank-7 decomposition
    
    # Actually, for simplicity and correctness, let's try a different approach:
    # Use the slice method with a fixed U that has integer/half-integer entries
    
    # The key insight: if we fix U, then for each column r of U, we need:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is bilinear in V and W for fixed U. We can linearize by fixing V and solving for W,
    # or using an alternating minimization approach.
    
    # Let's try a systematic approach:
    # 1. Fix U to have a specific structure (e.g., based on Strassen)
    # 2. Solve for V and W using linear algebra
    
    # For the 3x3x5 case, let's try rank values from 20 to 44
    
    # Use a simpler approach: try to find a decomposition by solving linear systems
    
    # The tensor T has shape (9, 15, 15)
    # We need U (9xR), V (15xR), W (15xR) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if valid, 0 otherwise
    
    # Let's try to use the fact that the tensor is sparse and structured
    
    # Approach: Use the slice method by fixing U to be the first R columns of the schoolbook
    # decomposition, but with some structure that allows for lower rank
    
    # Actually, let's try a more direct approach:
    # For each candidate rank R, try to find U, V, W by solving a system of equations
    
    # The number of constraints is 9*15*15 = 2025, but many are zero
    # Non-zero constraints: 3*3*5 = 45 (the valid multiplication terms)
    # Zero constraints: 2025 - 45 = 1980
    
    # Total unknowns: R*(9+15+15) = R*39
    
    # For R=45: 45*39 = 1755 unknowns, 2025 constraints (overdetermined but consistent)
    # For lower R: need to find a solution
    
    # Let's try a randomized approach with verification
    
    np.random.seed(42)
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 25 to 44
    for target_rank in range(25, 45):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        max_attempts = 20 if target_rank > 35 else 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            R = target_rank
            # Random initialization with small values
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Verify the decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if abs(s - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check some zero entries to ensure correctness
                zero_valid = True
                for _ in range(100):  # Spot check
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    c = np.random.randint(0, p*n)
                    # Check if this is a valid (i,j,k) triple
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue  # This is a valid entry, skip
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > 1e-6:
                        zero_valid = False
                        break
                
                if zero_valid:
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook
# EVOLVE-BLOCK-END
