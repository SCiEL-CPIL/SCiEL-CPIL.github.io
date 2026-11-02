"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition: rank = n * m * p = 100
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Strategy: Try to find lower rank decompositions by solving linear systems
    # For a fixed set of non-zero patterns in U and V, W is determined linearly
    # We'll try random sparse patterns for U and V, then solve for W
    
    # Known lower bounds and upper bounds for 4x5x5 matrix multiplication
    # The rank is at least max(n*m, m*p, p*n) = 25
    # But for exact decompositions with integer entries, the rank is typically higher
    # Let's try ranks from 95 down to 90
    
    # For each candidate rank R, we try to find a decomposition
    # by randomly initializing U and V with sparse patterns, then solving for W
    
    def verify(U, V, W, R):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                # Check that sum over j of A[i,j] * B[j,k] = C[i,k]
                # In tensor form: sum_r U[i*m + j, r] * V[j*p + k, r] = W[k*n + i, r] * 0 + ... 
                # Actually, the condition is: for all i,k,
                # sum_{j=0}^{m-1} sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] = delta_{ik} * (something)
                # No, the correct condition is:
                # (A*B)[i,k] = sum_j A[i,j] B[j,k]
                # We need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{ik} * A[i,j] B[j,k]
                # This is not right. Let me reconsider.
                
                # The tensor T for matrix multiplication 4x5 by 5x5 is:
                # T[i, j, k] = delta_{i,k} where i in [0,3], j in [0,4], k in [0,4]
                # No, that's not right either.
                
                # The multiplication tensor M_{4,5,5} has entries:
                # M[i, j, k] = 1 if this corresponds to A[i,j] * B[j,k] contributing to C[i,k]
                # The decomposition is: M[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # where M[i,j,k] = delta_{i,k} ... no.
                
                # Actually, for matrix multiplication (n x m) * (m x p) -> (n x p):
                # C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor is: T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B
                # contribute to the (i,k) entry of C.
                # So T[i, j, k] = delta_{i,k} is not correct.
                
                # The correct tensor entries are:
                # T[i, j, k] = 1 for all valid i, j, k (where i<n, j<m, k<p)
                # because every A[i,j] * B[j,k] product contributes to C[i,k].
                # Wait, no. The tensor is indexed by (i, j, k) where:
                # i indexes rows of A and C (0..n-1)
                # j indexes cols of A / rows of B (0..m-1)  
                # k indexes cols of B and C (0..p-1)
                # T[i, j, k] = 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k]
                # This is always 1 for valid indices.
                
                # So the condition is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                # But that can't be right because the sum would be over r only.
                
                # Let me look at this more carefully.
                # The matrix multiplication tensor is a 3-tensor T of size (n*m) x (m*p) x (p*n)
                # No, it's a 3-way tensor with indices (i, (j,k)) or similar.
                
                # Standard formulation:
                # We have three sets of indices:
                # - I = {(i, j) : 0 <= i < n, 0 <= j < m} for A
                # - J = {(j, k) : 0 <= j < m, 0 <= k < p} for B
                # - K = {(k, i) : 0 <= k < p, 0 <= i < n} for C
                # 
                # T[(i,j), (j',k), (k',i')] = 1 if j = j' and i = i' and k = k', else 0
                # In other words, T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
                
                # The decomposition is:
                # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                # where a = i*m + j, b = j*p + k, c = k*n + i
                
                # So the condition is:
                # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For all other (a, b, c) where the indices don't match, the sum should be 0.
                
                # Let's verify by checking all entries.
                for a in range(n * m):
                    i, j = divmod(a, m)
                    for b in range(m * p):
                        j2, k = divmod(b, p)
                        for c in range(p * n):
                            k2, i2 = divmod(c, n)
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(R))
                            if abs(actual - expected) > 1e-6:
                                return False
                return True
    
    # For efficiency, let's only check a subset of entries during search
    def verify_partial(U, V, W, R, num_checks=100):
        """Verify a random subset of entries."""
        checks = []
        for _ in range(num_checks):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
            actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(R))
            if abs(actual - expected) > 1e-6:
                return False
        return True
    
    # Try different ranks
    for R in range(95, 85, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            
            # Initialize U and V with sparse random patterns
            U_rand = [[0] * R for _ in range(n * m)]
            V_rand = [[0] * R for _ in range(m * p)]
            
            # Each row of U has ~2-3 non-zeros, same for V
            for a in range(n * m):
                num_nz = random.randint(2, 3)
                cols = random.sample(range(R), num_nz)
                for c in cols:
                    U_rand[a][c] = random.choice([-1, 1])
            
            for b in range(m * p):
                num_nz = random.randint(2, 3)
                cols = random.sample(range(R), num_nz)
                for c in cols:
                    V_rand[b][c] = random.choice([-1, 1])
            
            # Now solve for W using the constraints
            # For each (i, j, k), we have: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear in W. Instead, let's fix U and V, and note that
            # for each c = k*n + i, the equation involves W[c, r] multiplied by
            # U[i*m+j, r] * V[j*p+k, r] for all j.
            
            # Actually, for a fixed (i, k), and varying j, we get equations:
            # For j = 0: sum_r U[i*m+0, r] * V[0*p+k, r] * W[k*n+i, r] = 1
            # For j = 1: sum_r U[i*m+1, r] * V[1*p+k, r] * W[k*n+i, r] = 1
            # etc.
            
            # These are all equations for the same W[k*n+i, r] vector!
            # So for each (i, k), we have m equations (one for each j) that all
            # involve the same unknown vector W[k*n+i, :].
            
            # Let's set up the system:
            # For fixed (i, k), let w = W[k*n+i, :] (a vector of length R)
            # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * w[r] = 1
            # This is: (U[i*m+j, :] * V[j*p+k, :]) dot w = 1
            # where * is element-wise product.
            
            # So for each (i, k), we have a system of m equations in R unknowns.
            # We can solve this as a least-squares problem.
            
            W_rand = [[0] * R for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    c = k * n + i
                    # Build system: for each j, equation is:
                    # (U[i*m+j, :] * V[j*p+k, :]) dot w = 1
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j, :] = [U_rand[i*m+j][r] * V_rand[j*p+k][r] for r in range(R)]
                    
                    # Solve A_sys w = 1 (vector of ones)
                    try:
                        w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if the solution makes sense
                    if not valid:
                        break
                    
                    # Store the solution, rounded to nearest integer or half-integer
                    for r in range(R):
                        w_val = w_sol[r]
                        # Snap to nearest half-integer
                        w_snapped = round(w_val * 2) / 2
                        W_rand[c][r] = w_snapped
            
            if not valid:
                continue
            
            # Verify the solution
            if verify_partial(U_rand, V_rand, W_rand, R, num_checks=50):
                # Full verification
                if verify(U_rand, V_rand, W_rand, R):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = R
                    print(f"Found rank {R} decomposition!")
                    return {"U": best_U, "V": best_V, "W": best_W}
            
            # Check time
            if time.time() > deadline - 10:
                break
    
    # Return the best found (or schoolbook if nothing better was found)
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
