"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with schoolbook rank 36
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W correctly decompose the all-ones tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build a structured rank-15 decomposition using Strassen + direct sums
    # Strategy: Partition 3x3*3x4 into overlapping 2x2 blocks
    # Use Strassen for 2x2x2 (rank 7) and handle the rest with rank-1 terms
    
    def build_strassen_decomposition():
        """Build a rank-15 decomposition using algebraic structure."""
        # For 3x3 * 3x4, we can use:
        # - Strassen for the top-left 2x2 * 2x2 block (rank 7)
        # - Rank-1 terms for remaining elements
        # Total rank: 7 + 8 = 15 (approximately)
        
        # Actually, let's try a different approach:
        # Use the fact that 3x3x4 tensor can be decomposed as:
        # - One 2x2x2 tensor (rank 7 via Strassen)
        # - Remaining terms handled with rank-1 decompositions
        
        # Strassen's 7 multiplications for 2x2 matrices:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For our 3x3 * 3x4 case, let's partition:
        # A = [[A11, A12, A13],
        #      [A21, A22, A23],
        #      [A31, A32, A33]]
        # B = [[B11, B12, B13, B14],
        #      [B21, B22, B23, B24],
        #      [B31, B32, B33, B34]]
        
        # We'll use Strassen on the 2x2 block (rows 0,1 of A; cols 0,1 of B)
        # and handle the rest with additional terms
        
        rank = 15
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen terms for 2x2 block (i,j in {0,1}, k in {0,1})
        # M1 = (A00+A11)(B00+B11)
        U[0*3+0, 0] = 1; V[0*4+0, 0] = 1; W[0*3+0, 0] = 1
        U[1*3+1, 0] = 1; V[1*4+1, 0] = 1; W[1*3+1, 0] = 1
        U[0*3+1, 0] = 1; V[1*4+0, 0] = 1; W[1*3+0, 0] = 1
        U[1*3+0, 0] = 1; V[0*4+1, 0] = 1; W[0*3+1, 0] = 1
        
        # M2 = (A10+A11)B00
        U[1*3+0, 1] = 1; V[0*4+0, 1] = 1; W[0*3+1, 1] = 1
        U[1*3+1, 1] = 1; V[0*4+0, 1] = 1; W[0*3+1, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0*3+0, 2] = 1; V[1*4+0, 2] = 1; W[0*3+0, 2] = 1
        U[0*3+0, 2] = -1; V[1*4+1, 2] = 1; W[1*3+0, 2] = 1
        # Fix: need separate terms
        U[0*3+0, 2] = 1; V[1*4+0, 2] = 1; W[0*3+0, 2] = 1
        U[0*3+0, 3] = 1; V[1*4+1, 3] = -1; W[1*3+0, 3] = 1
        
        # M4 = A11(B01-B00)
        U[1*3+1, 4] = 1; V[0*4+1, 4] = 1; W[1*3+1, 4] = 1
        U[1*3+1, 5] = 1; V[0*4+0, 5] = -1; W[0*3+1, 5] = 1
        
        # M5 = (A00+A01)B11
        U[0*3+0, 6] = 1; V[1*4+1, 6] = 1; W[1*3+0, 6] = 1
        U[0*3+1, 6] = 1; V[1*4+1, 6] = 1; W[1*3+0, 6] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*3+0, 7] = 1; V[0*4+0, 7] = 1; W[0*3+1, 7] = 1
        U[1*3+0, 8] = 1; V[0*4+1, 8] = 1; W[1*3+1, 8] = 1
        U[0*3+0, 7] = -1; V[0*4+0, 7] = 1; W[0*3+1, 7] = 1
        U[0*3+0, 8] = -1; V[0*4+1, 8] = 1; W[1*3+1, 8] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*3+1, 9] = 1; V[1*4+0, 9] = 1; W[0*3+0, 9] = 1
        U[0*3+1, 10] = 1; V[1*4+1, 10] = 1; W[1*3+0, 10] = 1
        U[1*3+1, 9] = -1; V[1*4+0, 9] = 1; W[0*3+1, 9] = 1
        U[1*3+1, 10] = -1; V[1*4+1, 10] = 1; W[1*3+0, 10] = 1
        
        # This is getting complex. Let's use a simpler approach:
        # Just use numerical optimization but with better initialization
        
        return None
    
    # Try numerical optimization with structured initialization
    np.random.seed(42)
    
    # Try ranks from 15 upward
    for target_rank in range(15, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        max_trials = 80 if target_rank <= 20 else 40
        if time.time() > deadline - 15:
            max_trials = 10
            
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Use structured initialization: start with schoolbook and add noise
            if trial == 0:
                # Start from schoolbook and randomly project
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
            elif trial % 3 == 1:
                # Sparse initialization
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    i = random.randint(0, n*m-1)
                    j = random.randint(0, m*p-1)
                    k = random.randint(0, p*n-1)
                    U[i, r] = random.choice([-1, 1])
                    V[j, r] = random.choice([-1, 1])
                    W[k, r] = random.choice([-1, 1])
            else:
                U = np.random.randn(n*m, target_rank) * 0.3
                V = np.random.randn(m*p, target_rank) * 0.3
                W = np.random.randn(p*n, target_rank) * 0.3
            
            # Iterative refinement with vectorized operations
            for iteration in range(300):
                if time.time() > deadline - 3:
                    break
                    
                # Update U: for each (i,j), solve least squares
                for i in range(n):
                    for j in range(m):
                        # M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        
                        res = np.ones(p)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve least squares
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        
                        res = np.ones(n)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i), solve least squares
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        
                        res = np.ones(m)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check convergence every 10 iterations
                if iteration % 10 == 0:
                    if verify_decomposition(U, V, W, tol=1e-5):
                        found = True
                        break
            
            if found:
                # Polish with higher precision
                for polish_iter in range(150):
                    if time.time() > deadline - 2:
                        break
                    for i in range(n):
                        for j in range(m):
                            M = np.zeros((p, target_rank))
                            for k in range(p):
                                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            res = np.ones(p)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                                U[i*m+j, :] = sol
                            except:
                                pass
                    for j in range(m):
                        for k in range(p):
                            M = np.zeros((n, target_rank))
                            for i in range(n):
                                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            res = np.ones(n)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                                V[j*p+k, :] = sol
                            except:
                                pass
                    for k in range(p):
                        for i in range(n):
                            M = np.zeros((m, target_rank))
                            for j in range(m):
                                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            res = np.ones(m)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                                W[k*n+i, :] = sol
                            except:
                                pass
                    
                    if verify_decomposition(U, V, W, tol=1e-9):
                        break
                
                if verify_decomposition(U, V, W, tol=1e-7):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        
        if found:
            break
    
    # Convert to lists with integer/half-integer rounding if close
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Try to round to nearest 0.5
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    new_row.append(rounded)
                else:
                    # Try to round to nearest integer
                    rounded_int = round(val)
                    if abs(val - rounded_int) < 1e-6:
                        new_row.append(float(rounded_int))
                    else:
                        new_row.append(float(val))
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
