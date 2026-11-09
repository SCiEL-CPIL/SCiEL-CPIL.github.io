"""Matrix multiplication 2x3x5 - search for rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

def construct():
    """
    Return the matrix multiplication tensor for 2x3 * 3x5 as a rank decomposition.
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (10, rank)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k]
    
    Searches for a rank-minimal exact decomposition.
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def make_schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_rank(R, max_iters=5000):
        """Try to find a rank-R decomposition."""
        if time.time() > deadline:
            return None
        
        # Strategy: randomize U and V with small integer/half-integer values,
        # then solve for W linearly.
        
        # For fixed U, V, the equations for W are:
        # For each (i,k): sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for j=0,1,2
        # This is a linear system: A_{(i,k)} @ W_{(i,k)} = b
        # where A_{(i,k)}[j, r] = U[i*m+j, r] * V[j*p+k, r]
        
        # We need this system to be consistent for all (i,k) pairs.
        # There are n*p = 10 such systems, each with m=3 equations and R unknowns.
        
        # For the system to have a solution, we need rank(A) <= 3 and consistency.
        # If R >= 3 and the 3 rows are linearly independent, there's a unique solution
        # only if R=3. If R > 3, there are infinitely many solutions.
        
        # Let's try: initialize U and V randomly from {0, 1/2, 1, -1/2, -1}
        # Then for each (i,k), solve the 3xR system for W.
        
        values = [0, Fraction(1,2), 1, Fraction(-1,2), -1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Randomly initialize U and V
            U = [[random.choice(values) for _ in range(R)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(R)] for _ in range(m*p)]
            
            # Solve for W
            W = [[0] * R for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 3xR system
                    A = []
                    for j in range(m):
                        row = [U[i*m+j][r] * V[j*p+k][r] for r in range(R)]
                        A.append(row)
                    b = [Fraction(1), Fraction(1), Fraction(1)]
                    
                    # Solve A @ w = b where w = W[k*n+i, :]
                    # Use least squares or direct solve
                    # Convert to numpy for solving
                    A_np = np.array([[float(x) for x in row] for row in A])
                    b_np = np.array([float(x) for x in b])
                    
                    # Check if system is consistent
                    try:
                        # Use lstsq
                        w_sol, residuals, rank_A, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        
                        # Check residual
                        if np.max(np.abs(A_np @ w_sol - b_np)) > 1e-9:
                            valid = False
                            break
                        
                        # Try to round to nearest half-integer
                        w_rounded = [Fraction(round(x * 2)) / 2 for x in w_sol]
                        
                        # Verify the rounded solution
                        res = 0
                        for j in range(m):
                            for r in range(R):
                                res += U[i*m+j][r] * V[j*p+k][r] * w_rounded[r]
                            if res != 1:
                                valid = False
                                break
                        if not valid:
                            break
                        
                        W[k*n+i] = w_rounded
                    
                    except Exception:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Verify the full solution
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # First, get the schoolbook solution as fallback
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank, starting from 20 and going up
    for R in range(20, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(R, max_iters=3000)
        if result is not None:
            best = result
            best_rank = R
            break  # Found a solution at this rank, stop
    
    # If we didn't find anything better, return schoolbook
    return best
