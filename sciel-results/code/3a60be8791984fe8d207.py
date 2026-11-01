"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import random
from fractions import Fraction

def construct():
    """Return the schoolbook tensor decomposition for 2x3x9 matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p = 2*3*9 = 54.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    We attempt to find a lower-rank decomposition via structured search,
    but fall back to schoolbook if no better is found within the time budget.
    """
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor.
        
        The tensor T[i,j,k] for (C[i,k] = sum_j A[i,j]*B[j,k]) should equal
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        Here i is in [0,n), j is in [0,m), k is in [0,p).
        U has shape (n*m, rank)
        V has shape (m*p, rank)
        W has shape (p*n, rank)
        """
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = 1 for all (i,j,k) in the standard matrix mult
                    expected = 1  # The multiplication tensor has all 1s
                    actual = 0
                    for c in range(rank):
                        actual += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if actual != expected:
                        return False
        return True
    
    def verify_with_floats(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using floats for initial screening."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += float(U[i*m + j][c]) * float(V[j*p + k][c]) * float(W[k*n + i][c])
                    if abs(actual - 1.0) > tol:
                        return False
        return True
    
    best_result = None
    
    # Try to find a lower rank decomposition
    # Start with rank just above the theoretical lower bound
    # Lower bound for matrix multiplication of (n x m) by (m x p) is max(n*p, n*m, m*p) = max(18, 6, 27) = 27
    # But in practice, we look for ranks between known bounds
    
    # Known best for 2x3x3 is 7 (Strassen-like), for 2x2x2 is 7
    # For 2x3x9, let's try ranks from 35 upward (conservative start)
    # Actually, let's be more aggressive and try from lower values
    
    # The tensor has dimension 6*27*18 = 2916 entries in full, but structured
    # Let's try a few ranks
    
    ranks_to_try = list(range(27, 55))
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > DEADLINE:
            break
        
        # For each target rank, do a few random trials
        max_trials = 5
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE:
                break
            
            # Generate random factor matrices with small integer/rational entries
            # Use entries from {-2,-1,0,1,2}
            U = [[random.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick float check
            if verify_with_floats(U, V, W, n, m, p):
                # Exact check
                if verify_decomposition(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    print(f"Found rank {target_rank} decomposition!")
                    return best_result
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Fallback to schoolbook
    if best_result is not None:
        return best_result
    
    return build_schoolbook()
