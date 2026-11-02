"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Schoolbook baseline
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)
    
    # Try to find better decompositions using structured search
    # Strategy: fix U and V to have nice patterns, then solve for W
    # Or use known small-block optimizations
    
    # For 3x5x6, let's try various structured approaches
    
    # Approach 1: Try to use Kronecker product structures
    # If we can decompose 5 = 2+3 or use other factorizations
    
    # Approach 2: Random search with verification
    # Generate random U, V, W with small integer/half-integer entries
    # and check if they work
    
    # Approach 3: Use the fact that for small matrices, we can try
    # to find decompositions by solving linear systems
    
    # Let's try a systematic search for ranks from a lower bound up
    
    # Lower bound: the rank of the matmul tensor is at least max(n, m, p)
    # but actually it's at least n*m + m*p + p*n - n - m - p + 1 or similar
    # For 3x5x6: at least 15 + 30 + 18 - 3 - 5 - 6 + 1 = 50 (very loose)
    
    # Actually, a better lower bound: the rank is at least the max of the
    # ranks of all matrix slices. For matmul, the (i,j)-slice in k is
    # a rank-1 matrix with all entries 1, so rank 1. Not helpful.
    
    # Let's just try to find any decomposition with rank < 90.
    
    # Try ranks from 50 to 89
    for r in range(50, min(90, best_rank)):
        if time.time() > deadline - 5:
            break
        
        result = _try_find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_rank = r
            best_result = result
            # Don't break - try to find even better
            # But if we find one, we can stop after a few more tries
            break
    
    return best_result

def _schoolbook(n, m, p, rank):
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

def _verify(n, m, p, U, V, W, r):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_find_decomposition(n, m, p, r, deadline):
    """Try to find a rank-r decomposition."""
    
    # Strategy: use structured patterns for U, V, W
    # For matmul, a natural structure is to use indicator-like patterns
    
    # Try random search with small integer/half-integer values
    for trial in range(200):
        if time.time() > deadline - 5:
            return None
        
        U, V, W = _generate_structured(n, m, p, r, trial)
        if _verify(n, m, p, U, V, W, r):
            return {"U": U, "V": V, "W": W}
    
    # Try another strategy: fix U and V, solve for W
    for trial in range(50):
        if time.time() > deadline - 5:
            return None
        
        U, V = _generate_UV(n, m, p, r, trial)
        W = _solve_for_W(n, m, p, r, U, V)
        if W is not None:
            # Round to half-integers
            W_rounded = [[round(2.0 * W[i][c]) / 2.0 for c in range(r)] for i in range(p * n)]
            if _verify(n, m, p, U, V, W_rounded, r):
                return {"U": U, "V": V, "W": W_rounded}
    
    return None

def _generate_structured(n, m, p, r, seed):
    """Generate a structured candidate decomposition."""
    random.seed(seed * 7919 + n * 1000 + m * 100 + p)
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Use a pattern based on modular arithmetic
    for c in range(r):
        # Create patterns based on c
        for i in range(n):
            for j in range(m):
                # U pattern
                if (i + j + c) % 3 == 0:
                    U[i*m+j][c] = 1
                elif (i + 2*j + c) % 3 == 0:
                    U[i*m+j][c] = -1
        
        for j in range(m):
            for k in range(p):
                # V pattern
                if (j + k + c) % 3 == 0:
                    V[j*p+k][c] = 1
                elif (2*j + k + c) % 3 == 0:
                    V[j*p+k][c] = -1
        
        for k in range(p):
            for i in range(n):
                # W pattern
                if (k + i + c) % 3 == 0:
                    W[k*n+i][c] = 1
                elif (k + 2*i + c) % 3 == 0:
                    W[k*n+i][c] = -1
    
    return U, V, W

def _generate_UV(n, m, p, r, seed):
    """Generate U and V matrices."""
    random.seed(seed * 104729 + 7)
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    
    for c in range(r):
        for i in range(n):
            for j in range(m):
                U[i*m+j][c] = random.choice([1, -1])
        
        for j in range(m):
            for k in range(p):
                V[j*p+k][c] = random.choice([1, -1])
    
    return U, V

def _solve_for_W(n, m, p, r, U, V):
    """Solve for W given U and V."""
    W = [[0] * r for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            # For fixed (i, k), we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            # This is a linear system in W[k*n+i, :]
            
            A = np.zeros((m, r))
            b = np.ones(m)
            
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            
            # Check if the system is consistent
            rank_A = np.linalg.matrix_rank(A)
            rank_AB = np.linalg.matrix_rank(np.hstack([A, b.reshape(-1, 1)]))
            
            if rank_A != rank_AB:
                return None
            
            # Find a solution
            try:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                for c in range(r):
                    W[k*n+i][c] = w[c]
            except:
                return None
    
    return W
# EVOLVE-BLOCK-END
