import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _trivial_decomposition(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}, rank

def _check_decomposition(U, V, W, n, m, p, rank):
    """
    Check if the decomposition is valid.
    The tensor T[i,j,k] for matrix multiplication satisfies:
    C[i,k] = sum_j A[i,j] B[j,k]
    The tensor form is T[i,j,k] = 1 for the coefficient of A[i,j]B[j,k] in C[i,k].
    
    In the trace convention W[k*n+i], the decomposition must satisfy:
    sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] = 1 for all i,j,k.
    
    Wait, the standard tensor decomposition for matrix multiplication is:
    The tensor T has indices (i, j, k) where:
    i is row of A (and C)
    j is col of A (and row of B)
    k is col of B (and C)
    
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    
    The decomposition is:
    T[i,j,k] = sum_{t=1}^r U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    Note the index mapping in the problem statement:
    U shape: (n*m, rank) -> U[i*m+j, t]
    V shape: (m*p, rank) -> V[j*p+k, t]
    W shape: (p*n, rank) -> W[k*n+i, t]
    
    So we need: sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] == 1 for all i,j,k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
                    return False
    return True

def _try_winograd_2x3x7(n, m, p):
    """
    Attempt to find a lower rank decomposition using Winograd-like schemes.
    For 2x3 matrix multiplication, the rank is 7 (vs trivial 6).
    For 3x3, rank is 23 (vs trivial 27).
    
    Here we have 2x3 * 3x7.
    We can view this as:
    A is 2x3, B is 3x7.
    
    One approach: Split the 3 intermediate columns into groups.
    Since m=3 is small, we can try to exploit the 2x3 part.
    
    Let's try to construct a decomposition based on the known 2x3 rank-7 scheme
    and see if it can be extended or if a direct search works.
    
    Actually, let's just do a randomized search for rank < 42 with rational coefficients.
    """
    pass

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition
    best_result, best_rank = _trivial_decomposition(n, m, p)
    
    # If we have time, try to find a better decomposition
    # We will use a randomized search with rational numbers (Fractions)
    # to ensure exactness.
    
    # Define a set of candidate ranks to try
    # The rank must be at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But usually higher. Let's try a range.
    
    # Strategy: Randomly generate U, V, W with small integer coefficients,
    # then check if the tensor identity holds.
    # To increase chances, we can fix some structure.
    
    # For 2x3x7, the trivial rank is 42.
    # Let's try ranks from 41 down to 35.
    
    # Use a small pool of random coefficients
    coeff_pool = [-2, -1, 0, 1, 2]
    
    # To make the search more effective, we can try to satisfy the equations
    # for a subset of (i,j,k) first, then verify all.
    
    # However, a purely random search in high dimensions is unlikely to succeed
    # without a strong heuristic or structure.
    
    # Let's try a specific structured approach:
    # The 2x3 multiplication has a rank-7 decomposition.
    # If we can embed that into the 2x3x7 problem...
    
    # Actually, let's just do a limited random search for a few ranks.
    # If we don't find anything, return the trivial one.
    
    for target_rank in range(best_rank - 1, 34, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random attempts for this rank
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Generate random matrices
            U = [[random.choice(coeff_pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick check: For each (i,j,k), sum U[i,j,t]*V[j,k,t]*W[k,i,t] must be 1
            # We can check a random subset of (i,j,k) first to prune quickly
            indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(indices)
            
            # Check first 10 random indices
            valid = True
            for idx in indices[:10]:
                i, j, k = idx
                s = 0
                for t in range(target_rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
                    valid = False
                    break
            
            if valid:
                # Full check
                if _check_decomposition(U, V, W, n, m, p, target_rank):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break # Found a better one, stop searching for lower ranks?
                    # Or continue to see if we can go even lower?
                    # Let's continue to lower ranks.
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
