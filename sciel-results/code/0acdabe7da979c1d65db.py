import numpy as np
import time
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank
    # Lower bound is n*p = 14
    for r in range(n * p, rank_sb):
        if time.time() > deadline:
            break
            
        # Try multiple structured approaches for this rank
        found = False
        
        # Approach 1: Randomized with exact rational solving
        for restart in range(50):
            if time.time() > deadline:
                    break
                    
            random.seed(42 + restart * 7)
            
            # Initialize U and V with small integer values
            U = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(n * m)]
            V = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(m * p)]
            
            # For each (i,k), compute M[i,k,t] = sum_j U[i*m+j, t] * V[j*p+k, t]
            # Then solve: sum_t M[i,k,t] * W[k*n+i, t] = 1
            W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
            
            valid = True
            for i in range(n):
                for k in range(p):
                    # Compute M vector
                    M = [Fraction(0) for _ in range(r)]
                    for j in range(m):
                        for t in range(r):
                            M[t] += U[i * m + j][t] * V[j * p + k][t]
                    
                    # Check if M is not zero
                    if all(x == 0 for x in M):
                        valid = False
                        break
                    
                    # Solve M · w = 1 where w = W[k*n+i, :]
                    # Use: w[t] = M[t] / (sum_t M[t]^2)
                    sum_sq = sum(x * x for x in M)
                    if sum_sq == 0:
                        valid = False
                        break
                    
                    for t in range(r):
                        W[k * n + i][t] = M[t] / sum_sq
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition exactly
            if _verify_decomposition(n, m, p, U, V, W):
                # Check if all entries are integer or half-integer
                if _all_int_or_half(U) and _all_int_or_half(V) and _all_int_or_half(W):
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            break
    
    # If no lower rank found with simple approach, try structured patterns
    if best_rank == rank_sb:
        # Try specific structured patterns
        for r in range(n * p, min(rank_sb, 25)):
            if time.time() > deadline:
                break
            found = _try_structured(n, m, p, r)
            if found is not None:
                best_rank = r
                best_U = found["U"]
                best_V = found["V"]
                best_W = found["W"]
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact using Fraction arithmetic."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for t in range(r):
                    total += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if total != 1:
                    return False
    return True


def _all_int_or_half(mat):
    """Check if all entries are integer or half-integer."""
    for row in mat:
        for x in row:
            if x.denominator not in (1, 2):
                return False
    return True


def _try_structured(n, m, p, r):
    """Try structured patterns for rank-r decomposition."""
    # Pattern: use Kronecker-like structure
    # For rank r, try to exploit the fact that T[i,j,k]=1 is separable
    # T[i,j,k] = a[i,j] * b[j,k] * c[k,i] doesn't work directly
    
    # Try: U[i*m+j, t] = u1[i,j,t], V[j*p+k, t] = v1[j,k,t], W[k*n+i, t] = w1[k,i,t]
    # with specific small patterns
    
    # Simple structured attempt: block diagonal
    if r >= n * p:
        # This is the schoolbook case, already handled
        pass
    
    # Try a few specific patterns
    patterns = []
    
    # Pattern 1: U has 1s in specific positions
    for offset in range(min(5, r)):
        U = [[Fraction(0) for _ in range(r)] for _ in range(n * m)]
        V = [[Fraction(0) for _ in range(r)] for _ in range(m * p)]
        W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
        
        # Fill with a pattern
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = (i * p + k + j) % r
                    U[i * m + j][t] = Fraction(1)
                    V[j * p + k][t] = Fraction(1)
                    W[k * n + i][t] = Fraction(1)
        
        if _verify_decomposition(n, m, p, U, V, W):
            if _all_int_or_half(U) and _all_int_or_half(V) and _all_int_or_half(W):
                return {"U": U, "V": V, "W": W}
    
    return None
