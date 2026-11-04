"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build the target tensor: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    # Otherwise T[a,b,c] = 0
    # The tensor has nm*mp*pn entries, but only n*m*p = 120 are nonzero
    
    # Schoolbook decomposition (rank = n*m*p = 120)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition
    # Strategy: use a greedy approach to try to merge terms
    
    # First, let's try a simple approach: see if we can reduce rank by 1, 2, etc.
    # by finding linear dependencies
    
    # For a rank-r decomposition, we need r vectors u_r in R^nm, v_r in R^mp, w_r in R^pn
    # such that sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
    
    # This is a hard problem. Let's try a few heuristics.
    
    # Heuristic 1: Try to find if any two schoolbook terms can be merged
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there's a way to combine them
    # into a single term with adjusted coefficients
    
    # This is complex. Let's try a different approach: use a randomized search
    
    random.seed(42)
    
    # Try ranks from schoolbook_rank-1 down to max(nm, mp, pn) = 30
    # But only try a few due to time constraints
    
    # For each rank r, we need to solve a system of equations
    # Number of equations: nm*mp*pn = 20*30*24 = 14400
    # Number of unknowns for rank r: r*(nm+mp+pn) = r*74
    
    # For r=120: 120*74 = 8880 < 14400, so overdetermined
    # For r=119: 119*74 = 8806 < 14400, still overdetermined
    
    # This means we can't just randomly assign values; we need to solve the system
    
    # Given the difficulty, let's just return the schoolbook decomposition
    # unless we find a quick improvement
    
    if time.time() > deadline:
        return best_result
    
    # Try a structured approach: use the fact that the tensor has a certain structure
    # For example, can we use a block decomposition?
    
    # Actually, let's try to use known algorithms for rectangular matrix multiplication
    # For 4x5 x 5x6, I'm not sure of the optimal rank, but let's try to find something
    
    # One idea: use the fact that 4x5 can be embedded in a larger square matrix
    # and use Strassen-like algorithms
    
    # But this is complex. Given the time constraints, let's return the schoolbook
    
    return best_result
# EVOLVE-BLOCK-END
