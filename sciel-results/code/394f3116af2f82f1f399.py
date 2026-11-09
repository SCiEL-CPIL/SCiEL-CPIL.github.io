"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.

Strategy:
1. Start with the schoolbook decomposition (rank 60).
2. Attempt to reduce rank by identifying columns that can be merged or eliminated
   through linear dependence in the tensor structure.
3. Use a greedy approach: try to express one rank-1 term as a linear combination
   of others, and if successful, remove it.
4. Since exact integer/half-integer solutions are required, we work in Q (rationals).

For small dimensions like 2x5x6, the schoolbook rank is 60. Known results suggest
that the tensor rank of 2x5x6 matrix multiplication may be less than 60, but finding
the exact minimal rank is non-trivial. We'll attempt a heuristic reduction.

Key insight: The multiplication tensor has a structured sparsity. Each rank-1 term
corresponds to a triple (i,j,k). We can try to find linear relations among these
terms that allow us to replace multiple terms with fewer ones.

Approach:
- Represent the tensor as a 3D array T[a_idx, b_idx, c_idx] where T=1 for valid (i,j,k)
- For each pair of columns, check if their difference creates a new valid decomposition
- More practically: try to find a subset of columns that spans the same tensor space

Given the complexity, we'll implement a structured search:
1. Generate all 60 schoolbook terms
2. Try to find linear dependencies by solving for coefficients
3. If a term can be expressed as a linear combination of others with integer/half-integer
   coefficients, remove it

For this specific case, we'll use a more direct approach:
- The tensor is sparse with exactly 60 non-zero entries
- We'll try to find a basis of size < 60 that spans the same set of rank-1 tensors
"""

import numpy as np
from fractions import Fraction
import itertools
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication.
    Returns the best decomposition found.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    r_school = n * m * p
    
    # Initialize schoolbook decomposition
    U_school = np.zeros((n * m, r_school))
    V_school = np.zeros((m * p, r_school))
    W_school = np.zeros((p * n, r_school))
    
    # Store the terms as (a_idx, b_idx, c_idx) tuples
    terms = []
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U_school[a_idx, col] = 1.0
                V_school[b_idx, col] = 1.0
                W_school[c_idx, col] = 1.0
                terms.append((a_idx, b_idx, c_idx))
                col += 1
                
    terms_set = set(terms)
    
    # Helper to verify decomposition
    def verify_decomp(U, V, W):
        r = U.shape[1]
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = Fraction(0)
                    for t in range(r):
                        s += Fraction(U[a, t]) * Fraction(V[b, t]) * Fraction(W[c, t])
                    expected = Fraction(1) if (a, b, c) in terms_set else Fraction(0)
                    if s != expected:
                        return False
        return True

    # Helper to solve linear system A x = b using Fraction
    # A is list of lists of Fractions, b is list of Fractions
    def solve_linear_system(A, b):
        # A is M x N, b is M x 1
        M = len(A)
        N = len(A[0])
        # Augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Gaussian elimination
        rank = 0
        for col in range(N):
            # Find pivot
            pivot_row = None
            for row in range(rank, M):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                continue
                
            # Swap rows
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            
            # Normalize pivot row
            pivot_val = aug[rank][col]
            for j in range(col, N + 1):
                aug[rank][j] /= pivot_val
                
            # Eliminate
            for row in range(M):
                if row == rank:
                    continue
                factor = aug[row][col]
                if factor != 0:
                    for j in range(col, N + 1):
                        aug[row][j] -= factor * aug[rank][j]
                        
            rank += 1
            
        # Check for consistency
        for row in range(rank, M):
            if aug[row][N] != 0:
                return None # Inconsistent
                
        # Extract solution (set free variables to 0)
        x = [Fraction(0)] * N
        for i in range(rank):
            # Find the leading column
            lead_col = None
            for col in range(N):
                if aug[i][col] != 0:
                    lead_col = col
                    break
            if lead_col is not None:
                x[lead_col] = aug[i][N]
                
        return x

    # Randomized search for lower rank
    best_rank = r_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a decomposition of rank r < r_school
    # We will try ranks from 50 down to 42
    # For each rank, we will try several random initializations of U and V
    
    search_ranks = range(50, 41, -1)
    
    for r in search_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try a few random trials for this rank
        trials = 20
        found = False
        
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
                
            # Randomly initialize U and V with entries from {-1, 0, 1}
            U_rand = np.zeros((n * m, r))
            V_rand = np.zeros((m * p, r))
            
            for a in range(n * m):
                for t in range(r):
                    if random.random() < 0.5:
                        U_rand[a, t] = random.choice([-1, 1])
                        
            for b in range(m * p):
                for t in range(r):
                    if random.random() < 0.5:
                        V_rand[b, t] = random.choice([-1, 1])
                        
            # Solve for W
            W_sol = np.zeros((p * n, r))
            consistent = True
            
            for c in range(p * n):
                # Construct A_c and b_c
                # A_c has size (n*m) x r
                # A_c[a, t] = U_rand[a, t] * V_rand[b, t] ?
                # Wait, the equation is sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c]
                # This is linear in W[c,t] for fixed a,b,c?
                # No, for fixed c, we have equations for all (a,b).
                # The unknowns are W[c,0], ..., W[c,r-1].
                # The coefficient of W[c,t] in equation (a,b) is U[a,t] * V[b,t].
                # So A_c is a matrix of size (n*m) x r.
                # A_c[(a,b), t] = U_rand[a, t] * V_rand[b, t]
                
                A_c = []
                b_c = []
                for a in range(n * m):
                    for b in range(m * p):
                        row = []
                        for t in range(r):
                            row.append(Fraction(U_rand[a, t]) * Fraction(V_rand[b, t]))
                        A_c.append(row)
                        b_c.append(Fraction(1) if (a, b, c) in terms_set else Fraction(0))
                        
                # Solve A_c w_c = b_c
                w_c = solve_linear_system(A_c, b_c)
                if w_c is None:
                    consistent = False
                    break
                for t in range(r):
                    W_sol[c, t] = w_c[t]
                    
            if not consistent:
                continue
                
            # Verify the solution
            if verify_decomp(U_rand, V_rand, W_sol):
                best_rank = r
                best_U = U_rand
                best_V = V_rand
                best_W = W_sol
                found = True
                break # Break trials for this rank, move to next lower rank
                
        if found:
            # We found a decomposition of rank r. We can continue to try for lower ranks.
            pass
            
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
