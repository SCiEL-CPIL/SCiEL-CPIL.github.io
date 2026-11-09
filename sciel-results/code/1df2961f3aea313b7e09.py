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
    deadline = start_time + 110  # Safety margin before 120s kill
    
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
    
    # Helper to verify decomposition exactly using Fractions
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

    # Helper to check if a numpy array contains only integers or half-integers
    def is_half_integer(arr):
        # Multiply by 2 and check if integer
        arr2 = arr * 2
        return np.allclose(arr2, np.round(arr2), atol=1e-9)

    # Helper to round to nearest half-integer
    def round_to_half_integer(arr):
        return np.round(arr * 2) / 2.0

    # Randomized search for lower rank
    best_rank = r_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Search ranks from 59 down to 45
    # We use numpy for fast solving and then exact verification
    search_ranks = range(59, 44, -1)
    
    for r in search_ranks:
        if time.time() > deadline - 5:
            break
            
        # Try random trials for this rank
        trials = 100
        found = False
        
        for trial in range(trials):
            if time.time() > deadline - 2:
                break
                
            # Randomly initialize U and V with entries from {-2, -1, 0, 1, 2}
            # Using sparsity to keep it structured but random
            U_rand = np.zeros((n * m, r))
            V_rand = np.zeros((m * p, r))
            
            # Density ~ 0.5
            for a in range(n * m):
                for t in range(r):
                    if random.random() < 0.5:
                        U_rand[a, t] = random.choice([-2, -1, 1, 2])
                        
            for b in range(m * p):
                for t in range(r):
                    if random.random() < 0.5:
                        V_rand[b, t] = random.choice([-2, -1, 1, 2])
                        
            # Solve for W using numpy
            W_sol = np.zeros((p * n, r))
            consistent = True
            
            for c in range(p * n):
                # Construct A_c and b_c
                # A_c is (n*m) x r
                # A_c[(a,b), t] = U_rand[a, t] * V_rand[b, t]
                
                A_c = np.zeros((n * m * m * p, r)) # n*m rows for a, m*p for b? No, a is 0..n*m-1, b is 0..m*p-1
                # Total equations: (n*m) * (m*p) ? No, the tensor T[a,b,c] is defined for all a,b.
                # So for a fixed c, we have equations for all a in 0..n*m-1 and b in 0..m*p-1.
                # Total rows = n*m * m*p = 10 * 30 = 300.
                # Columns = r.
                
                # Reshape U_rand and V_rand to form the Kronecker product structure
                # A_c[(a,b), t] = U_rand[a, t] * V_rand[b, t]
                # This is the Hadamard product of U_rand and V_rand, but stacked.
                # Let's build it efficiently.
                
                # U_rand is (n*m, r)
                # V_rand is (m*p, r)
                # We want a matrix of size (n*m * m*p, r)
                # Row index is a * (m*p) + b
                # Value is U_rand[a, t] * V_rand[b, t]
                
                # We can use broadcasting
                # U_rand[:, np.newaxis, :] has shape (n*m, 1, r)
                # V_rand[np.newaxis, :, :] has shape (1, m*p, r)
                # Product has shape (n*m, m*p, r)
                # Reshape to (n*m * m*p, r)
                
                A_c = (U_rand[:, np.newaxis, :] * V_rand[np.newaxis, :, :]).reshape(n * m * m * p, r)
                
                # b_c is the vector of T[a,b,c] for all a,b
                b_c = np.zeros(n * m * m * p)
                for a in range(n * m):
                    for b in range(m * p):
                        if (a, b, c) in terms_set:
                            b_c[a * m * p + b] = 1.0
                            
                # Solve A_c w_c = b_c
                # Use least squares
                try:
                    w_c, residuals, rank, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                except np.linalg.LinAlgError:
                    consistent = False
                    break
                    
                # Check if the solution is exact
                # Residual should be close to 0
                if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-9:
                    consistent = False
                    break
                else:
                    # Verify exactness with integer arithmetic if possible
                    # Or just check if w_c is close to half-integers
                    if not is_half_integer(w_c):
                        # It might still be a valid rational solution, but the problem asks for integer/half-integer
                        # If it's not half-integer, we can't use it directly unless we scale?
                        # The problem says "integer/half-integer entries".
                        # So we must round and verify.
                        w_c_rounded = round_to_half_integer(w_c)
                        # Verify the rounded solution exactly
                        # Compute residual exactly
                        # We can do a quick check with numpy first
                        residual_check = np.dot(A_c, w_c_rounded) - b_c
                        if np.max(np.abs(residual_check)) > 1e-9:
                            consistent = False
                            break
                        else:
                            # Store the rounded solution
                            W_sol[c, :] = w_c_rounded
                    else:
                        W_sol[c, :] = w_c
                        
            if not consistent:
                continue
                
            # Final exact verification
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
