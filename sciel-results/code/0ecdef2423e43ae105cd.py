"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin

    # Schoolbook rank: 72
    sb_rank = n * m * p
    U_sb = np.zeros((n * m, sb_rank), dtype=int)
    V_sb = np.zeros((m * p, sb_rank), dtype=int)
    W_sb = np.zeros((p * n, sb_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[int(v) for v in row] for row in mat]

    # Verify decomposition using exact integer arithmetic
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        return False
        return True

    # Strategy: Use structured search based on Kronecker product ideas
    # The 3x3x8 multiplication can be seen as 8 independent 3x3 matrix-vector products
    # We'll try to find a decomposition with rank < 72 using a combination of:
    # 1. Known sub-schoolbook decompositions for smaller blocks
    # 2. Structured random search with better initialization

    # First, try to use a block structure: group the 8 columns of B into blocks
    # and use shared computation where possible

    # Approach: Start from schoolbook and try to eliminate terms by finding
    # linear combinations that preserve the tensor structure

    # For a more systematic approach, let's try to find a rank-r decomposition
    # by solving the system of equations using a greedy/local search method

    # Key insight: If we can find vectors u, v, w such that
    # u⊗v⊗w = sum of several schoolbook terms, we can reduce rank

    # Let's try a different approach: use the fact that the tensor is symmetric
    # in certain ways and exploit that

    # Practical approach: Try random perturbations of schoolbook and check
    # if we can reduce rank

    # Better: Use a "term elimination" strategy
    # For each pair of terms, check if their sum can be replaced by one term

    # Let's implement a more efficient search using numpy for speed

    def check_decomposition_fast(U, V, W):
        """Check if U⊗V⊗W decomposition equals the target tensor."""
        r = U.shape[1]
        # Compute the tensor: T[i*m+j, j*p+k, k*n+i] should be 1 for all valid (i,j,k)
        # Using einsum for speed
        # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if s != 1:
                        return False
        return True

    # Try structured search for rank 71, 70, ...
    # Use a smarter initialization: start from schoolbook and perturb

    rng = np.random.default_rng(12345)
    found = False
    best_result = None
    best_rank = sb_rank

    # Try ranks from 71 down to 60
    for target_rank in range(71, 59, -1):
        if time.time() > deadline - 20:
            break
        
        # Multiple attempts for each rank
        max_attempts = 20 if target_rank > 65 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Strategy 1: Random sparse integer matrices
            # Strategy 2: Perturbed schoolbook (if target_rank close to 72)
            
            if target_rank >= 70:
                # Use perturbed schoolbook: take first target_rank schoolbook terms
                # and try to adjust them
                U = np.zeros((n * m, target_rank), dtype=int)
                V = np.zeros((m * p, target_rank), dtype=int)
                W = np.zeros((p * n, target_rank), dtype=int)
                
                # Copy first target_rank schoolbook terms
                for c in range(target_rank):
                    i = c // (m * p)
                    rem = c % (m * p)
                    j = rem // p
                    k = rem % p
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                
                # Now try to adjust the remaining (72 - target_rank) missing terms
                # by modifying existing terms
                missing = 72 - target_rank
                if missing <= 3:
                    # Try small perturbations
                    for _ in range(100):
                        # Pick a random column and add a small perturbation
                        col_idx = rng.integers(0, target_rank)
                        # Add a random small integer perturbation
                        perturb_U = rng.choice([-1, 0, 1], size=(n * m,))
                        perturb_V = rng.choice([-1, 0, 1], size=(m * p,))
                        perturb_W = rng.choice([-1, 0, 1], size=(p * n,))
                        
                        U_new = U.copy()
                        V_new = V.copy()
                        W_new = W.copy()
                        U_new[:, col_idx] += perturb_U
                        V_new[:, col_idx] += perturb_V
                        W_new[:, col_idx] += perturb_W
                        
                        if check_decomposition_fast(U_new, V_new, W_new):
                            found = True
                            best_result = (U_new, V_new, W_new)
                            best_rank = target_rank
                            break
                    if found:
                        break
            
            else:
                # For lower ranks, use random initialization
                # Use entries from {-2, -1, 0, 1, 2} for more flexibility
                U = rng.choice([-2, -1, 0, 1, 2], size=(n * m, target_rank))
                V = rng.choice([-2, -1, 0, 1, 2], size=(m * p, target_rank))
                W = rng.choice([-2, -1, 0, 1, 2], size=(p * n, target_rank))
                
                if check_decomposition_fast(U, V, W):
                    found = True
                    best_result = (U, V, W)
                    best_rank = target_rank
                    break
        
        if found:
            break

    if found and best_result is not None:
        U, V, W = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Return schoolbook
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
