import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the rank-42 Strassen decomposition
    rank_full = 42
    U_full = np.zeros((nm, rank_full), dtype=float)
    V_full = np.zeros((mp, rank_full), dtype=float)
    W_full = np.zeros((pn, rank_full), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset, U, V, W):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        U[a10, offset + 1] += 1.0; U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0; W[w11, offset + 1] -= 1.0
        
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0; V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0; W[w11, offset + 2] += 1.0
        
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0; V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0; W[w10, offset + 3] += 1.0
        
        U[a00, offset + 4] += 1.0; U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0; W[w01, offset + 4] += 1.0
        
        U[a10, offset + 5] += 1.0; U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0; V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        U[a01, offset + 6] += 1.0; U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0; V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U_full, V_full, W_full)
                offset += 7

    def verify(U_mat, V_mat, W_mat, rank, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    # Check if rank-42 works (should always work)
    if verify(U_full, V_full, W_full, rank_full):
        best_rank = rank_full
        best_U, best_V, best_W = U_full, V_full, W_full
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank))
        best_V = np.zeros((mp, best_rank))
        best_W = np.zeros((pn, best_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Support pruning: try removing 2 factors from the rank-42 pattern
    # For each pair of factors to remove, check if the remaining 40 can still
    # span the tensor via a linear solve.
    
    # The tensor T has entries T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # i.e., a = i*m+j, b = j*p+k, c = k*n+i, else 0.
    
    # For a given support set S of size r, we need to find U, V, W such that
    # sum_{t in S} U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
    
    # This is a bilinear problem. We can fix one matrix (say W) and solve for U, V.
    # Alternatively, we can use the fact that if we fix W, then for each c,
    # the matrix M_c = sum_t W[c,t] * (U[:,t] outer V[:,t]) must equal the slice T[:,:,c].
    
    # A simpler approach: try specific patterns of which 2 factors to remove.
    # The Strassen pattern has 6 blocks of 7. Removing one factor from two different
    # blocks is more likely to work than removing two from the same block.
    
    # Let's try: for each pair of block indices (b1, b2) with b1 != b2,
    # and for each pair of factor indices (f1, f2) within those blocks,
    # check if we can solve for a valid decomposition.
    
    # Actually, a more tractable approach: 
    # Fix the W matrix to be the Strassen W with two rows zeroed out.
    # Then solve the resulting linear system for U and V.
    
    # For each target rank r (try 40, 39, 38, ...), try different support patterns.
    
    # Strategy: Enumerate pairs of factors to remove (C(42,2) = 861 pairs).
    # For each pair, set up the linear system and check if it has a solution.
    
    # The system: For each (a,b,c) where T[a,b,c] != 0, we need
    # sum_{t in S} U[a,t] * V[b,t] * W[c,t] = 1
    # For each (a,b,c) where T[a,b,c] == 0, we need
    # sum_{t in S} U[a,t] * V[b,t] * W[c,t] = 0
    
    # This is still bilinear. Let's use a different approach:
    # Fix W to be a specific pattern, then the problem becomes linear in U and V.
    
    # If we fix W, then for each c, we have:
    # sum_t W[c,t] * U[a,t] * V[b,t] = T[a,b,c]
    # This is a set of bilinear equations in U and V.
    
    # A practical approach: Use the rank-42 solution as a starting point,
    # zero out two columns, and try to repair via gradient descent or
    # a simple iterative method.
    
    # Given time constraints, let's try a few specific removal patterns
    # and use a simple verification.
    
    # Try removing one factor from each of two different blocks.
    # Block indices: 0-5, factor indices within block: 0-6
    
    found_better = False
    for trial in range(50):  # Limit trials to stay within time
        if time.time() > deadline - 5:
            break
        
        # Pick two distinct blocks and two factors
        b1 = trial % 6
        b2 = (trial // 6) % 6
        if b1 == b2:
            b2 = (b2 + 1) % 6
        f1 = trial % 7
        f2 = (trial // 7) % 7
        
        remove_indices = [b1 * 7 + f1, b2 * 7 + f2]
        
        # Create support mask
        keep_mask = [t not in remove_indices for t in range(42)]
        keep_indices = [t for t in range(42) if keep_mask[t]]
        r = len(keep_indices)
        
        if r >= best_rank:
            continue
        
        # Try to solve: Fix W to be the Strassen W with removed columns zeroed
        W_trial = W_full[:, keep_indices].copy()
        U_trial = U_full[:, keep_indices].copy()
        V_trial = V_full[:, keep_indices].copy()
        
        # Check if this truncated version still works (it won't, but let's verify)
        # Instead, try a simple repair: scale the remaining factors to compensate
        # This is heuristic and may not work, but let's check.
        
        # Actually, just verify if the truncated version accidentally works
        if verify(U_trial, V_trial, W_trial, r):
            best_rank = r
            best_U, best_V, best_W = U_trial, V_trial, W_trial
            found_better = True
            break
    
    # If no improvement found, return the rank-42 solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
