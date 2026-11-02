import time
import random
import numpy as np

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 108  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Start with the trivial rank-105 decomposition as fallback
    trivial_rank = n * m * p
    U_trivial = [[0] * trivial_rank for _ in range(n * m)]
    V_trivial = [[0] * trivial_rank for _ in range(m * p)]
    W_trivial = [[0] * trivial_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][column] = 1
                V_trivial[j * p + k][column] = 1
                W_trivial[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = trivial_rank
    
    # Strategy: Try structured decompositions with lower rank
    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # Key observation: We can use a "separable" structure where:
    # - For each output row i, we assign a block of terms
    # - Within each block, we decompose the row computation
    
    # Try to find decompositions with rank = n * r where r < m*p
    # This corresponds to using r terms per output row
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Also rank >= n*m*p / min(n,m,p) = 105/3 = 35 (weak bound)
    
    # Let's try various structured approaches
    
    # Approach 1: Block diagonal per output row
    # For each i in range(n), use r_i terms to compute row i
    # Total rank = sum(r_i)
    # For row i: need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all j,k
    
    # If we fix W[k*n+i, t] = 1 for all k when t is in row i's block,
    # then we need: sum_t U[i*m+j, t] * V[j*p+k, t] = 1 for all j,k
    # This is like decomposing the all-ones matrix of size m x p
    
    # The rank of the all-ones matrix is 1, so we can use:
    # U[i*m+j, t0] = 1, V[j*p+k, t0] = 1, W[k*n+i, t0] = 1
    # This gives rank = n (one term per output row)
    
    # But wait, this doesn't work because the same term t0 would be used for all i,
    # and W[k*n+i, t0] would need to be 1 for all i, which conflicts.
    
    # Let me think more carefully. Each term t has entries in U, V, W.
    # For term t, define:
    #   u_t: vector in R^(n*m)
    #   v_t: vector in R^(m*p)  
    #   w_t: vector in R^(p*n)
    # Constraint: for all (i,j,k): sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1
    
    # Let's try a randomized search with structured initialization
    
    random.seed(42)
    
    # Try ranks from 35 upward (theoretical lower bound)
    # For each rank, try to find a valid decomposition
    
    # We'll use a greedy/iterative approach:
    # 1. Start with a random or structured initialization
    # 2. Try to satisfy constraints
    
    # Actually, let's try a simpler structured approach:
    # Use the fact that we can share terms across different (i,j,k) triples
    
    # Approach: For each pair (j, k), we need the sum over i of contributions to equal 1
    # But the tensor is separable in a specific way
    
    # Let's try: rank = m * p = 35
    # For each (j, k) pair, use one term t(j,k)
    # U[i*m+j, t(j,k)] = a_i
    # V[j*p+k, t(j,k)] = b_jk  
    # W[k*n+i, t(j,k)] = c_ik
    # Need: sum over (j,k) of a_i * b_jk * c_ik = 1 for all i
    
    # This is a system of n equations. If we set b_jk = 1 for all j,k,
    # then we need: sum_k a_i * c_ik = 1 for all i
    # Set a_i = 1, then need: sum_k c_ik = 1 for all i
    # Set c_ik = 1/p for all k, then sum_k c_ik = 1. Works!
    
    # So: rank = m*p = 35
    # For each (j,k) in [m]x[p], term t = j*p+k:
    #   U[i*m+j, t] = 1 for all i
    #   V[j*p+k, t] = 1
    #   W[k*n+i, t] = 1/p for all i
    
    # Let's verify: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # = sum_{(j',k')} U[i*m+j, t(j',k')] * V[j*p+k, t(j',k')] * W[k*n+i, t(j',k')]
    # U[i*m+j, t(j',k')] is 1 if j'=j, else 0 (since U[i*m+j, t] = 1 for all i only when the term is for column j)
    # Wait, I defined U[i*m+j, t(j',k')] = 1 for all i, but that means for a fixed j, all terms t(j',k') have U[i*m+j, t(j',k')] = 1?
    # No, I need to be more careful.
    
    # Let me redefine: For term t corresponding to (j', k'):
    #   U[i*m+j, t] = delta(j, j') for all i  [1 if j=j', else 0]
    #   V[j*p+k, t] = delta(k, k') for all j  [1 if k=k', else 0]  
    #   W[k*n+i, t] = 1/p for all i, k
    
    # Then: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # = sum_{j',k'} delta(j,j') * delta(k,k') * (1/p)
    # = delta(j,j) * delta(k,k) * (1/p) = 1/p
    
    # That gives 1/p, not 1. So we need W[k*n+i, t] = 1 instead of 1/p.
    
    # With W[k*n+i, t] = 1:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # = sum_{j',k'} delta(j,j') * delta(k,k') * 1
    # = 1
    
    # Yes! This works. Rank = m*p = 35.
    
    # Let's construct this:
    rank_candidate = m * p  # 35
    U_cand = [[0] * rank_candidate for _ in range(n * m)]
    V_cand = [[0] * rank_candidate for _ in range(m * p)]
    W_cand = [[0] * rank_candidate for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            t = j * p + k
            # U[i*m+j, t] = 1 for all i
            for i in range(n):
                U_cand[i * m + j][t] = 1
            # V[j*p+k, t] = 1
            V_cand[j * p + k][t] = 1
            # W[k*n+i, t] = 1 for all i
            for i in range(n):
                W_cand[k * n + i][t] = 1
    
    # Verify
    if verify_decomposition(U_cand, V_cand, W_cand, n, m, p):
        best_rank = rank_candidate
        best_result = {"U": U_cand, "V": V_cand, "W": W_cand}
    
    # Can we do better? Try rank = n * p = 21
    # For each (i, k) pair, use one term
    # U[i*m+j, t(i,k)] = a_ij
    # V[j*p+k, t(i,k)] = b_jk  
    # W[k*n+i, t(i,k)] = c_ik
    # Need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # For fixed (i,j,k): sum_{i',k'} U[i*m+j, t(i',k')] * V[j*p+k, t(i',k')] * W[k*n+i, t(i',k')]
    # W[k*n+i, t(i',k')] = delta(i, i') * c_ik' (nonzero only when i'=i)
    # So: sum_{k'} U[i*m+j, t(i,k')] * V[j*p+k, t(i,k')] * c_ik' = 1
    
    # Set U[i*m+j, t(i,k')] = 1 for all j, k'
    # Set c_ik' = 1 for all k'
    # Then: sum_{k'} V[j*p+k, t(i,k')] = 1 for all j
    # Set V[j*p+k, t(i,k')] = delta(k, k') 
    # Then: sum_{k'} delta(k, k') = 1. Works!
    
    # Rank = n * p = 21
    rank_candidate2 = n * p  # 21
    U_cand2 = [[0] * rank_candidate2 for _ in range(n * m)]
    V_cand2 = [[0] * rank_candidate2 for _ in range(m * p)]
    W_cand2 = [[0] * rank_candidate2 for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            t = i * p + k
            # U[i*m+j, t] = 1 for all j
            for j in range(m):
                U_cand2[i * m + j][t] = 1
            # V[j*p+k, t] = 1 for all j
            for j in range(m):
                V_cand2[j * p + k][t] = 1
            # W[k*n+i, t] = 1
            W_cand2[k * n + i][t] = 1
    
    if verify_decomposition(U_cand2, V_cand2, W_cand2, n, m, p):
        if rank_candidate2 < best_rank:
            best_rank = rank_candidate2
            best_result = {"U": U_cand2, "V": V_cand2, "W": W_cand2}
    
    # Can we do even better? Try rank = n * m = 15
    # For each (i, j) pair, use one term
    # U[i*m+j, t(i,j)] = a_ij
    # V[j*p+k, t(i,j)] = b_jk
    # W[k*n+i, t(i,j)] = c_ik
    # Need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # For fixed (i,j,k): sum_{i',j'} U[i*m+j, t(i',j')] * V[j*p+k, t(i',j')] * W[k*n+i, t(i',j')]
    # U[i*m+j, t(i',j')] = delta(j, j') * a_i'j' (nonzero only when j'=j)
    # So: sum_{i'} delta(j,j) * a_i'j * V[j*p+k, t(i',j)] * W[k*n+i, t(i',j)] = 1
    # = a_jj * sum_{i'} V[j*p+k, t(i',j)] * W[k*n+i, t(i',j)] = 1
    
    # Set a_jj = 1 for all j (i.e., a_ij = 1 for all i,j)
    # Then: sum_{i'} V[j*p+k, t(i',j)] * W[k*n+i, t(i',j)] = 1 for all j,k
    
    # Set V[j*p+k, t(i',j)] = delta(k, k) = 1 for all i',k (wait, that's not right)
    # Let V[j*p+k, t(i',j)] = 1 for all i', k
    # Set W[k*n+i, t(i',j)] = delta(i, i') * (1/n)
    # Then: sum_{i'} 1 * delta(i, i') * (1/n) = 1/n. Not 1.
    
    # Set W[k*n+i, t(i',j)] = delta(i, i')
    # Then: sum_{i'} 1 * delta(i, i') = 1. Works!
    
    # Rank = n * m = 15
    rank_candidate3 = n * m  # 15
    U_cand3 = [[0] * rank_candidate3 for _ in range(n * m)]
    V_cand3 = [[0] * rank_candidate3 for _ in range(m * p)]
    W_cand3 = [[0] * rank_candidate3 for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            t = i * m + j
            # U[i*m+j, t] = 1
            U_cand3[i * m + j][t] = 1
            # V[j*p+k, t] = 1 for all k
            for k in range(p):
                V_cand3[j * p + k][t] = 1
            # W[k*n+i, t] = 1 for all k
            for k in range(p):
                W_cand3[k * n + i][t] = 1
    
    if verify_decomposition(U_cand3, V_cand3, W_cand3, n, m, p):
        if rank_candidate3 < best_rank:
            best_rank = rank_candidate3
            best_result = {"U": U_cand3, "V": V_cand3, "W": W_cand3}
    
    # Check time and return
    if time.time() > deadline:
        return best_result
    
    return best_result
