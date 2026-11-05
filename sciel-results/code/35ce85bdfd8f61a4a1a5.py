import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Build a rank-28 decomposition using Strassen on 2x2 blocks.
    rank = 28
    random.seed(20240517)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    strassen_block(0, 1, 0, 1, 0, 1)
    strassen_block(0, 1, 0, 1, 2, 3)
    schoolbook_block([0, 1], [0, 1], [4])
    schoolbook_block([0, 1], [2], [0, 1])
    schoolbook_block([0, 1], [2], [2, 3])
    schoolbook_block([0, 1], [2], [4])

    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback to rank-30 schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by checking for linear dependencies among the rank-28 terms
    # We have 28 vectors in R^(30) (the tensor entries). Check if any subset of 27 spans the same space.
    # If rank of the 28x30 matrix (or 30x28) is < 28, we can reduce.
    # Actually, we need to check if the 28 terms are linearly dependent in the tensor space.
    # The tensor space is R^(n*m*p) = R^30.
    # We form a 30x28 matrix M where M[i,j,k, c] = U[c,i,j] * V[c,j,k] * W[c,k,i]
    # If rank(M) < 28, there is a linear relation sum_c alpha_c * T_c = 0.
    # This allows us to express one term as a combination of others, reducing the rank.
    
    # However, a more direct approach: 
    # The current decomposition uses 28 terms. 
    # Can we do better? 
    # Known lower bound for 2x3x5 is 24. 
    # Let's try a random search for a rank-27 decomposition using the existing structure as a starting point.
    # Or simply check if the 28 terms are linearly independent.
    
    # Construct the 30x28 matrix
    M = np.zeros((n*m*p, rank))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i*m*p + j*p + k, c] = uc[i, j] * vc[j, k] * wc[k, i]
    
    # Compute rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    
    if rank_M < rank:
        # There is a dependency. We can try to eliminate one term.
        # This is complex to implement robustly in limited time.
        # For now, if rank_M < 28, it's a sign we might be able to do better,
        # but we don't have a guaranteed algorithm to find the new decomposition instantly.
        # However, if the terms are linearly dependent, the "effective" rank is lower.
        # But we need an explicit decomposition with fewer terms.
        
        # Let's try a greedy elimination.
        # If col c is in the span of others, we can remove it.
        # This requires solving for coefficients.
        
        # Given the time constraint and complexity, we stick to the valid rank-28 solution
        # unless we can quickly find a dependency and resolve it.
        
        # Let's try a simple check: if any column is a linear combination of others.
        # This is equivalent to checking if removing any column drops the rank.
        
        # If we find such a column, we can't just remove it because the coefficients might not be simple.
        # We need to rewrite the decomposition.
        
        # For this iteration, we will just return the valid rank-28 solution.
        # The potential for improvement is noted, but implementing a full rank-reduction algorithm
        # (like Bini's or others) is too complex for the remaining time/scope.
        
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
