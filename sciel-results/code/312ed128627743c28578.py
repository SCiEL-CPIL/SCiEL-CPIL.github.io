"""Optimized tensor decomposition for 3x3 x 3x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # Start with schoolbook rank 63
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank by removing linearly dependent columns
    # First, let's try a simple approach: use random vectors and solve
    
    # For rank r, we need to find U, V, W such that for all i,j,k:
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # This is equivalent to: for each (i,k), the vector [V[k*p+j, r] * W[j*n+i, r] for j=0..p-1, r=0..rank-1]
    # dotted with [U[i*m+k, r] for r] should give 1 for all j.
    
    # Let's try a structured approach: since T is all-ones, maybe we can use
    # a low-rank approximation that's exact.
    
    # Key insight: T[i,j,k] = 1 can be factored as sum_r a_{i,k}^{(r)} * b_{k,j}^{(r)} * c_{j,i}^{(r)}
    
    # For the all-ones tensor, one simple factorization is:
    # Use r = max(n, p) factors where we separate the i and j dependencies
    
    # Actually, let's try: rank = n + p - 1 or similar
    # For T[i,j,k] = 1, we can write:
    # T[i,j,k] = sum_{r=1}^{R} u_{i,k}^{(r)} v_{k,j}^{(r)} w_{j,i}^{(r)}
    
    # One approach: fix k and note that for each k, we need a factorization of the 
    # n x p all-ones matrix into rank-r factors with shared structure.
    
    # Let me try a concrete construction with rank = n * p = 21
    # For each (i,j) pair, use one term:
    # T[i,j,k] = sum_{(i',j')} delta(i,i') * delta(j,j') * 1 for k... no this doesn't work.
    
    # Better: Let's use the fact that 1 = sum_{r} alpha_r * beta_r * gamma_r
    # We need this to hold for all (i,j,k) simultaneously with the index structure.
    
    # Simplest non-trivial: rank = n * p = 21
    # U[i*m+k, (i',j')] = delta(i, i') for all k
    # V[k*p+j, (i',j')] = delta(j, j') for all k  
    # W[j*n+i, (i',j')] = 1
    # Then sum_{(i',j')} U[i*m+k, (i',j')] * V[k*p+j, (i',j')] * W[j*n+i, (i',j')]
    # = sum_{(i',j')} delta(i,i') * delta(j,j') * 1 = 1. Yes!
    
    # So rank = n*p = 21 works!
    
    rank = n * p  # 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i_prime in range(n):
        for j_prime in range(p):
            for i in range(n):
                for k in range(m):
                    U[i * m + k, col] = 1.0 if i == i_prime else 0.0
            for k in range(m):
                for j in range(p):
                    V[k * p + j, col] = 1.0 if j == j_prime else 0.0
            for j in range(p):
                for i in range(n):
                    W[j * n + i, col] = 1.0
            col += 1
    
    # Verify
    # Check a few entries
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = sum(U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    col = 0
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                U[i * m + k, col] = 1.0
                                V[k * p + j, col] = 1.0
                                W[j * n + i, col] = 1.0
                                col += 1
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        # All checks passed
        pass
    
    # Convert to lists of floats (integers where possible)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
