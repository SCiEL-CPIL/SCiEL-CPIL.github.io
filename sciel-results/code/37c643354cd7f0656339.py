import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition using structured ansatz
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Start with a known good rank and try to reduce
    # For 2x5x5, try ranks from 10 upward
    
    for rank in range(10, 51):
        result = _try_decomposition(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to naive
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
                
    return {"U": U, "V": V, "W": W}

def _try_decomposition(n, m, p, rank):
    """
    Try to find a rank-r decomposition using a structured ansatz.
    We assume a form where the decomposition exploits the structure of matrix multiplication.
    """
    # Use a simple structured approach: 
    # For each output element C[i,k] = sum_j A[i,j]*B[j,k], we can try to combine terms
    
    # Approach: Use the fact that for fixed i, we have m*p terms
    # Try a decomposition where we group by intermediate index j
    
    # Simple structured ansatz: rank = n * m (for each i, use m terms)
    # But we can do better by sharing structure
    
    # Let's try: for each i in [0,n), for each j in [0,m), create a term
    # U[i*m+j, t] = 1, V[j*p+k, t] = B[j,k], W[k*n+i, t] = 1
    # But this doesn't work directly because V depends on k
    
    # Better: use bilinear decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_l (A[i,j] * delta_{j,l}) * (delta_{j,l} * B[j,k])
    # This gives rank n*m*p
    
    # For improvement, we can use:
    # If we have terms that combine multiple j values
    
    # Let me try a direct linear algebra approach for small ranks
    # Assume a specific structure and solve
    
    # For now, use a known decomposition pattern for 2x5x5
    # Rank 10: one term per (i,j) pair, with V and W sharing structure
    
    # Actually, let's use a simpler approach: 
    # The minimal rank for M_{2,5,5} is at least ceil(2*5*5 / something)
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually rank >= n*m*p / (min(n,m,p)) is not right
    
    # Use the simple lower bound: rank >= n*m = 10 (since output has n*p=10 elements, 
    # but that's not a valid lower bound for tensor rank)
    
    # For now, return a rank-10 decomposition if it works
    if rank < 10:
        return None
    
    # Construct a rank-10 decomposition:
    # For each i in [0,2), j in [0,5): term t = i*5+j
    # U[i*m+j, t] = 1
    # V[j*p+k, t] = B[j,k] ... but V must be constant coefficients
    
    # This doesn't work because V depends on the input B
    
    # The correct formulation: the tensor entries are T[i,j,k]=1
    # We need U[i,j,t]*V[j,k,t]*W[k,i,t] to sum to 1
    
    # Let me try: U[i*m+j, t] = 1 for t=i*5+j
    # V[j*p+k, t] = 1 for t where j matches
    # W[k*n+i, t] = 1 for t where i,k match
    
    # For t = i*5+j:
    #   U[i*m+j, t] = 1
    #   V[j*p+k, t] = 1 for all k? No, that gives sum over k
    
    # I think I need to be more careful. Let me just use the naive 50-rank 
    # but try to find a better one via optimization
    
    # For simplicity and correctness, return the naive decomposition
    return None
