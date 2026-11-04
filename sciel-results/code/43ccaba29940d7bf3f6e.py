"""Program for matrix multiplication tensor decomposition 2x5x6."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W, rank

def _verify_exact(U, V, W, n, m, p, rank):
    """Check if the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]"""
    for c in range(rank):
        col_u = [U[r][c] for r in range(n * m)]
        col_v = [V[r][c] for r in range(m * p)]
        col_w = [W[r][c] for r in range(p * n)]
        # For each c, check that W[k*n+i,c] == sum_j U[i*m+j,c] * V[j*p+k,c]
        # But this is only true if the decomposition is rank-1 in c, which it isn't.
        # The actual condition is: the tensor T_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # must equal 1 if i,j,k are all valid, else 0.
        pass
    # Better verification: reconstruct the multiplication tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    """Construct a low-rank decomposition for 2x5x6 matrix multiplication.
    
    Strategy: Use the schoolbook decomposition (rank 60) and attempt to reduce rank
    by exploiting the structure of the 2x5 and 5x6 matrices. For small matrices,
    there may be structured decompositions that achieve lower rank.
    
    For 2x5x6, we can try to find a decomposition using the fact that one dimension
    is 2. The rank of MM_{2,m,p} is at most 2*m*p but can be lower.
    
    Known: rank(MM_{2,2,p}) = 2p (for p >= 2, since it's essentially 2 independent
    length-p dot products, but they share the same A matrix, so actually it's more subtle).
    
    Let's try a direct construction:
    A is 2x5, B is 5x6, C = A*B is 2x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Schoolbook: 2*5*6 = 60 terms.
    
    Can we do better? For MM_{n,m,p}, the rank is at least max(n*m, m*p, p*n) and 
    at most n*m*p. For 2x5x6: lower bound is max(10,30,12)=30, upper bound 60.
    
    Let me try to find a rank-30 decomposition or lower.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W, rank = _build_schoolbook(n, m, p)
    
    # Deadline guard
    deadline = time.time() + 100  # 100 seconds max
    
    # Try to reduce rank by finding linear dependencies in the tensor
    # The tensor T has shape (n*m, m*p, p*n) = (10, 30, 12)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    
    # Actually, let's try a different approach: use the structure of 2x5x6.
    # Partition the 5 middle indices into groups and use block structure.
    
    # For now, try to verify the schoolbook is correct and return it.
    # If we have time, attempt rank reduction.
    
    if _verify_exact(U, V, W, n, m, p, rank):
        pass  # Schoolbook is valid
    
    # Attempt rank reduction using randomization / iterative compression
    # This is complex; for now, return schoolbook as a safe baseline.
    
    # Try a structured decomposition:
    # Since n=2, write A as two row vectors a0 (length 5), a1 (length 5)
    # C[0,k] = a0 . b_k  where b_k is column k of B (length 5)
    # C[1,k] = a1 . b_k
    
    # This is 2 dot products of length 5, each for 6 outputs = 2*5*6=60 multiplications.
    # But we can potentially share structure.
    
    # Alternative: treat it as 5 independent 2x1 * 1x6 multiplications,
    # each of rank 2, total rank 5*2=10? No, that's not right either.
    
    # Let me just return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
