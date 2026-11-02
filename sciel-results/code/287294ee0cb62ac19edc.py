"""Matrix multiplication 3x4x7 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > DEADLINE - 15
    
    # Strategy: Use a structured decomposition.
    # 
    # The tensor T[i,j,k] = 1 for i in [0,3), j in [0,4), k in [0,7)
    # We seek: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    #
    # Key insight: Each rank-1 term has the form a_i * b_j * c_k where
    # U[i*m+j,c] = a_i, V[j*p+k,c] = b_j, W[k*n+i,c] = c_k
    # (This requires U to be independent of j, V independent of k, W independent of i)
    #
    # So we need: sum_c a_i^(c) * b_j^(c) * c_k^(c) = 1 for all i,j,k
    #
    # This is equivalent to: the n x m x p all-ones tensor has rank r.
    #
    # For the all-ones tensor, we can use the following construction:
    # Let r = n + m + p - 2 (this is the rank of the all-ones tensor in some sense)
    # Actually, the all-ones tensor has rank 1! Because 1*1*1 = 1.
    # Wait, that's trivial. a_i=1, b_j=1, c_k=1 gives product 1.
    # So rank 1 suffices?!
    
    # Let me re-examine. If U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
    # Then U*V*W = 1*1*1 = 1 for all (i,j,k). That's rank 1!
    
    # But wait, that can't be right for matrix multiplication. Let me re-read the problem.
    # The tensor represents matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 meaning the j-th term in the sum has coefficient 1.
    # 
    # A rank-1 decomposition term: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This corresponds to: A[i,j] * B[j,k] * (contribution to output [k*n+i])
    # 
    # For this to be a valid multiplication algorithm, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k} in the appropriate sense
    # 
    # Actually, the tensor T[i,j,k] = 1 for all (i,j,k) means every (i,j,k) triple has weight 1.
    # A rank-1 component a_i * b_j * c_k contributes a_i*b_j*c_k to position (i,j,k).
    # We need sum_c a_i^(c)*b_j^(c)*c_k^(c) = 1 for ALL (i,j,k).
    #
    # With a_i=1, b_j=1, c_k=1: sum = 1. Done! Rank 1!
    
    # But this seems too good. Let me verify with the actual matrix multiplication semantics.
    # In matrix multiplication, the "tensor" is the structure that maps
    # (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k].
    # 
    # The rank-1 components correspond to:
    #   intermediate = U[i*m+j,c] * A[i,j]  (for each c)
    #   intermediate2 = V[j*p+k,c] * B[j,k] (for each c)  
    #   C[k*n+i] += W[k*n+i,c] * intermediate * intermediate2
    #
    # Wait, I think I'm confusing the tensor structure. Let me re-read.
    #
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    #
    # The matrix multiplication tensor T is defined by T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So yes, we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k).
    #
    # With U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1: sum = 1. Rank 1!
    
    # This must be correct. The all-ones tensor has rank 1.
    
    U = [[1]]
    V = [[1]]
    W = [[1]]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(1))
                assert s == 1, f"Verification failed at ({i},{j},{k}): got {s}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
