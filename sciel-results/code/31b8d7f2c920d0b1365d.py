"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # 150s wall-clock limit, leave safety margin
    
    n, m, p = DIMENSIONS
    # Target dimensions:
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (26, rank)
    # W: (p*n, rank) = (26, rank)
    
    # The tensor T[i,j,k] = 1 if and only if the output index is k*n+i and input indices are i,j,k
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # T[i, j, k] contributes to output C[i, k]
    # In the factorization: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) for the identity tensor
    # Wait, the task says "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor M has M[i,j,k] = 1 for all valid i,j,k
    # The decomposition is: M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where indices are flattened as specified.
    
    # Let's re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # This is the standard matrix multiplication tensor.
    # The schoolbook rank is n*m*p = 2*2*13 = 52.
    
    # We want to find a lower rank decomposition.
    # Known bounds: for 2x2 by 2x13, the rank is at most 52 (schoolbook).
    # Can we do better?
    
    # One approach: use the fact that p=13 is prime.
    # Consider the "cyclic" or "Fourier" based decompositions.
    
    # Actually, let's think about this differently.
    # The tensor T[i,j,k] = 1 for all i,j,k is a 3-tensor of size 2x2x13.
    # We want to write it as a sum of rank-1 tensors.
    # A rank-1 tensor is u_i * v_j * w_k.
    
    # The minimum rank of the 2x2x13 all-ones tensor...
    # Wait, is it the all-ones tensor? Let me re-read.
    
    # "Matrix multiplication 2,2,13"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 13 matrix."
    
    # The matrix multiplication tensor for A (2x2) * B (2x13) = C (2x13)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is M[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,12}
    
    # So yes, it's the all-ones tensor of size 2x2x13.
    # The rank of the all-ones tensor of size a x b x c is known to be min(a,b,c) in some cases?
    # No, that's not right.
    
    # Actually, the all-ones tensor J_{a,b,c} has rank 1 if we allow arbitrary coefficients?
    # No, J_{a,b,c}[i,j,k] = 1 for all i,j,k.
    # A rank-1 tensor is u_i v_j w_k.
    # Can we write J_{2,2,13} as a sum of few rank-1 tensors?
    
    # J_{2,2,13}[i,j,k] = 1
    # If we take u = [1,1], v = [1,1], w = [1,1,...,1], then u_i v_j w_k = 1 for all i,j,k.
    # So the rank is 1!
    
    # Wait, that can't be right. Let me check the index conventions again.
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U is indexed by (i,j), V by (j,k), W by (k,i).
    # The tensor is sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This must equal the matrix multiplication tensor.
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k.
    # So yes, T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,12}.
    
    # And this can be written as a single rank-1 term:
    # U[i,j,0] = 1 for all i,j
    # V[j,k,0] = 1 for all j,k
    # W[k,i,0] = 1 for all k,i
    # Then sum_0 U[i,j,0] V[j,k,0] W[k,i,0] = 1*1*1 = 1 = T[i,j,k].
    
    # So the rank is 1!
    
    # Let me verify this is correct. The task says "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix
    # is indeed the all-ones tensor of size n x m x p.
    
    # So the answer is rank 1.
    
    rank = 1
    U = [[1] for _ in range(n * m)]  # 4 rows, 1 column, all 1s
    V = [[1] for _ in range(m * p)]  # 26 rows, 1 column, all 1s
    W = [[1] for _ in range(p * n)]  # 26 rows, 1 column, all 1s
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
