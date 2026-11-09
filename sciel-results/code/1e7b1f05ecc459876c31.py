"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6

    # Schoolbook rank = n*m*p = 90
    # Try to reduce by combining terms via linear dependencies
    # For small dimensions, the tensor rank of M[n,m,p] is known to be:
    # M[3,3,3]: rank 23 (Strassen)
    # M[3,4,4]: rank 36
    # M[3,5,5]: rank 48
    # M[3,5,6]: rank 54 (lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30)
    # Actually, the tensor rank of M[3,5,6] is not exactly known, but schoolbook gives 90.
    # We can use known decompositions or try to find better ones.

    # For now, implement Strassen-like or other known fast matrix multiplication
    # For 3x5 * 5x6, we can split into blocks and use fast multiplication
    # But the simplest improvement is to use the known result that rank M[n,m,p] <= n*m*p
    # and try to find a better decomposition.

    # Let's implement a generic approach: try to find rank < n*m*p by solving
    # the linear system for the tensor decomposition.
    # The tensor T[i,j,k] = 1 if we're multiplying (i,j) * (j,k) to get (i,k)
    # Actually, the matrix multiplication tensor M[n,m,p] has entries:
    # T[i,j,k,l] where i in [0,n), j in [0,m), k in [0,p), l in [0,n)
    # M[i,j,k,l] = delta_{j,l} if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Wait, the tensor is T[i,j,k] -> C[i,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has 4 indices: (i,j,k,l) but actually it's a 3-tensor:
    # For each output (i,k), we sum over j: A[i,j] * B[j,k]
    # So the tensor is M[i,j,k] where i is row of A, j is column of A / row of B, k is column of B
    # And the output is at position (i,k)
    # The tensor decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_r (sum_j A[i,j] U[j,r]) * (sum_j B[j,k] V[r,j]) * W[i,k,r]
    # No, that's not right. The standard form is:
    # C[i,k] = sum_r U[i,j,r] A[i,j] V[r,j] B[j,k] W[i,k,r]
    # But that's not linear in A and B.

    # The correct tensor decomposition is:
    # C[i,k] = sum_r (U[i,_,r] dot A[:,r]) ... no.

    # Let me think again. The matrix multiplication tensor M[n,m,p] is a 3-tensor
    # with indices (i, j, k) where i in [n], j in [m], k in [p].
    # But actually, it's a 4-way tensor: (i, j, k, l) where C[i,l] = sum_j A[i,j] B[j,k]... 

    # I'm confusing myself. Let me use the standard formulation:
    # We want to compute C = AB where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor M has entries M[i,j,k] = A[i,j] * B[j,k] for each (i,j,k)
    # But the "tensor" for matrix multiplication is actually a 3-tensor T[i,j,k] 
    # where the output C[i,k] = sum_j T[i,j,k] with T being bilinear in A and B.
    
    # The rank of the matrix multiplication tensor is the minimum number of 
    # multiplications needed. The schoolbook algorithm uses n*m*p multiplications.
    
    # For 3x5 * 5x6, schoolbook uses 3*5*6 = 90 multiplications.
    # The lower bound is max(15, 30, 18) = 30 (from the "bilinear complexity" bounds).
    # Known results: rank M[3,3,3] = 23 (Strassen), rank M[2,2,2] = 7 (Strassen)
    # rank M[3,4,4] = 36, rank M[3,5,5] = 48, rank M[3,5,6] is not exactly known
    # but upper bound is 54 (from Bini et al.) or lower.
    
    # For this task, let's just use the schoolbook decomposition since finding
    # a better one requires solving a hard optimization problem.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
