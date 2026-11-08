"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Rank-minimal decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    # The tensor T_{i,j,k} has support where 0<=i<2, 0<=j<3, 0<=k<7.
    # We use a decomposition of rank n*m*p = 42, but with a smarter indexing scheme
    # that exploits the rectangular structure.
    #
    # Actually, let's think about this more carefully.
    # The standard schoolbook uses rank n*m*p = 2*3*7 = 42.
    #
    # For a 2x3 * 3x7 multiplication, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor is T_{i,k,j} = A[i,j] * B[j,k]
    # Index ranges: i in [0,1], k in [0,6], j in [0,2]
    #
    # In the trace convention W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    # The decomposition is: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # The minimal rank for this specific case... let's think.
    # For 2x3*3x7, the border rank and tensor rank might differ.
    # 
    # Let's try a decomposition where we fix j and sum over i,k.
    # For each j, we need to decompose the 2x7 matrix of (i,k) pairs.
    # The 2x7 matrix can be decomposed with rank at most 2 (since it's 2x7, rank <= 2).
    # So for each j, we use 2 terms, giving total rank 3*2 = 6? That seems too low.
    #
    # Wait, the tensor is T_{i,j,k}. For fixed j, it's a 2x7 matrix.
    # But the decomposition must work for all j simultaneously.
    #
    # Let's use the fact that n=2 is small.
    # For each (j,k) pair, the dependence on i is linear with 2 coefficients.
    # We can write: T_{i,j,k} = a_{j,k} * (i==0) + b_{j,k} * (i==1)
    # But this doesn't help directly.
    #
    # Let's just implement the schoolbook 42-rank decomposition, which is guaranteed correct.
    n, m, p = DIMENSIONS
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
    
    # Convert to lists as the interface expects
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
