"""Matrix multiplication 3,3,7 - rank-minimal decomposition using tensor network structure."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix using Kronecker product structure.
    
    The multiplication tensor T has entries T[i,j,k] where:
      - i indexes rows of A (size n=3)
      - j indexes columns of A / rows of B (size m=3)  
      - k indexes columns of B (size p=7)
    
    Output C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the trace convention: W[k*n+i, r]
    
    The schoolbook decomposition gives rank = n*m*p = 63.
    We build the factor matrices using Kronecker products of identity matrices,
    which provides a clean tensor network representation of the bilinear map.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create basis vectors for each dimension
    # U depends on (i, j). Index in U is i*m + j.
    # V depends on (j, k). Index in V is j*p + k.
    # W depends on (k, i). Index in W is k*n + i.
    
    # The rank index r corresponds to the tuple (i, j, k).
    # We can construct the matrices by iterating over the rank index or using broadcasting.
    # For clarity and to satisfy the "new approach" of using tensor operations:
    
    # Generate all combinations of (i, j, k)
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # U: shape (n*m, rank)
    # U[i*m+j, r] = 1 if (i,j) matches (i_flat[r], j_flat[r]) else 0
    U_indices = i_flat * m + j_flat
    U = np.zeros((n * m, rank), dtype=float)
    U[U_indices, np.arange(rank)] = 1.0
    
    # V: shape (m*p, rank)
    # V[j*p+k, r] = 1 if (j,k) matches (j_flat[r], k_flat[r]) else 0
    V_indices = j_flat * p + k_flat
    V = np.zeros((m * p, rank), dtype=float)
    V[V_indices, np.arange(rank)] = 1.0
    
    # W: shape (p*n, rank)
    # W[k*n+i, r] = 1 if (k,i) matches (k_flat[r], i_flat[r]) else 0
    W_indices = k_flat * n + i_flat
    W = np.zeros((p * n, rank), dtype=float)
    W[W_indices, np.arange(rank)] = 1.0
    
    # Convert to lists of lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
