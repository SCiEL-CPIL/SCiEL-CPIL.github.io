import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We want to construct U, V, W such that:
    # sum_c U[i, c] V[j, c] W[k, c] = 1 if i==i', j==j', k==k' (for the multiplication tensor)
    # Actually, the standard decomposition for matrix multiplication C = AB is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is defined such that T[i, j, k] = A[i, j] B[j, k] C[i, k] is not quite right.
    # The task defines the output factor W[k*n+i, r].
    # The condition is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
    # No, the standard matrix multiplication tensor T_{i, j, k} has rank 1 components corresponding to each (i, j, k) triple?
    # Let's look at the indices in the prompt:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Entry U[i*m+j, r] corresponds to A[i, j]
    # Entry V[j*p+k, r] corresponds to B[j, k]
    # Entry W[k*n+i, r] corresponds to C[k, i] ? Or C[i, k]?
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # Usually C is n x p. C[i, k].
    # If W is indexed by k*n + i, then W[k, i] corresponds to C[i, k] if we transpose the index logic?
    # Let's assume the standard decomposition:
    # For each i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # We need a rank-1 component that contributes to C[i, k] when A[i, j] and B[j, k] are 1.
    # The product A[i, j] * B[j, k] contributes to C[i, k].
    # So we need a term where U picks out A[i, j], V picks out B[j, k], and W picks out C[i, k].
    # But the indices in W are k*n + i. This is the linear index for a p x n matrix.
    # If C is n x p, then C[i, k] is at index i*p + k.
    # The prompt specifies W has shape (p*n, rank) and index k*n+i.
    # This implies W is effectively storing the transpose of C, or the index mapping is specific.
    # Let's verify the "trace convention".
    # If we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, and W[k*n+i, r] = 1,
    # then sum_r ... = 1 if there exists r such that all are 1.
    # If we create one rank-1 component for each (i, j, k):
    # r = i*m*p + j*p + k
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # Then for a specific (i, j, k), the term is 1.
    # Does this sum to the identity tensor?
    # The matrix multiplication tensor is T_{i, j, k} = 1 if we are computing C[i, k] from A[i, j] and B[j, k]?
    # Actually, the tensor for matrix multiplication is usually defined as T[i, j, k] = A[i, j] B[j, k] C[i, k].
    # The rank is the minimum number of rank-1 tensors summing to T.
    # The schoolbook algorithm uses n*m*p terms.
    # Each term corresponds to a specific (i, j, k).
    # Term r corresponds to (i_r, j_r, k_r).
    # U has 1 at (i_r, j_r) in row i_r*m + j_r.
    # V has 1 at (j_r, k_r) in row j_r*p + k_r.
    # W has 1 at (k_r, i_r) in row k_r*n + i_r.
    # This matches the prompt's index convention for W.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Vectorized construction
    # Generate indices i, j, k
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # These arrays have length n*m*p
    # U index: i*m + j
    u_rows = i_idx * m + j_idx
    # V index: j*p + k
    v_rows = j_idx * p + k_idx
    # W index: k*n + i
    w_rows = k_idx * n + i_idx
    
    # The column index is just 0..rank-1
    cols = np.arange(rank)
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
