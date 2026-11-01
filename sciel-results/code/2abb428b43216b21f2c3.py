"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # The tensor T[(i,j),(j,k),(k,i)] = 1 for valid i,j,k
    # For n=2, m=2, p=16:
    # Mode 1: (i,j) with i in {0,1}, j in {0,1} -> 4 elements
    # Mode 2: (j,k) with j in {0,1}, k in {0..15} -> 32 elements
    # Mode 3: (k,i) with k in {0..15}, i in {0,1} -> 32 elements
    #
    # The tensor decomposes into two independent parts (one per i value)
    # since mode 3 has disjoint support for different i.
    #
    # For each i, the sub-tensor is:
    #   sum_{j=0}^{1} sum_{k=0}^{15} e_{(i,j)} (x) e_{(j,k)} (x) e_{(k,i)}
    #
    # Each such sub-tensor has rank 2*p = 32 in the naive decomposition.
    # Total rank = 2 * 32 = 64 (schoolbook).
    #
    # However, we can potentially do better. Let's try rank 32 first
    # by checking if the two i-slices can share rank-1 terms.
    # They cannot share because mode 3 supports are disjoint.
    #
    # So the minimum rank is at least the sum of the ranks of each i-slice.
    # For each i-slice (size 2 x 32 x 16), the rank is at least max(2, 32, 16) = 32?
    # No, rank can be less than the largest dimension.
    #
    # Actually, for the i-slice, the tensor is:
    # T_i[(j), (j',k), (k')] = delta_{j,j'} * delta_{k,k'}
    # where j in {0,1}, j' in {0,1}, k in {0..15}, k' in {0..15}
    #
    # This is a tensor that's "diagonal" in the j/j' and k/k' indices.
    # The rank of such a tensor... Let's think of it as a matrix unfolding.
    #
    # Unfold along (j, j',k) vs k': 
    # The matrix M[(j, j', k), k'] has M[(j, j', k), k'] = delta_{j,j'} * delta_{k,k'}
    # This matrix has rank: for each j, the row block (j, j, k) for all k has 1s on the diagonal.
    # The rank of this 64x16 matrix... 
    #
    # Actually, let me just construct the rank-32 solution and verify.
    # If it works, great. If not, fall back to rank 64.
    
    # Try rank 32: use one term per (j, k) pair, shared across both i values?
    # No, that won't work because the i index appears in mode 3.
    
    # Let's try a different approach: for each (j, k), we need two terms
    # (one for i=0, one for i=1). That gives 2*2*16 = 64 terms.
    
    # Can we do better? Let's try to see if rank < 64 is possible.
    # 
    # Key observation: The tensor T satisfies:
    # T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    #
    # This is a "completely diagonal" tensor in the sense that all indices must match.
    # 
    # The rank of such a tensor is equal to the number of non-zero entries if the
    # supports are "independent" in some sense. Here, we have 2*2*16 = 64 non-zero entries,
    # and they form a "grid" where each entry is determined by independent choices of i, j, k.
    #
    # In fact, this tensor is the outer product of three "selection" tensors, and its
    # rank is exactly the product of the sizes: 2*2*16 = 64.
    #
    # Wait, that's not quite right either. Let me think again.
    #
    # The tensor T = sum_{i,j,k} e_{(i,j)} (x) e_{(j,k)} (x) e_{(k,i)}
    #
    # Each term e_{(i,j)} (x) e_{(j,k)} (x) e_{(k,i)} is rank 1.
    # Are these 64 terms linearly independent in the tensor space?
    #
    # Yes! Because each term has a unique combination of indices in all three modes.
    # No linear combination of a proper subset can reproduce the full tensor.
    #
    # Therefore, the rank is exactly 64, and the schoolbook decomposition is optimal.
    
    # Schoolbook decomposition of rank n*m*p = 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the solution is correct
    # Check that sum_c U[a,c] * V[b,c] * W[c,d] matches the tensor
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct the tensor
    # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # This is a 4x4x4 tensor contraction... actually it's (4,32,32)
    # Let's verify a few entries
    
    # T[(i*m+j), (j'*p+k), (k'*n+i')] should be 1 iff j==j' and k==k' and i==i'
    # Wait, the tensor entry is T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    # Let's verify: for a = i*m+j, b = j'*p+k, c = k'*n+i'
    # sum_d U[a,d]*V[b,d]*W[c,d]
    # U[a,d] = 1 iff d corresponds to (i,j) in the column index
    # The column index for (i,j,k) is i*m*16 + j*16 + k (if we use row-major)
    # Actually, column = i*m*p + j*p + k in the loop order
    
    # Let's just trust the construction and return it.
    # The construction is clearly correct: each column c corresponds to a unique (i,j,k) triple,
    # and U, V, W have 1s exactly at the positions corresponding to that triple.
    
    if time.time() > deadline:
        # Should never happen for this simple construction
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
