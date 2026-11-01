"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product
from collections import defaultdict

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    deadline = time.time() + 110.0  # Safety margin
    
    # The tensor T[u,v,w] = 1 iff u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    # i.e., T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'
    
    # Strategy: Exploit the block structure.
    # For each pair (i, i') of output rows, we have a sub-tensor:
    #   T_sub[i*m+j, j'*p+k, k'*n+i'] for fixed i, i'
    # This sub-tensor is non-zero only when i = i' (diagonal in the n×n block structure)
    # Wait, that's not quite right. Let me reconsider.
    
    # Actually, T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'
    # So for a fixed output row i', the non-zero entries are at positions
    # (i*m+j, j*p+k) where i=i', j=j', k=k' for any j,k.
    # This means the tensor has a block diagonal structure in the (u,w) indices.
    
    # Let me think differently. Group by the middle index j.
    # For fixed j, the non-zero entries are at (i*m+j, j*p+k, k*n+i) for i in {0,1}, k in {0,...,14}
    # So for each j, we have a 2×15 sub-tensor (in i,k indices).
    
    # The full tensor is the direct sum over j of these 2×15 sub-tensors,
    # but they share the same U and V indices in a structured way.
    
    # Actually, let me use a different decomposition strategy.
    # The matrix multiplication (2×2) * (2×15) can be viewed as:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # for i in {0,1}, k in {0,...,14}
    
    # This is a 2×15 matrix product with a 2×2 intermediate.
    # The schoolbook rank is 2*2*15 = 60.
    
    # Key insight: We can use the fact that the intermediate dimension is 2.
    # For 2×2 * 2×p, we can potentially do better than 4p.
    
    # Let's try to find a decomposition of the 2×2 * 2×15 tensor.
    # The tensor T has entries T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'
    
    # For each (i, k) pair in the output, we need:
    #   T[i*2+j, j*15+k, k*2+i] = 1 for j in {0,1}
    # So the output entry (i,k) depends on A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # This is a linear combination. We can try to find a low-rank decomposition.
    
    # Approach: Use the structure to find a decomposition.
    # The tensor can be written as:
    #   T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # Let's try to find a decomposition with rank < 60.
    
    # One idea: For each i, the terms involving i are:
    #   sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is a 2×15 tensor in the (j,k) indices.
    
    # For fixed i, this tensor is:
    #   sum_{j=0}^{1} sum_{k=0}^{14} e_j ⊗ e_{j*15+k} ⊗ e_k
    # (where the first index is j, second is j*15+k, third is k)
    
    # This is a 2×15 matrix multiplication tensor (with the middle dimension being 2).
    # The rank of this sub-tensor is at most 2*15 = 30 (schoolbook).
    
    # But we have two such sub-tensors (one for i=0, one for i=1), and they share
    # some structure.
    
    # Let me try a different approach: search for a decomposition using the
    # known structure.
    
    # For the 2×2 * 2×p problem, the minimum rank is known to be 4p for p=1,2
    # (schoolbook), but for larger p, it might be less.
    
    # Actually, for 2×2 * 2×p, the tensor is:
    #   sum_{i,j,k} e_{2i+j} ⊗ e_{2j+k} ⊗ e_{pk+i}  (wait, this doesn't look right)
    
    # Let me re-index. The tensor T has:
    #   T[u,v,w] = 1 iff u = i*m + j, v = j*p + k, w = k*n + i
    #   for some i in {0,1}, j in {0,1}, k in {0,...,p-1}
    
    # So T[i*2+j, j*15+k, k*2+i] = 1 for i,j in {0,1}, k in {0,...,14}
    
    # This is a 4×30×30 tensor with 60 non-zero entries.
    
    # Let's try to find a decomposition.
    
    # Strategy: Use the fact that for each k, the entries are at
    #   (i*2+j, j*15+k, k*2+i) for i,j in {0,1}
    # So for fixed k, we have 4 entries.
    
    # For k=0: (0,0,0), (0,15,1), (2,0,0), (2,15,1)
    # Wait, that's not right. Let me recalculate.
    # For i=0, j=0, k=0: u=0*2+0=0, v=0*15+0=0, w=0*2+0=0
    # For i=0, j=1, k=0: u=0*2+1=1, v=1*15+0=15, w=0*2+0=0
    # For i=1, j=0, k=0: u=1*2+0=2, v=0*15+0=0, w=0*2+1=1
    # For i=1, j=1, k=0: u=1*2+1=3, v=1*15+0=15, w=0*2+1=1
    
    # So for k=0, the non-zero entries are at:
    #   (0,0,0), (1,15,0), (2,0,1), (3,15,1)
    
    # For k=1:
    #   (0,1,0), (1,16,0), (2,1,1), (3,16,1)
    
    # And so on.
    
    # This structure suggests that we can group by k.
    # For each k, we have 4 rank-1 terms.
    
    # The total rank is at most 4*15 = 60 (schoolbook).
    
    # To find a lower rank, we need to find linear dependencies.
    
    # Let's try a computational approach: set up the linear system and solve.
    
    # The tensor T can be flattened to a vector of length N*M*P = 4*30*30 = 3600.
    # A rank-r decomposition corresponds to finding r vectors U_col, V_col, W_col
    # such that T = sum_{r=0}^{R-1} U_col[r] ⊗ V_col[r] ⊗ W_col[r]
    
    # This is equivalent to finding R such that the matrix of inner products
    # <T, U_col[r] ⊗ V_col[r] ⊗ W_col[r]> = 1 for the correct entries.
    
    # This is a non-linear problem, but we can try a greedy approach.
    
    # Greedy approach:
    # 1. Start with the schoolbook decomposition (rank 60).
    # 2. Try to remove terms by finding linear combinations.
    
    # Alternatively, we can try to find a decomposition from scratch.
    
    # Let me try a different structural approach.
    
    # The tensor T[i*2+j, j*15+k, k*2+i] = 1
    # can be rewritten as:
    #   T = sum_{i,j,k} e_{2i+j} ⊗ e_{15j+k} ⊗ e_{2k+i}
    
    # Let's try to factor this.
    
    # For fixed j, the terms are:
    #   sum_{i,k} e_{2i+j} ⊗ e_{15j+k} ⊗ e_{2k+i}
    # = sum_{i,k} (e_{2i+j} ⊗ e_{15j+k} ⊗ e_{2k+i})
    
    # For j=0:
    #   sum_{i,k} e_{2i} ⊗ e_{k} ⊗ e_{2k+i}
    # = e_0 ⊗ e_0 ⊗ e_0 + e_0 ⊗ e_1 ⊗ e_2 + ... + e_0 ⊗ e_14 ⊗ e_28
    #   + e_2 ⊗ e_0 ⊗ e_1 + e_2 ⊗ e_1 ⊗ e_3 + ... + e_2 ⊗ e_14 ⊗ e_29
    
    # For j=1:
    #   sum_{i,k} e_{2i+1} ⊗ e_{15+k} ⊗ e_{2k+i}
    # = e_1 ⊗ e_15 ⊗ e_0 + e_1 ⊗ e_16 ⊗ e_2 + ... + e_1 ⊗ e_29 ⊗ e_28
    #   + e_3 ⊗ e_15 ⊗ e_1 + e_3 ⊗ e_16 ⊗ e_3 + ... + e_3 ⊗ e_29 ⊗ e_29
    
    # This doesn't immediately suggest a lower rank decomposition.
    
    # Let me try a computational search.
    
    # Approach: Use the fact that the tensor is sparse and has a specific pattern.
    # Try to find a decomposition with rank 59 or lower by searching for
    # linear dependencies.
    
    # For now, let's return the schoolbook decomposition as a baseline.
    # In a future iteration, we can try to find a lower rank decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
