import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 32, p*n = 32
    # Rank lower bound: max(4, 32, 32) = 32
    # Strassen block approach gives 56.
    # We search for a decomposition with rank < 56, ideally 32-40.
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # The tensor T_{i,j,k} = delta_{i,k} (since C = A * B, and (AB)_{i,k} = sum_j A_{i,j} B_{j,k})
    # Wait, the task is: A is 2x2, B is 2x16. Output C is 2x16.
    # C_{i,k} = sum_{j=0}^{1} A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = delta_{i,k} ? No.
    # The multiplication tensor T is defined by T_{(i,j),k} ... wait.
    # Standard tensor decomposition for matrix multiplication:
    # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i,j,k) and T_{i,j,k} = 1 if the operation is A_{i,j} * B_{j,k} contributing to C_{i,k}.
    # Actually, the tensor T in the decomposition sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} = T_{i,j,k}
    # Here, the "multiplication" is not a standard matrix multiply of two large matrices, but a specific operation.
    # The task says: "multiplying a 2 by 2 matrix by a 2 by 16 matrix".
    # So A is 2x2, B is 2x16.
    # U has shape (n*m, rank) = (4, rank). U_{(i,j), r} corresponds to A_{i,j}.
    # V has shape (m*p, rank) = (32, rank). V_{(j,k), r} corresponds to B_{j,k}.
    # W has shape (p*n, rank) = (32, rank). W_{(k,i), r} corresponds to C_{i,k}.
    # The identity is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{(i,j), (j,k), (k,i)} = 1 for valid i,j,k.
    # Indices:
    # U index: i*m + j (0..3)
    # V index: j*p + k (0..31)
    # W index: k*n + i (0..31)
    
    # We want to find U, V, W with rank R such that:
    # For all i in 0..1, j in 0..1, k in 0..15:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations (which don't exist in the sum), the sum should be 0?
    # No, the tensor is sparse. The only non-zero entries of T are at (i,j,k) where the term A_{i,j}B_{j,k} appears in C_{i,k}.
    # This is true for ALL i,j,k.
    # Wait, is it?
    # C_{i,k} = sum_{j=0}^{1} A_{i,j} B_{j,k}
    # So the tensor T_{(i,j),(j,k),(k,i)} = 1 for all i in {0,1}, j in {0,1}, k in {0..15}.
    # Are there any other non-zero entries? No.
    
    # This is a structured tensor.
    # Let's verify the rank.
    # If we use the trivial decomposition:
    # For each i, j, k, we can have a rank-1 term.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # This requires 2*2*16 = 64 ranks.
    
    # Can we do better?
    # Yes, because the sum over j is shared for a fixed i,k.
    # C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}
    # This is a 2-term sum.
    # We can use Strassen-like tricks?
    # Strassen works for 2x2 * 2x2. Here we have 2x2 * 2x16.
    # The "16" is just a repetition of the 2x2 structure? No, B has 2 rows and 16 columns.
    # It's like multiplying 2x2 by 16 distinct 2x1 matrices.
    # Or 2x2 by a 2x2 matrix, 8 times?
    # If we group columns of B into pairs: B = [B_0, B_1, ..., B_7] where B_r is 2x2.
    # Then C = A * B = [A B_0, A B_1, ..., A B_7].
    # This is 8 independent 2x2 matrix multiplications.
    # Rank of 2x2 matmul is 7 (Strassen).
    # So total rank = 8 * 7 = 56.
    # This matches the current program's score.
    
    # Can we do better than 56?
    # The tensor is the direct sum of 8 copies of the 2x2 matmul tensor?
    # If the blocks are completely independent (disjoint support in V and W), then the rank is the sum of the ranks.
    # V indices for block r: j*p + 2r and j*p + 2r+1.
    # W indices for block r: (2r)*n + i and (2r+1)*n + i.
    # Since V and W have disjoint indices for different r, the tensor is indeed the direct sum.
    # Therefore, the rank is exactly 8 * rank(2x2 matmul) = 8 * 7 = 56.
    
    # Is it possible that the rank of 2x2 matmul is less than 7?
    # No, it is a known open problem whether rank is 7 or higher, but 7 is the best known upper bound (Strassen).
    # Actually, the rank of 2x2 matrix multiplication is known to be 7. (Strassen 1969).
    # Wait, is it proven to be exactly 7?
    # The complexity of 2x2 matrix multiplication is known to be 7.
    # So the minimum rank for each block is 7.
    # Since the blocks are independent, the total minimum rank is 56.
    
    # Therefore, the current program likely already achieves the minimum possible rank.
    # Any attempt to go lower would fail.
    # We should just return the rank 56 solution, but perhaps verify it quickly.
    
    # Let's construct the rank 56 solution efficiently.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U: A indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # U index = i*m + j
            # li=0 -> i=0,j=0 -> 0
            # li=1 -> i=0,j=1 -> 1
            # li=2 -> i=1,j=0 -> 2
            # li=3 -> i=1,j=1 -> 3
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V: B indices
            # V index = j*p + k
            # Block r covers k=2r, 2r+1
            # li=0 -> j=0, k=2r -> 0*16 + 2r = 2r
            # li=1 -> j=0, k=2r+1 -> 2r+1
            # li=2 -> j=1, k=2r -> 1*16 + 2r = 16+2r
            # li=3 -> j=1, k=2r+1 -> 16+2r+1
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
                    
            # W: C indices
            # W index = k*n + i
            # li=0 -> C[0,0] of block -> i=0, k=2r -> 2r*2 + 0 = 4r
            # li=1 -> C[0,1] of block -> i=0, k=2r+1 -> (2r+1)*2 + 0 = 4r+2
            # li=2 -> C[1,0] of block -> i=1, k=2r -> 2r*2 + 1 = 4r+1
            # li=3 -> C[1,1] of block -> i=1, k=2r+1 -> (2r+1)*2 + 1 = 4r+3
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
