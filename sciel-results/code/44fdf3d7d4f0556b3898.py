import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen's 7-rank decomposition for 2x2x2 matrix multiplication.
    U, V, W have shape (4, 7) each.
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    # Strassen's 7 products
    # M1 = (A00 - A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A10 - A00) * (B01 + B11)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: rows are (i,j) pairs, columns are the 7 Strassen factors
    # U[i*2+j, r]
    U = [
        [1,  0, 1,  0, 1,  0,  0],  # i=0,j=0: A00
        [0,  1, 0,  0, 0,  0,  0],  # i=0,j=1: A01
        [0,  1, 0,  0, 1, -1, -1],  # i=1,j=0: A10
        [-1, 1, 0,  1, 1,  0,  0],  # i=1,j=1: A11
    ]
    # V: rows are (j,k) pairs, columns are the 7 Strassen factors
    # V[j*2+k, r]
    V = [
        [1,  1, 0, -1, 1,  0,  0],  # j=0,k=0: B00
        [0,  0, 0, -1, 0,  1,  1],  # j=0,k=1: B01
        [1,  0, 1,  0, 1,  1,  1],  # j=1,k=0: B10
        [1,  0, -1, 0,  1,  0,  0],  # j=1,k=1: B11
    ]
    # W: rows are (k,i) pairs, columns are the 7 Strassen factors
    # W[k*2+i, r]
    W = [
        [1,  0, 0,  0, -1,  0,  1],  # k=0,i=0: C00
        [0,  0, 1,  0,  1,  0,  0],  # k=1,i=0: C01
        [1,  1, 0,  0, -1,  0,  0],  # k=0,i=1: C10
        [1, -1, 1,  0,  0,  1,  0],  # k=1,i=1: C11
    ]
    return U, V, W

def _combine_decompositions(n, m, p, decomps, offsets):
    """Combine multiple decompositions.
    decomps: list of (U_d, V_d, W_d, i_offset, j_offset, k_offset, i_size, j_size, k_size)
    offsets: not used, kept for clarity
    """
    total_rank = sum(len(U_d[0]) for U_d, V_d, W_d, *_ in decomps)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_d, V_d, W_d, i_off, j_off, k_off, i_sz, j_sz, k_sz in decomps:
        rank_d = len(U_d[0])
        for i in range(i_sz):
            for j in range(j_sz):
                for r in range(rank_d):
                    U[(i + i_off) * m + (j + j_off)][col_offset + r] = U_d[i * j_sz + j][r]
        for j in range(j_sz):
            for k in range(k_sz):
                for r in range(rank_d):
                    V[(j + j_off) * p + (k + k_off)][col_offset + r] = V_d[j * k_sz + k][r]
        for k in range(k_sz):
            for i in range(i_sz):
                for r in range(rank_d):
                    W[(k + k_off) * n + (i + i_off)][col_offset + r] = W_d[k * i_sz + i][r]
        col_offset += rank_d
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to stay safely under 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook decomposition of rank n*m*p = 36
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = [row[:] for row in U_sb]
    best_V = [row[:] for row in V_sb]
    best_W = [row[:] for row in W_sb]
    
    # Strategy: Use Strassen for 2x2x2 sub-blocks where applicable
    # For 2x3x6, we can partition the 3 columns of A and 6 columns of B.
    # 
    # Approach 1: Split into 2x2x2 blocks
    # A is 2x3, B is 3x6. We can think of this as:
    #   C = A * B = [A1 | A2] * [[B1], [B2]] where A1 is 2x1, A2 is 2x2, B1 is 1x6, B2 is 2x6
    #   C = A1*B1 + A2*B2
    # A1*B1 is 2x1 * 1x6 = 2x6, which is an outer product (rank 2 in tensor sense? No, it's a rank-1 matrix but the tensor decomposition for 2x1x6 has rank 2*1*6=12 schoolbook, but can be done in rank 2 since it's just scaling)
    # Actually, for 2x1x6: T[i,j,k] = 1 for i in {0,1}, j={0}, k in {0..5}
    # This is a rank-2 tensor: sum over r=0,1 of u_i * v_j * w_k where we need u_i*v_j*w_k = 1
    # We can set v_0 = 1, and for each k, we need sum_r u_i,r * w_k,r = 1 for all i,k
    # This is achievable with rank 2: u_0=1,u_1=1 for r=0; w_k,0=1 for all k. Then sum = 1.
    # Wait, that's rank 1 if u and w are constant. Let me reconsider.
    # For 2x1x6, the tensor is just a 2x6 matrix (since j is fixed). The tensor rank is the matrix rank, which is 1 (all entries are 1, so it's rank 1 as a matrix).
    # So 2x1x6 has tensor rank 1? No, tensor rank for a 2x1x6 tensor...
    # A rank-1 tensor of shape (2,1,6) is u_i * v_j * w_k. Since j is only 0, v_0 is a scalar.
    # So we need u_i * w_k = 1 for all i,k. This requires u_0=u_1=1 and w_0=...=w_5=1, with v_0=1. That's rank 1.
    # So 2x1x6 has tensor rank 1.
    
    # A2*B2 is 2x2 * 2x6. This is 2x2x6.
    # We can further split B2 into two 2x3 blocks: B2 = [B2a | B2b]
    # A2*B2a is 2x2x3, A2*B2b is 2x2x3.
    # 
    # For 2x2x2, Strassen gives rank 7.
    # For 2x2x3, we can split the 3 into 2+1:
    #   2x2x3 = 2x2x2 + 2x2x1
    #   2x2x1 has tensor rank 2 (it's a 2x2 matrix, rank 2)
    #   So 2x2x3 <= 7 + 2 = 9
    # Similarly for the other 2x2x3.
    # Total: 1 (for 2x1x6) + 9 + 9 = 19
    # 
    # But we can do better. Let's think about 2x2x6 directly.
    # 2x2x6: split 6 into 3+3, each 2x2x3 <= 9, so total <= 18.
    # Or split 6 into 2+2+2: three 2x2x2 blocks, each rank 7, total 21. Worse.
    # 
    # Actually, let's be more careful. The decomposition is not simply additive in this way
    # because the U, V, W matrices must be consistent across the full problem.
    
    # Let's try a different structured approach:
    # Use Strassen-like ideas for the full 2x3x6.
    
    # Alternative: Try to find a decomposition by exploiting the structure.
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0..5}.
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1.
    
    # Let's try a decomposition based on polynomial interpolation or similar.
    
    # For now, let's try combining Strassen 2x2x2 with schoolbook for the rest.
    # Partition: 
    #   j in {0,1} handled by Strassen-like, j=2 handled by schoolbook
    #   But this doesn't directly work because Strassen is for 2x2x2.
    
    # Let's try: split the problem into:
    #   Part 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
    #   Part 2: i in {0,1}, j in {0,1}, k in {2,3,4,5} -> 2x2x4, use schoolbook (rank 16) or better
    #   Part 3: i in {0,1}, j=2, k in {0..5} -> 2x1x6, rank 1
    #   Total: 7 + 16 + 1 = 24
    
    # For 2x2x4, we can split into 2x2x2 + 2x2x2, each rank 7, total 14.
    # So total: 7 + 14 + 1 = 22.
    
    # Let's implement this.
    
    # Decomposition 1: 2x2x2 for (i in {0,1}, j in {0,1}, k in {0,1})
    U1, V1, W1 = _strassen_2x2x2()
    
    # Decomposition 2: 2x2x2 for (i in {0,1}, j in {0,1}, k in {2,3})
    U2, V2, W2 = _strassen_2x2x2()
    
    # Decomposition 3: 2x2x2 for (i in {0,1}, j in {0,1}, k in {4,5})
    U3, V3, W3 = _strassen_2x2x2()
    
    # Decomposition 4: 2x1x6 for (i in {0,1}, j=2, k in {0..5})
    # Rank 1: U[i,0]=1, V[0,0]=1, W[k,0]=1 for all
    U4 = [[1], [1]]  # 2x1, rows are i (since j is fixed, U index is i*m+j but we're in a sub-problem)
    # In the sub-problem, U has shape (i_sz * j_sz, rank) = (2*1, 1) = (2,1)
    # V has shape (j_sz * k_sz, rank) = (1*6, 1) = (6,1)
    # W has shape (k_sz * i_sz, rank) = (6*2, 1) = (12,1)
    V4 = [[1] for _ in range(6)]  # 6x1
    W4 = [[1] for _ in range(12)]  # 12x1
    
    # Now combine all four decompositions
    # Decomposition 1: i_off=0, j_off=0, k_off=0, i_sz=2, j_sz=2, k_sz=2
    # Decomposition 2: i_off=0, j_off=0, k_off=2, i_sz=2, j_sz=2, k_sz=2
    # Decomposition 3: i_off=0, j_off=0, k_off=4, i_sz=2, j_sz=2, k_sz=2
    # Decomposition 4: i_off=0, j_off=2, k_off=0, i_sz=2, j_sz=1, k_sz=6
    
    decomps = [
        (U1, V1, W1, 0, 0, 0, 2, 2, 2),
        (U2, V2, W2, 0, 0, 2, 2, 2, 2),
        (U3, V3, W3, 0, 0, 4, 2, 2, 2),
        (U4, V4, W4, 0, 2, 0, 2, 1, 6),
    ]
    
    U_combined, V_combined, W_combined = _combine_decompositions(n, m, p, decomps, None)
    
    # Verify
    if _verify_decomposition(n, m, p, U_combined, V_combined, W_combined):
        rank_combined = len(U_combined[0])
        if rank_combined < best_rank:
            best_rank = rank_combined
            best_U = U_combined
            best_V = V_combined
            best_W = W_combined
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to optimize further if time permits
    # We can try to find a lower rank by random perturbation and local search
    
    # For now, return the best we have
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
