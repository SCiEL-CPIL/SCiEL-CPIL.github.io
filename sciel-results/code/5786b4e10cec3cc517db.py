import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strassen-based decomposition for 2x4x5 - Rank 36
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 0, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    def to_list(mat):
        return [[int(round(mat[i, r] * 2)) / 2 for r in range(mat.shape[1])] for i in range(mat.shape[0])]

    # Verify rank 36
    if verify_decomposition(U, V, W):
        best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        best_rank = total_rank

    # Now attempt rank 35: search for a pair of columns (r1, r2) such that
    # replacing them with their sum (or a linear combination) still works.
    # The key insight: if columns r1 and r2 of U,V,W satisfy that the tensor
    # formed by their sum is still a valid rank-1 component, we can merge them.
    
    # A more practical approach: try to find a linear dependency.
    # For each pair (r1, r2), check if there exist scalars alpha, beta such that
    # the modified decomposition with column r1 replaced by alpha*U[:,r1]+beta*U[:,r2]
    # (and similarly for V, W) still satisfies the tensor equations.
    
    # Actually, the simplest structural change: try all pairs and check if
    # removing one column and adjusting the other can still work.
    
    # Let's try a different structural approach: use the fact that the two
    # 2x2x1 blocks (rank 4 each) might be reducible. The 5th column of B
    # only interacts with the 5th column, so we can potentially use a
    # different decomposition for the 2x2x1 part.
    
    # Try: Can we decompose 2x2x1 in rank 3 instead of 4?
    # C[i,k] = sum_j A[i,j] * B[j,k] for fixed k
    # This is a 2x2 matrix times a 2x1 vector = 2x1 vector
    # The tensor has 4 nonzero entries (one for each (i,j) pair contributing to each i)
    # Actually for 2x2x1: C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is rank 4 in general (no known rank-3 decomposition for 2x2x1)
    
    # Let's try a completely different structural approach:
    # Use a 2x4x5 decomposition that treats the 4x5 part differently.
    # Split B into two 2x5 halves and use Strassen on each 2x2x5, then combine.
    
    # Actually, let's try the most promising structural change:
    # Search for rank 35 by trying to merge columns from the rank-36 solution.
    
    if time.time() < deadline - 10:
        # Try to find a rank-35 solution by searching for mergeable column pairs
        # For each pair (r1, r2), try replacing with a single column that is
        # a linear combination, and check if the result is valid.
        
        # The condition for merging: if U[:,r1] = U[:,r2], V[:,r1] = V[:,r2], W[:,r1] = W[:,r2]
        # then we can just remove one. But that's unlikely.
        
        # More generally: if there exist alpha, beta such that the tensor contribution
        # of column r1 + alpha*column r2 (in U, V, W simultaneously) is zero
        # for all off-diagonal terms and preserves diagonal terms.
        
        # This is a complex search. Let's try a simpler structural change:
        # Use a different partition of the 4x5 matrix B.
        
        # Alternative: decompose as (2x4) * (4x5) = (2x2)*(2x5) + (2x2)*(2x5)
        # where we split B's rows into two groups of 2.
        # Each (2x2)*(2x5) can be done with Strassen-like methods.
        
        # For (2x2)*(2x5): split the 2x5 into (2x2) + (2x2) + (2x1)
        # (2x2)*(2x2) = rank 7 (Strassen)
        # (2x2)*(2x2) = rank 7 (Strassen)  
        # (2x2)*(2x1) = rank 4 (schoolbook)
        # Total per half: 18, two halves: 36. Same as before.
        
        # Let's try: can we share Strassen terms between the two halves?
        # The U matrix is shared (same A), but V and W differ.
        # If we use the same Strassen patterns for U in both halves,
        # we might be able to combine.
        
        # Actually, the U patterns are the same for both halves since A is the same.
        # The difference is in V and W. So we can't simply merge.
        
        # Let's try yet another approach: use the known lower bound.
        # For 2x4x5, the known lower bound is 34 (from Bini et al.)
        # But we don't have a known rank-34 or rank-35 construction.
        
        # Given the constraints, let's try to verify if rank 35 is possible
        # by a more systematic search.
        
        # For now, return the rank 36 solution which is verified.
        pass

    return best_result
# EVOLVE-BLOCK-END
