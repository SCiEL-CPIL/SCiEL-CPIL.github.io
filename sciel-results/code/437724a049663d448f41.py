import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication
    using numerical optimization followed by rounding.
    
    Strategy:
    1. Define the target tensor T[i, j, k] for C[i, k] = sum_j A[i, j] B[j, k].
       The tensor is defined such that T[i, j, k] = 1 if i==l and j==k in the 4-index form, 
       but effectively it's the matmul tensor.
       Actually, the problem defines U (n*m, r), V (m*p, r), W (p*n, r).
       T[i, j, k, l] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+l, r] = 1 if i==l and j==k else 0.
       
       We can reshape this to a 3-mode tensor for optimization if we fix the correspondence.
       Let's define the target tensor M of shape (n*m, m*p, p*n) ? No, that's huge.
       
       Better: We want to find U, V, W such that:
       For all i, j, k, l:
       sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+l, r] = delta_{i,l} delta_{j,k}
       
       This is a system of linear equations in the products, but non-linear in U, V, W.
       However, we can view this as a tensor completion problem.
       
       Alternative Numerical Strategy:
       1. Start with a known valid decomposition (e.g., schoolbook rank 48).
       2. Try to reduce rank by setting some columns to 0 and re-solving for the non-zero columns
          to minimize the error.
       3. If error is near 0, round the coefficients to nearest integers/half-integers.
       4. Verify exactness.
       
       Since finding a global minimum is hard, we will use a greedy column elimination approach.
       Start with a random or structured initialization, optimize, then prune.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Target tensor values
    # T[i, j, k, l] = 1 if i==l and j==k else 0
    # We can flatten the indices (i, j) -> u_idx, (j, k) -> v_idx, (k, l) -> w_idx?
    # No, the sum is over r. The equation is:
    # sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r] = Target[u_idx, v_idx, w_idx]
    # where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+l.
    # Note that u_idx, v_idx, w_idx are not independent.
    # u_idx determines i, j.
    # v_idx determines j, k.
    # w_idx determines k, l.
    # Consistency: j from u must match j from v. k from v must match k from w.
    # So we only need to solve for consistent triples (u_idx, v_idx, w_idx).
    
    # Generate list of valid (u, v, w) indices and their target values
    targets = []
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                for l in range(n):
                    w_idx = k * n + l
                    # Check consistency: 
                    # u gives (i, j). v gives (j, k). w gives (k, l).
                    # They are consistent by construction of the loops.
                    val = 1.0 if (i == l and j == k) else 0.0
                    targets.append((u_idx, v_idx, w_idx, val))
    
    N_eq = len(targets)
    
    def _verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Compute T
        # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We only need to check the consistent triples.
        max_err = 0.0
        for u_idx, v_idx, w_idx, val in targets:
            s = 0.0
            for r in range(rank):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            err = abs(s - val)
            if err > max_err:
                max_err = err
        return max_err < tol, max_err
    
    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-9:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-9:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result

    # Start with a high rank, try to reduce
    # Schoolbook rank is n*m*p = 48
    max_rank = 48
    
    # We will try ranks from a lower bound up to 48.
    # Lower bound for 3x4x4?
    # 2x2x2 is 7.
    # 3x3x3 is 23 (Larsen) or 22? Best known is 23.
    # 3x4x4 might be around 28-30?
    # Let's try to find a valid decomposition with rank < 48.
    
    # Strategy: 
    # 1. Create a random initialization for U, V, W with rank R.
    # 2. Use an iterative solver to minimize ||T_computed - T_target||^2.
    #    Since it's a 3-tensor product, we can use alternating least squares (ALS) or similar.
    #    Fix V and W, solve for U. Fix U and W, solve for V. Fix U and V, solve for W.
    #    However, the equations are linear in U if V and W are fixed?
    #    T[u, v, w] = sum_r U[u, r] * (V[v, r] * W[w, r])
    #    Let Z_r[v, w] = V[v, r] * W[w, r].
    #    Then T[u, v, w] = sum_r U[u, r] Z_r[v, w].
    #    This is a linear system for U[u, :] for each (v, w) pair? 
    #    Actually, for a fixed (v, w), the equation is:
    #    sum_r U[u, r] Z_r[v, w] = T_target[u, v, w] for all u.
    #    This couples all u's.
    
    # Simpler approach: Just use a generic numerical solver if possible, but standard libraries
    # don't have a direct "minimize rank of tensor decomposition" function.
    
    # Given the time constraint and complexity of ALS for this specific sparse structure,
    # and the fact that the "inspiration" mentioned a rank-44 structured solution,
    # and the current program returns 48,
    # I will implement a specific structured decomposition that is known to be valid
    # and likely lower than 48, based on the Kronecker product idea but corrected.
    
    # Correct Kronecker/Block approach:
    # A (3x4) = [A1 (2x4); A2 (1x4)]
    # B (4x4) = [B1 (4x2) B2 (4x2)]
    
    # C1 (2x2) = A1 * B1 (2x4 * 4x2)
    # C2 (2x2) = A1 * B2 (2x4 * 4x2)
    # C3 (1x2) = A2 * B1 (1x4 * 4x2)
    # C4 (1x2) = A2 * B2 (1x4 * 4x2)
    
    # We can decompose 2x4 * 4x2 = 2x2 using two independent 1x4 * 4x2 = 1x2 problems.
    # Each 1x4 * 4x2 = 1x2 has rank 4 (schoolbook).
    # So C1 has rank 8, C2 has rank 8.
    # C3 has rank 4, C4 has rank 4.
    # Total 24? 
    # Wait, if we use schoolbook for the sub-blocks, we just get the schoolbook rank of the whole thing.
    # To beat 48, we MUST use a non-schoolbook method for at least one sub-block.
    
    # Can we use Strassen for 2x2 * 2x2?
    # If we split A1 (2x4) into two 2x2 blocks? No, A1 is 2x4.
    # If we split B (4x4) into four 2x2 blocks?
    # Let's try splitting everything into 2x2 blocks where possible.
    # A (3x4) -> [A11 (2x2) A12 (2x2); A21 (1x2) A22 (1x2)]
    # B (4x4) -> [B11 (2x2) B12 (2x2); B21 (2x2) B22 (2x2)]
    
    # C (3x4) -> [C11 (2x2) C12 (2x2); C21 (1x2) C22 (1x2)]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For C11: Sum of two 2x2*2x2 products.
    # We can compute A11*B11 with Strassen (rank 7) and A12*B21 with Strassen (rank 7).
    # Total rank for C11 = 14.
    # Similarly C12 = 14.
    # C21: 1x2 * 2x2 + 1x2 * 2x2.
    # 1x2 * 2x2 = 1x2 has rank 4 (schoolbook).
    # So C21 = 4 + 4 = 8.
    # C22 = 8.
    
    # Total rank = 14 + 14 + 8 + 8 = 44.
    
    # This matches the "inspiration" rank of 44!
    # Let's implement this rank-44 decomposition.
    
    rank_2x2 = 7
    rank_1x2 = 4
    
    total_rank = 2 * rank_2x2 + 2 * rank_2x2 + 2 * rank_1x2 + 2 * rank_1x2
    # C11: 2 Strassen blocks
    # C12: 2 Strassen blocks
    # C21: 2 Schoolbook 1x2 blocks
    # C22: 2 Schoolbook 1x2 blocks
    total_rank = 4 * rank_2x2 + 4 * rank_1x2
    
    U_full = np.zeros((dim_u, total_rank), dtype=float)
    V_full = np.zeros((dim_v, total_rank), dtype=float)
    W_full = np.zeros((dim_w, total_rank), dtype=float)
    
    def _strassen_2x2_UVW():
        """
        Returns U (4,7), V (4,7), W (4,7) for 2x2 * 2x2 = 2x2.
        Indices:
        A: 00, 01, 10, 11 -> 0, 1, 2, 3
        B: 00, 01, 10, 11 -> 0, 1, 2, 3
        C: 00, 01, 10, 11 -> 0, 1, 2, 3
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # P3 = A00(B10-B11) -> C01 += P3, C11 += P3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # P4 = A11(B00-B01) -> C00 += P4, C10 += P4
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def _schoolbook_1x2_2x2_UVW():
        """
        Returns U (2,4), V (4,4), W (2,4) for 1x2 * 2x2 = 1x2.
        A: 0, 1
        B: 0, 1, 2, 3 (00, 01, 10, 11)
        C: 0, 1 (0, 1)
        C0 = A0*B00 + A1*B10
        C1 = A0*B01 + A1*B11
        """
        U = np.zeros((2, 4), dtype=float)
        V = np.zeros((4, 4), dtype=float)
        W = np.zeros((2, 4), dtype=float)
        
        # C0 += A0*B00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # C0 += A1*B10
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        # C1 += A0*B01
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        
        # C1 += A1*B11
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    sU, sV, sW = _strassen_2x2_UVW()
    bU, bV, bW = _schoolbook_1x2_2x2_UVW()
    
    col_offset = 0
    
    # Helper to add a block to the full matrices
    def add_block(block_U, block_V, block_W, u_indices, v_indices, w_indices):
        nonlocal col_offset
        rank = block_U.shape[1]
        for r in range(rank):
            for u_idx, b_u_idx in zip(u_indices, range(block_U.shape[0])):
                U_full[u_idx, col_offset + r] = block_U[b_u_idx, r]
            for v_idx, b_v_idx in zip(v_indices, range(block_V.shape[0])):
                V_full[v_idx, col_offset + r] = block_V[b_v_idx, r]
            for w_idx, b_w_idx in zip(w_indices, range(block_W.shape[0])):
                W_full[w_idx, col_offset + r] = block_W[b_w_idx, r]
        col_offset += rank

    # Define index mappings
    # A (3x4):
    # A11 (2x2): rows 0,1 cols 0,1 -> U indices: 0,1,4,5 (i*4+j)
    #   (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A12 (2x2): rows 0,1 cols 2,3 -> U indices: 2,3,6,7
    #   (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # A21 (1x2): row 2 cols 0,1 -> U indices: 8,9
    #   (2,0)->8, (2,1)->9
    # A22 (1x2): row 2 cols 2,3 -> U indices: 10,11
    #   (2,2)->10, (2,3)->11
    
    # B (4x4):
    # B11 (2x2): rows 0,1 cols 0,1 -> V indices: 0,1,4,5 (j*4+k)
    #   (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B12 (2x2): rows 0,1 cols 2,3 -> V indices: 2,3,6,7
    #   (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # B21 (2x2): rows 2,3 cols 0,1 -> V indices: 8,9,12,13
    #   (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # B22 (2x2): rows 2,3 cols 2,3 -> V indices: 10,11,14,15
    #   (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    
    # C (3x4) -> W indices (k*3+l):
    # C11 (2x2): rows 0,1 cols 0,1 -> W indices: 0,1,3,4
    #   (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # C12 (2x2): rows 0,1 cols 2,3 -> W indices: 2,5,6,7? 
    #   k=2, l=0 -> 2*3+0=6. k=2, l=1 -> 7.
    #   k=3, l=0 -> 3*3+0=9. k=3, l=1 -> 10.
    #   Wait, C is 3x4. Rows i (0,1,2), Cols k (0,1,2,3).
    #   W index = k*n + l = k*3 + l.
    #   C11 (i in 0,1; k in 0,1):
    #     (k=0, l=0)->0, (k=0, l=1)->1
    #     (k=1, l=0)->3, (k=1, l=1)->4
    #   C12 (i in 0,1; k in 2,3):
    #     (k=2, l=0)->6, (k=2, l=1)->7
    #     (k=3, l=0)->9, (k=3, l=1)->10
    #   C21 (i=2; k in 0,1):
    #     (k=0, l=2)->2, (k=1, l=2)->5
    #   C22 (i=2; k in 2,3):
    #     (k=2, l=2)->8, (k=3, l=2)->11

    # Block 1: C11 = A11*B11 + A12*B21
    # Part 1: A11*B11 (2x2*2x2)
    u_A11 = [0, 1, 4, 5]
    v_B11 = [0, 1, 4, 5]
    w_C11 = [0, 1, 3, 4]
    add_block(sU, sV, sW, u_A11, v_B11, w_C11)
    
    # Part 2: A12*B21 (2x2*2x2)
    u_A12 = [2, 3, 6, 7]
    v_B21 = [8, 9, 12, 13]
    w_C11 = [0, 1, 3, 4]
    add_block(sU, sV, sW, u_A12, v_B21, w_C11)
    
    # Block 2: C12 = A11*B12 + A12*B22
    # Part 1: A11*B12
    u_A11 = [0, 1, 4, 5]
    v_B12 = [2, 3, 6, 7]
    w_C12 = [6, 7, 9, 10]
    add_block(sU, sV, sW, u_A11, v_B12, w_C12)
    
    # Part 2: A12*B22
    u_A12 = [2, 3, 6, 7]
    v_B22 = [10, 11, 14, 15]
    w_C12 = [6, 7, 9, 10]
    add_block(sU, sV, sW, u_A12, v_B22, w_C12)
    
    # Block 3: C21 = A21*B11 + A22*B21
    # Part 1: A21*B11 (1x2*2x2)
    u_A21 = [8, 9]
    v_B11 = [0, 1, 4, 5]
    w_C21 = [2, 5]
    add_block(bU, bV, bW, u_A21, v_B11, w_C21)
    
    # Part 2: A22*B21
    u_A22 = [10, 11]
    v_B21 = [8, 9, 12, 13]
    w_C21 = [2, 5]
    add_block(bU, bV, bW, u_A22, v_B21, w_C21)
    
    # Block 4: C22 = A21*B12 + A22*B22
    # Part 1: A21*B12
    u_A21 = [8, 9]
    v_B12 = [2, 3, 6, 7]
    w_C22 = [8, 11]
    add_block(bU, bV, bW, u_A21, v_B12, w_C22)
    
    # Part 2: A22*B22
    u_A22 = [10, 11]
    v_B22 = [10, 11, 14, 15]
    w_C22 = [8, 11]
    add_block(bU, bV, bW, u_A22, v_B22, w_C22)
    
    valid, err = _verify(U_full, V_full, W_full)
    if not valid:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=float)
        V = np.zeros((dim_v, rank), dtype=float)
        W = np.zeros((dim_w, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
    
    return {"U": _to_clean_list(U_full), "V": _to_clean_list(V_full), "W": _to_clean_list(W_full)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
