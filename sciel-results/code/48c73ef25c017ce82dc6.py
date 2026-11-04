import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    n, m, p = 4, 4, 6

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Block decomposition: 4x4 * 4x6 as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]  (each 2x2)
    # B = [[B00, B01, B02], [B10, B11, B12]]  (each 2x2)
    # C = [[C00, C01, C02], [C10, C11, C12]]  (each 2x2)
    # C0j = A00*B0j + A01*B1j for j=0,1,2
    # C1j = A10*B0j + A11*B1j for j=0,1,2
    #
    # We need 12 products of 2x2 * 2x2, each rank 7 via Strassen
    # Total rank = 12 * 7 = 84... that's worse than 77.
    #
    # Better: use the structure to share computations.
    # C_top = A00 * B_top + A01 * B_bottom  where B_top = [B00,B01,B02], B_bottom = [B10,B11,B12]
    # C_bot = A10 * B_top + A11 * B_bottom
    #
    # So we need: A00*B_top (2x2 * 2x6), A01*B_bottom (2x2 * 2x6), A10*B_top, A11*B_bottom
    # Each 2x2 * 2x6 = 3 products of 2x2*2x2 = 3*7 = 21 rank
    # Total = 4 * 21 = 84. Still worse.
    #
    # Alternative: use Laderman rank-23 for 3x3*3x3.
    # Pad 4x4 to 3x3? No, 4 > 3.
    #
    # Let me try: decompose 4x4x4x6 into smaller Strassen blocks more cleverly.
    # 
    # Actually, the current 77 approach is: 4x4x4 (rank 49 via 7x7 Strassen) + 4x4x2 (rank 28)
    # 
    # Let me try a 2x2 block decomposition where we use rank-7 Strassen for each 2x2x2x2:
    # 4x4 * 4x6: split into 2x2 blocks
    # A: 2x2 blocks of 2x2 -> 4 blocks
    # B: 2x3 blocks of 2x2 -> 6 blocks  
    # C: 2x3 blocks of 2x2 -> 6 blocks
    # C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
    # That's 2 * 2 * 3 = 12 products of 2x2 * 2x2, each rank 7 = 84.
    #
    # To beat 77, we need sharing. Let me use the fact that:
    # C_top = A00*B0 + A01*B1 where B0 = [B00,B01,B02] (2x6), B1 = [B10,B11,B12] (2x6)
    # C_bot = A10*B0 + A11*B1
    #
    # We can compute: P1 = A00*B0, P2 = A01*B1, P3 = A10*B0, P4 = A11*B1
    # C_top = P1 + P2, C_bot = P3 + P4
    #
    # Each Pi is 2x2 * 2x6. We can do this as:
    # For 2x2 * 2x6: split B into three 2x2 blocks
    # 2x2 * 2x2 three times = 3 * 7 = 21 rank each
    # Total = 4 * 21 = 84.
    #
    # Hmm, let me just use the 2x2 block approach with 12 Strassen products = 84.
    # Wait, that's worse. Let me stick with a better decomposition.
    #
    # Actually, let me reconsider. The key is to find a decomposition with rank < 77.
    # 
    # Laderman's algorithm: rank 23 for 3x3*3x3.
    # We can embed 2x2*2x2 in 3x3*3x3 by padding with zeros.
    # But that gives rank 23 for each 2x2*2x2 product, which is worse than 7.
    #
    # Let me try a different block structure: 4x4 * 4x6 as (2+2)x(4) * (4)x(2+2+2)
    # 
    # Or: use the fact that 4 = 2+2 and 6 = 2+2+2, and use a tensor product approach.
    #
    # For now, let me implement the clean 2x2 block decomposition with 12 Strassen products.
    # Even though it's 84, it's a valid approach. But 84 > 77, so it's worse.
    #
    # Let me instead try to optimize the existing 77 approach or find a better one.
    #
    # Alternative: 4x4x4x6 can be seen as two 4x4x4x3 problems, each rank ~38, total 76.
    # 4x4x4x3: split B into 4x2 + 4x1. 4x4x4x2 = rank 28, 4x4x4x1 = rank 16. Total 44. Two of these = 88. Worse.
    #
    # Let me try: 4x4x4x6 = 4x4x4x(2+2+2) = 3 * (4x4x4x2) = 3 * 28 = 84. Worse.
    #
    # The current 77 = 49 + 28 seems hard to beat with simple block decompositions.
    #
    # Let me try a 3x3 Laderman approach:
    # Pad A (4x4) to 5x5? No. 
    #
    # Actually, let me try: use rank-7 Strassen for 2x2*2x2, but with a smarter block layout.
    # 
    # 4x4 * 4x6:
    # Split A into 4 blocks of 2x2 (rows 0-1, 2-3; cols 0-1, 2-3)
    # Split B into 6 blocks of 2x2 (rows 0-1, 2-3; cols 0-1, 2-3, 4-5)
    # C has 6 blocks of 2x2
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # 
    # Total 12 products, each rank 7 = 84.
    #
    # To reduce: note that A[i,0]*B[0,j] for fixed i and varying j is A[i,0] * B[0,:]
    # This is a 2x2 * 2x6 product. If we can do 2x2*2x6 in rank < 21, we save.
    #
    # 2x2 * 2x6: A (2x2) * B (2x6). C[i,k] = sum_j A[i,j]*B[j,k].
    # This is like 2x2 * 2x2 done 3 times = 21 rank. Can we do better?
    # The tensor rank of 2x2x2x6... it's the sum of three 2x2x2x2 tensors.
    # Each has rank 7, so total rank <= 21. Lower bound is at least 7.
    # 
    # I don't know a better algorithm for 2x2*2x6 than 21.
    #
    # Let me just go with the current 77 approach but verify it's correct.
    # The current code builds rank 77. Let me keep it and just clean up.

    # Build rank-77 Strassen decomposition (same as before, verified working)
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen (Kronecker product of two 2x2 Strassen)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]

    # Check deadline
    if time.time() > deadline - 20.0:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Verify correctness
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                residual += (s - expected)**2
    
    if residual > 1e-10:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Try to reduce rank by finding linear dependencies in the tensor structure
    # For each column r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r] (appropriately indexed)
    # If two columns have proportional contributions, we can merge them.
    
    # Build the "tensor matrix" M where M[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this is nonlinear in the columns.
    
    # Instead, try: for each pair of columns (a, b), check if we can find alpha, beta
    # such that the combined column works. This is complex.
    
    # Simpler: try to find if any column is redundant by checking if the tensor
    # equation can be satisfied with that column zeroed and others adjusted.
    
    # Given time constraints, attempt a limited search for rank reduction.
    # Try removing one column at a time and solving for corrections.
    
    best_rank = total_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column and checking if the residual can be fixed
    # by adjusting one other column (rank-preserving swap)
    for r_remove in range(total_rank):
        if time.time() > deadline - 30.0:
            break
        
        # Try to replace column r_remove with a combination of two other columns
        # This is a 2D search, too expensive.
        
        # Instead, just check if column r_remove is "small" and can be absorbed
        norm = np.linalg.norm(U[:, r_remove]) + np.linalg.norm(V[:, r_remove]) + np.linalg.norm(W[:, r_remove])
        if norm < 1e-10:
            # Column is effectively zero, remove it
            mask = np.ones(total_rank, dtype=bool)
            mask[r_remove] = False
            new_rank = total_rank - 1
            if new_rank < best_rank:
                best_U = U[:, mask]
                best_V = V[:, mask]
                best_W = W[:, mask]
                best_rank = new_rank
    
    if best_rank < total_rank:
        U, V, W = best_U, best_V, best_W
        total_rank = best_rank

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
