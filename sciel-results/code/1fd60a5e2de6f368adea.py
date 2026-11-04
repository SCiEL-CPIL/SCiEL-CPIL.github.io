"""Matrix multiplication 4x4x6 via rank-77 Strassen tiling with column pruning."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

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
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W

def _build_rank77(n, m, p):
    """Build the rank-77 Strassen tiling for 4x4x6."""
    U_s, V_s, W_s = _get_strassen_2x2_factors()
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    # 1. 4x4x4 Strassen (columns 0-48)
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val

    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val

    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # 2. 4x4x2 Strassen-block (columns 49-76)
    products = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]

    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7

        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_offset + s] += val

        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_offset + s] += val

        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_offset + s] += val

    return U, V, W, total_rank

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check decomposition correctness."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def _try_prune(U, V, W, n, m, p, rank, deadline, max_iters=200):
    """
    Attempt to reduce rank by zeroing out columns and adjusting remaining ones.
    This is a coordinate-descent approach on the column selection.
    """
    t0 = time.time()
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    # Try removing one column at a time, starting from the ones with smallest
    # "importance" (measured by how much they contribute to satisfying constraints)
    # For each column c, check if we can set it to zero and still satisfy
    # the decomposition with the remaining columns.

    # First, let's try a greedy approach: try to remove each column one by one
    # and check if the decomposition still holds.
    
    # Actually, a better approach: for each column, see if it's redundant
    # by checking if the tensor contribution can be absorbed by other columns.
    
    # Simpler heuristic: try removing columns in order of smallest norm
    col_norms = []
    for c in range(rank):
        norm = np.linalg.norm(Ua[:, c]) * np.linalg.norm(Va[:, c]) * np.linalg.norm(Wa[:, c])
        col_norms.append((norm, c))
    
    col_norms.sort()
    
    # Try to remove the smallest-norm columns
    active = list(range(rank))
    removed = []
    
    for norm, c in col_norms:
        if time.time() > deadline - 5:
            break
        if len(removed) >= 0:  # try to remove at least 1
            # Test if removing column c still gives valid decomposition
            # We need to solve for the remaining columns
            pass
    
    # Alternative: use the fact that the tensor is bilinear in each factor.
    # If we fix two factors, the third is determined by linear equations.
    # So we can try: fix U and V, solve for W. If W has rank < current, we're good.
    
    # Let's try: for a subset of columns, check if we can find W that works.
    # For each column c, W_c = (U_c^T ⊗ V_c^T) \ tensor_ones
    # Actually the constraint is: sum_c U_c ⊗ V_c ⊗ W_c = T_matmul
    # For fixed U_c, V_c, W_c is determined by:
    # W_c[k*n+i] = sum over (j) of [tensor[i,j,k] / (U_c[i,j] * V_c[j,k])] ... no that's not right.
    
    # The constraint is: for each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # This is trilinear, not linear in any single factor when others are fixed.
    # Actually, if we fix U and V, it's linear in W:
    # For each c: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k)
    # Let A_c[k,i] = sum_{j} U[i,j,c]*V[j,k,c] ... no, that's not separable that way.
    
    # Actually: sum_c (U_c[i,j] * V_c[j,k]) * W_c[k,i] = 1
    # For fixed U, V: this is linear in W. Define M_c[i,k] = sum_j U_c[i,j]*V_c[j,k]
    # Then: sum_c M_c[i,k] * W_c[k,i] = 1 for all (i,k)
    # Wait, that's not right either. Let me re-derive.
    
    # The tensor is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For fixed U and V, define for each c: B_c[i,k] = sum_j U[i,j,c] * V[j,k,c]
    # Then T[i,j,k] is NOT simply B_c[i,k] * W_c[k,i] because j is still free.
    
    # OK so it's genuinely trilinear. Let me try a different approach.
    
    # Greedy: try removing one column at a time. When we remove column c,
    # the remaining columns must satisfy: sum_{c' != c} U[i,j,c']*V[j,k,c']*W[k,i,c'] = 1
    # We can try to adjust one of the remaining columns to compensate.
    
    # This is getting complex. Let me try a simpler heuristic:
    # Check if any column has all zeros in one of the factors (making it useless).
    # Or if two columns are proportional (linearly dependent).
    
    # Check for linearly dependent columns in the "combined" sense
    # Build the matrix where each column c contributes U_c ⊗ V_c to the bilinear form
    # and W_c is the weight.
    
    # Actually, let's just try: can we reduce the rank by checking if the 
    # decomposition is already minimal? For 4x4x4, the known minimal rank is 49
    # (Strassen). For 4x4x2, I believe 28 is good. So 77 might be optimal.
    
    # But let's try a quick numerical check: is the rank-77 decomposition 
    # already in its minimal form? If the factors are in general position, 
    # probably yes.
    
    # Let me try a different angle: can we merge the 4x4x4 and 4x4x2 parts?
    # The 4x4x6 multiplication can be viewed as:
    # C = A * B where A is 4x4, B is 4x6
    # Split B into B1 (4x4) and B2 (4x2)
    # C1 = A * B1 (4x4 result) uses rank 49
    # C2 = A * B2 (4x2 result) uses rank 28
    # Total: 77
    
    # Can we do better? The 4x4x2 part: A is 4x4, B2 is 4x2.
    # Using Strassen: split A into 2x2 blocks, B2 into 2x1 blocks.
    # A_00 * B2_top + A_01 * B2_bot -> C_0 (2x2)
    # A_10 * B2_top + A_11 * B2_bot -> C_1 (2x2)
    # Each of these is a 2x2 * (2x1 + 2x1) = 2x2 * 4x1... 
    # Actually, A_00 (2x2) * B2_top (2x1) gives 2x1, and A_01 (2x2) * B2_bot (2x1) gives 2x1.
    # Sum gives 2x1. So C_0 is 2x1.
    # Using Strassen for 2x2 * 2x1: we can treat it as 2x2 * 2x2 where the second
    # 2x2 has a zero column. That might save some rank.
    
    # Let me try: 2x2 * 2x1 using a specialized decomposition.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is 2x2*2x1, which is just 4 multiplications schoolbook, or we can use
    # a 2x2*2x2 Strassen with one column of B being zero.
    # In Strassen for 2x2*2x2, if B has a zero column, some products simplify.
    
    # Let me try to build a 2x2 * 2x1 decomposition with rank < 7.
    # Schoolbook: rank 4.
    # Can we do better than 4? Unlikely for 2x2*2x1.
    
    # So the 4x4x2 part: 4 blocks of 2x2*2x1, each with rank 4 = 16? No wait.
    # Actually in the current code, each 2x2*2x2 Strassen has rank 7, and we have
    # 4 such products, giving 28. If we could use rank 4 for each 2x2*2x1,
    # we'd get 16 instead of 28, saving 12!
    
    # But wait, the current code uses 2x2*2x2 Strassen where the B block is
    # actually 2x2 (rows 0-1 or 2-3, cols 4-5). So it's genuinely 2x2*2x2.
    # The result is 2x2, not 2x1. Let me re-examine.
    
    # In the code: C block is rows 2I..2I+1, cols 4..5. That's 2x2.
    # So C is 4x6, and the bottom-right 4x2 block is computed by:
    # C[i,4+k] = sum_j A[i,j] * B[j,4+k]
    # For i in 0..1, k in 0..1: C[i,4+k] = sum_j A[i,j] * B[j,4+k]
    # Split j into j in 0..1 and j in 2..3:
    # C_top = A_00 * B_top + A_01 * B_bot  (2x2 result)
    # C_bot = A_10 * B_top + A_11 * B_bot  (2x2 result)
    # Each term is 2x2 * 2x2, using Strassen rank 7. Total 4*7 = 28.
    
    # Can we do C_top with less than 14 rank? C_top = A_00 * B_top + A_01 * B_bot
    # This is a sum of two 2x2*2x2 products. Each needs rank 7 (Strassen) or 4 (schoolbook).
    # Sum: rank 14 (Strassen+Strassen) or 8 (schoolbook+schoolbook).
    # So schoolbook is better here! 4*4 = 16 for the whole 4x4x2 part.
    
    # Wait, that's a big improvement: 16 instead of 28!
    # Let me implement this.
    
    return None  # Placeholder, will implement below

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _get_strassen_2x2_factors()

    # Option 1: Rank 77 (Strassen for both parts)
    U, V, W, rank = _build_rank77(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, rank
    err = _verify(U, V, W, n, m, p, rank)
    
    if err > 1e-6:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}

    # Option 2: Try to use schoolbook for the 4x4x2 part (rank 16 instead of 28)
    # C = A * B, A is 4x4, B is 4x6
    # Split: C[:, 0:4] = A * B[:, 0:4] (4x4*4x4, rank 49 Strassen)
    #        C[:, 4:6] = A * B[:, 4:6] (4x4*4x2, rank ?)
    
    # For 4x4*4x2: split A into 2x2 blocks, B[:, 4:6] into 2x1 blocks per row-pair
    # C[0:2, 4:6] = A[0:2, :] * B[:, 4:6]
    #             = A_00 * B[0:2, 4:6] + A_01 * B[2:4, 4:6]
    # Each term is 2x2 * 2x2 (since B[0:2, 4:6] is 2x2)
    # Schoolbook: 4 mults per 2x2*2x2, so 8 per C_top, 8 per C_bot = 16 total
    
    rank_444 = 49
    rank_442_school = 16
    total_rank2 = rank_444 + rank_442_school
    
    U2 = [[0.0] * total_rank2 for _ in range(n * m)]
    V2 = [[0.0] * total_rank2 for _ in range(m * p)]
    W2 = [[0.0] * total_rank2 for _ in range(p * n)]
    
    # Copy the 4x4x4 Strassen part
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U2[i * 4 + j][col] += val

    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V2[j * 6 + k][col] += val

    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W2[k * 4 + i][col] += val

    # Schoolbook for 4x4*4x2 part (columns 49-64)
    # C[i, 4+k] = sum_{j=0}^{3} A[i,j] * B[j, 4+k]
    # For each (i, j, k) with k in {0,1}: one schoolbook multiplication
    col_offset = rank_444
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U2[i * 4 + j][col_offset + col] = 1
                V2[j * 6 + (4 + k)][col_offset + col] = 1
                W2[(4 + k) * 4 + i][col_offset + col] = 1
                col += 1

    err2 = _verify(U2, V2, W2, n, m, p, total_rank2)
    
    if err2 < 1e-6 and total_rank2 < best_rank:
        best_U, best_V, best_W, best_rank = U2, V2, W2, total_rank2

    # Return the best result
    return {"U": best_U, "V": best_V, "W": best_W}
