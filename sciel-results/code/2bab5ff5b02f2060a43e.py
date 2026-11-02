import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    # Try to construct a lower rank solution using Strassen blocks
    # Rank 84 solution:
    # 4 A-blocks (2x2) * 3 B-blocks (2x2) * 7 Strassen terms = 84
    
    try:
        U_new = [[0] * 84 for _ in range(n * m)]
        V_new = [[0] * 84 for _ in range(m * p)]
        W_new = [[0] * 84 for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # M1..M7
        # X indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Y indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Z indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # U (X) contributions for M1..M7
        U_strassen = [
            [1, 0, 0, 1], # M1: x00 + x11
            [0, 0, 1, 1], # M2: x10 + x11
            [1, 0, 0, 0], # M3: x00
            [0, 0, 0, 1], # M4: x11
            [1, 1, 0, 0], # M5: x00 + x01
            [1, 0, -1, 0],# M6: x10 - x00
            [0, 1, 0, -1] # M7: x01 - x11
        ]
        
        # V (Y) contributions for M1..M7
        V_strassen = [
            [1, 0, 0, 1], # M1: y00 + y11
            [1, 0, 0, 0], # M2: y00
            [0, 0, 1, -1],# M3: y10 - y11
            [1, -1, 0, 0],# M4: y01 - y00
            [0, 0, 0, 1], # M5: y11
            [1, 0, 1, 0], # M6: y00 + y10
            [0, 1, 0, 1]  # M7: y11 + y01
        ]
        
        # W (Z) reconstruction coefficients for M1..M7
        # z00 = M1 + M4 - M5 + M7
        # z01 = M3 + M5
        # z10 = M2 + M4
        # z11 = M1 - M2 + M3 + M6
        W_strassen = [
            [1, 0, 0, 1],  # M1 coeffs for z00, z01, z10, z11
            [0, 0, 1, -1], # M2
            [0, 1, 0, 1],  # M3
            [1, 0, 1, 0],  # M4
            [-1, 1, 0, 0], # M5
            [0, 0, 0, 1],  # M6
            [1, 0, 0, 0]   # M7
        ]
        
        col = 0
        # Iterate over A blocks (2x2 grid)
        for ai in range(2):
            for aj in range(2):
                # Iterate over B blocks (3 chunks of 2x2)
                for bk in range(3):
                    # Iterate over Strassen terms
                    for s in range(7):
                        # A block indices
                        # A is 4x4. Block (ai, aj) is rows ai*2..ai*2+1, cols aj*2..aj*2+1
                        # Local indices 0..3 map to global
                        # 0: (ai*2, aj*2), 1: (ai*2, aj*2+1), 2: (ai*2+1, aj*2), 3: (ai*2+1, aj*2+1)
                        
                        for local_idx in range(4):
                            val = U_strassen[s][local_idx]
                            if val != 0:
                                # Global U index: row = ai*2 + (local_idx // 2)? No.
                                # U index is i*m + j.
                                # i = ai*2 + (0 if local_idx < 2 else 1)
                                # j = aj*2 + (0 if local_idx % 2 == 0 else 1)
                                i = ai * 2 + (local_idx // 2)
                                j = aj * 2 + (local_idx % 2)
                                U_new[i * m + j][col] += val
                                
                        # B block indices
                        # B is 4x6. Block (bk) corresponds to cols bk*2..bk*2+1.
                        # Rows are split into 2 blocks of 2?
                        # Wait, the Strassen algorithm multiplies X (2x2) by Y (2x2).
                        # Here X is A_block (2x2).
                        # Y is B_block?
                        # B_block needs to be 2x2.
                        # B is 4x6.
                        # We split B into rows 0-1 and 2-3?
                        # The product C = A B.
                        # C_block (ai, bk) = A_block (ai, 0) * B_block (0, bk) + A_block (ai, 1) * B_block (1, bk).
                        # This is not a single 2x2 * 2x2.
                        # It is a sum of two 2x2 * 2x2 products.
                        # My previous logic was:
                        # For each A_block (ai, aj), we multiply by the corresponding B_strip (rows aj*2..aj*2+1, cols bk*2..bk*2+1).
                        # Wait, A is 4x4. B is 4x6.
                        # C[i,k] = sum_j A[i,j] B[j,k].
                        # If we block A into 2x2, and B into 2x2 (vertically) and 2x2 (horizontally)?
                        # B is 4x6. Vertically 2 blocks of 2 rows. Horizontally 3 blocks of 2 cols.
                        # So B has 6 blocks of 2x2.
                        # A has 4 blocks of 2x2.
                        # C has 6 blocks of 2x2.
                        # C_block (ai, bk) = sum_{aj=0,1} A_block (ai, aj) * B_block (aj, bk).
                        # This is a sum of two 2x2 * 2x2 products.
                        # We can compute each 2x2 * 2x2 product using Strassen (rank 7).
                        # So for each (ai, bk), we need 2 * 7 = 14 terms.
                        # Total terms: 6 C-blocks * 14 = 84 terms.
                        
                        # So we iterate over:
                        # ai (0..1), bk (0..2), aj (0..1), s (0..6)
                        
                        # Reset col logic, we are inside the loop
                        # The loop structure above was:
                        # for ai, aj, bk, s.
                        # This iterates 2*2*3*7 = 84 times.
                        # But the product is C(ai, bk) = A(ai, 0)B(0, bk) + A(ai, 1)B(1, bk).
                        # So for a fixed (ai, bk), we need to process aj=0 and aj=1.
                        # The term for aj=0 uses A_block(ai, 0) and B_block(0, bk).
                        # The term for aj=1 uses A_block(ai, 1) and B_block(1, bk).
                        
                        # In the loop:
                        # ai, aj, bk, s.
                        # U uses A_block(ai, aj).
                        # V uses B_block(aj, bk).
                        # W uses C_block(ai, bk).
                        
                        # Let's re-verify the V index.
                        # B_block(aj, bk) is rows aj*2..aj*2+1, cols bk*2..bk*2+1.
                        # Local Y indices 0..3 map to:
                        # 0: (aj*2, bk*2), 1: (aj*2, bk*2+1), 2: (aj*2+1, bk*2), 3: (aj*2+1, bk*2+1)
                        
                        for local_idx in range(4):
                            val = V_strassen[s][local_idx]
                            if val != 0:
                                # Global V index: row * p + col
                                # row = aj * 2 + (local_idx // 2)
                                # col = bk * 2 + (local_idx % 2)
                                r = aj * 2 + (local_idx // 2)
                                c = bk * 2 + (local_idx % 2)
                                V_new[r * p + c][col] += val
                                
                        # W index
                        # C_block(ai, bk) is rows ai*2..ai*2+1, cols bk*2..bk*2+1.
                        # W index is k * n + i.
                        # k is the col of C, i is the row of C.
                        # Local Z indices 0..3 map to:
                        # 0: (ai*2, bk*2), 1: (ai*2, bk*2+1), 2: (ai*2+1, bk*2), 3: (ai*2+1, bk*2+1)
                        # Wait, W_strassen[s] gives coefficients for z00, z01, z10, z11.
                        # z00 corresponds to C[ai*2, bk*2].
                        # W index: k * n + i.
                        # k = bk*2 + (local_idx % 2)
                        # i = ai*2 + (local_idx // 2)
                        
                        for local_idx in range(4):
                            val = W_strassen[s][local_idx]
                            if val != 0:
                                k = bk * 2 + (local_idx % 2)
                                i = ai * 2 + (local_idx // 2)
                                W_new[k * n + i][col] += val
                                
                        col += 1

        # Verify the new solution
        U_arr = np.array(U_new, dtype=float)
        V_arr = np.array(V_new, dtype=float)
        W_arr = np.array(W_new, dtype=float)
        
        U_3d = U_arr.reshape(n, m, 84)
        V_3d = V_arr.reshape(m, p, 84)
        W_3d = W_arr.reshape(p, n, 84)
        
        result = np.zeros((n, m, p))
        for c in range(84):
            result += np.einsum('ij,jk,ki->ijk', U_3d[:,:,c], V_3d[:,:,c], W_3d[:,:,c])
            
        if np.allclose(result, np.ones((n,m,p)), atol=1e-9):
            best_rank = 84
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            
    except Exception as e:
        pass

    return best_result
