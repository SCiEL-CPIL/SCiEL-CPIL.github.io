"""Matrix multiplication 2x4x8: rank-56 Strassen-based decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56  # 8 Strassen instances * 7 terms each
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y01+y11)
    #
    # Output:
    # z00 = M1 - M4 + M5 + M7
    # z01 = M2 + M4
    # z10 = M3 + M5
    # z11 = M1 - M2 + M3 + M6
    
    # Define the 7 Strassen terms as (U_factors, V_factors, W_factors)
    # Each factor is a list of (local_index, coefficient)
    # U local indices: (i, j) where i in {0,1}, j in {0,1} -> local index i*2+j
    # V local indices: (j, k) where j in {0,1}, k in {0,1} -> local index j*2+k
    # W local indices: (k, i) where k in {0,1}, i in {0,1} -> local index k*2+i
    
    # Term definitions: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (local_u_idx, coeff)
    # V_coeffs: list of (local_v_idx, coeff)
    # W_coeffs: list of (local_w_idx, coeff)
    
    strassen_terms = [
        # M1 = (x00+x11)(y00+y11)
        {
            'U': [(0, 1), (3, 1)],   # x00 + x11
            'V': [(0, 1), (3, 1)],   # y00 + y11
            'W': [(0, 1), (1, -1), (2, 1), (3, 1)]  # z00 - z01 + z10 + z11
        },
        # M2 = (x10+x11)y00
        {
            'U': [(2, 1), (3, 1)],   # x10 + x11
            'V': [(0, 1)],           # y00
            'W': [(1, 1), (3, -1)]   # z01 - z11
        },
        # M3 = x00(y10-y11)
        {
            'U': [(0, 1)],           # x00
            'V': [(2, 1), (3, -1)],  # y10 - y11
            'W': [(2, 1), (3, 1)]    # z10 + z11
        },
        # M4 = x11(y01-y00)
        {
            'U': [(3, 1)],           # x11
            'V': [(1, 1), (0, -1)],  # y01 - y00
            'W': [(0, -1), (1, 1)]   # -z00 + z01
        },
        # M5 = (x00+x01)y11
        {
            'U': [(0, 1), (1, 1)],   # x00 + x01
            'V': [(3, 1)],           # y11
            'W': [(0, 1), (2, 1)]    # z00 + z10
        },
        # M6 = (x10-x00)(y00+y10)
        {
            'U': [(2, 1), (0, -1)],  # x10 - x00
            'V': [(0, 1), (2, 1)],   # y00 + y10
            'W': [(3, 1)]            # z11
        },
        # M7 = (x01-x11)(y01+y11)
        {
            'U': [(1, 1), (3, -1)],  # x01 - x11
            'V': [(1, 1), (3, 1)],   # y01 + y11
            'W': [(0, 1)]            # z00
        },
    ]
    
    # For the 2x4x8 problem:
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x8: rows j=0,1,2,3; cols k=0,1,...,7
    # C is 2x8: rows i=0,1; cols k=0,1,...,7
    #
    # We can decompose this into 4 independent 2x2x4 problems or 
    # more naturally, we group columns of B.
    #
    # Actually, let's think differently. The standard way to apply Strassen
    # to rectangular matrices:
    #
    # A (2x4) * B (4x8) = C (2x8)
    #
    # Split A into two 2x2 blocks: A = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
    # Split B into two 4x4 blocks: B = [B0; B1] where B0 = B[0:2, :], B1 = B[2:4, :]
    #
    # C = A0 * B0 + A1 * B1
    #
    # Now A0*B0 is 2x2 * 4x8 = 2x8, and A1*B1 is 2x2 * 4x8 = 2x8
    #
    # For each 2x2 * 4x8 multiplication, we can further split the 4x8 matrix
    # into two 2x4 matrices and use Strassen on 2x2 * 2x4.
    #
    # 2x2 * 2x4: Split the 2x4 into two 2x2 blocks.
    #   X (2x2) * [Y0 | Y1] (2x4) = [X*Y0 | X*Y1] (2x4)
    #   This requires two Strassen 2x2*2x2 multiplications.
    #
    # So total: 2 (from A0/A1) * 2 (from splitting 4x8 into two 2x4) * 2 (from splitting 2x4 into two 2x2) = 8 Strassen instances.
    #
    # Let me be more precise:
    # A (2x4) = [A_col0 | A_col1] where A_col0 = A[:, 0:2], A_col1 = A[:, 2:4]
    # B (4x8) = [B_row0; B_row1] where B_row0 = B[0:2, :], B_row1 = B[2:4, :]
    #
    # C = A_col0 * B_row0 + A_col1 * B_row1
    #
    # Now A_col0 is 2x2, B_row0 is 2x8.
    # Split B_row0 (2x8) into four 2x2 blocks: B_row0[:, 0:2], B_row0[:, 2:4], B_row0[:, 4:6], B_row0[:, 6:8]
    #
    # A_col0 * B_row0 = A_col0 * [B00 | B01 | B02 | B03] = [A_col0*B00 | A_col0*B01 | A_col0*B02 | A_col0*B03]
    #
    # Each A_col0*B0k is 2x2 * 2x2, using Strassen. That's 4 Strassen instances for A_col0 * B_row0.
    #
    # Similarly, A_col1 * B_row1 gives another 4 Strassen instances.
    #
    # Total: 8 Strassen instances.
    
    # Global index mapping:
    # U row index: i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # V row index: j*p + k = j*8 + k, where j in {0,1,2,3}, k in {0,...,7}
    # W row index: k*n + i = k*2 + i, where k in {0,...,7}, i in {0,1}
    
    # For each Strassen instance, we need to map local indices to global indices.
    # The 8 instances correspond to:
    # Instance (a, b, c) where:
    #   a in {0, 1}: which column block of A (A_col_a)
    #   b in {0, 1}: which row block of B (B_row_b)
    #   c in {0, 1, 2, 3}: which 2-column block of the 8-column output
    
    # For instance (a, b, c):
    # X = A[:, a*2 : a*2+2]  (2x2)
    #   Local index (i, j_local) -> global (i, a*2 + j_local)
    # Y = B[b*2 : b*2+2, c*2 : c*2+2]  (2x2)
    #   Local index (j_local, k_local) -> global (b*2 + j_local, c*2 + k_local)
    # Z = C[:, c*2 : c*2+2]  (2x2)
    #   Local index (k_local, i) -> global (c*2 + k_local, i)
    
    col = 0
    for a in range(2):      # A column block
        for b in range(2):  # B row block
            for c in range(4):  # output column block
                for term in strassen_terms:
                    # Update U
                    for local_u_idx, coeff in term['U']:
                        i = local_u_idx // 2
                        j_local = local_u_idx % 2
                        global_u = i * m + (a * 2 + j_local)
                        U[global_u][col] += coeff
                    
                    # Update V
                    for local_v_idx, coeff in term['V']:
                        j_local = local_v_idx // 2
                        k_local = local_v_idx % 2
                        global_v = (b * 2 + j_local) * p + (c * 2 + k_local)
                        V[global_v][col] += coeff
                    
                    # Update W
                    for local_w_idx, coeff in term['W']:
                        k_local = local_w_idx // 2
                        i = local_w_idx % 2
                        global_w = (c * 2 + k_local) * n + i
                        W[global_w][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
