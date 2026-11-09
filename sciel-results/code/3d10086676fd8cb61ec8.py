import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """
    Returns Strassen decomposition for 2x2 x 2x2 -> 2x2.
    U: 4 x 7, V: 4 x 7, W: 4 x 7
    Convention: W[k*n+i, r] where n=2
    """
    # 7 multiplication factors for Strassen
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B21-B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U rows: A11(row0), A12(row1), A21(row2), A22(row3)
    # V rows: B11(row0), B12(row1), B21(row2), B22(row3)
    # W rows: C11(row0), C12(row1), C21(row2), C22(row3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)*(B11+B22)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    W[1, 0] += 0; W[2, 0] += 0
    
    # M2: (A21+A22)*B11
    U[1, 1] += 1; U[2, 1] += 1  # A21 is index 2? Wait.
    # Let's define indices clearly:
    # A = [[A11, A12], [A21, A22]]
    # Flat A: [A11, A12, A21, A22] -> indices 0,1,2,3
    # B = [[B11, B12], [B21, B22]]
    # Flat B: [B11, B12, B21, B22] -> indices 0,1,2,3
    # C = [[C11, C12], [C21, C22]]
    # Flat C: [C11, C12, C21, C22] -> indices 0,1,2,3
    
    # Reset
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = 0; W[2, 0] = 0
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B11-B21)
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    W[0, 2] = 1
    
    # M4 = A22(B21-B22)
    U[3, 3] = 1
    V[2, 3] = 1; V[3, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1
    
    return U, V, W


def _decompose_2x4x8():
    """
    Decompose 2x4 x 4x8 -> 2x8.
    Split columns of A (4) into two groups of 2: A = [A_L, A_R] where A_L, A_R are 2x2.
    Split rows of B (4) into two groups of 2: B = [B_T; B_B] where B_T, B_B are 2x2.
    C = A_L * B_T + A_R * B_B.
    Each term is a 2x2 * 2x2 multiplication, which can be done with Strassen rank 7.
    Total rank = 7 + 7 = 14.
    
    U shape: (2*4=8) x 14
    V shape: (4*8=32) x 14
    W shape: (8*2=16) x 14
    
    A flat indices: 0..7. Rows 0,1 (i=0,1) x Cols 0,1 (j=0,1) -> A_L.
    A flat indices: Rows 0,1 x Cols 2,3 (j=2,3) -> A_R.
    So A_L indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    A_R indices: (0,2)->2? No.
    A is 2x4. Flat index = i*4 + j.
    i=0, j=0 -> 0
    i=0, j=1 -> 1
    i=0, j=2 -> 2
    i=0, j=3 -> 3
    i=1, j=0 -> 4
    i=1, j=1 -> 5
    i=1, j=2 -> 6
    i=1, j=3 -> 7
    
    A_L (cols 0,1): indices 0,1,4,5.
    A_R (cols 2,3): indices 2,3,6,7.
    
    B is 4x8. Flat index = j*8 + k.
    Rows 0,1 (j=0,1): indices 0..15.
    Rows 2,3 (j=2,3): indices 16..31.
    B_T (rows 0,1): indices 0..15.
    B_B (rows 2,3): indices 16..31.
    
    C is 2x8. Flat index = k*2 + i.
    k=0..7, i=0,1.
    C_L (from A_L*B_T): 2x2 result.
    C_R (from A_R*B_B): 2x2 result.
    Wait, A_L is 2x2, B_T is 2x2. Result is 2x2.
    Which part of C does it correspond to?
    C = A * B.
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    If we split j into {0,1} and {2,3}, and k into... wait.
    B is 4x8. The split of B rows corresponds to j indices.
    A_L * B_T computes sum_{j in {0,1}} A_{i,j} B_{j,k}. This contributes to ALL k (0..7).
    But A_L is only 2x2. B_T is 2x2. The product is 2x2.
    This only covers k in {0,1} if B_T is 2x2?
    No, B_T is 2x8 if we don't split columns of B.
    The problem is 2x4 * 4x8.
    If we use Strassen for 2x2 * 2x2, we need the inner dimension to be 2.
    Here inner dimension is 4.
    We can split the inner dimension 4 into two 2s.
    C = A_{:, 0:2} B[0:2, :] + A_{:, 2:4} B[2:4, :].
    First term: (2x2) * (2x8). This is not 2x2 * 2x2.
    We can further split the 8 columns of B into four 2s?
    Then we have 4 multiplications of (2x2) * (2x2).
    Each has rank 7. Total rank 28.
    Or split 8 columns into two 4s?
    (2x2) * (2x4). Rank of 2x2 * 2x4 is 8 (schoolbook) or 7?
    Actually, 2x2 * 2x4 is equivalent to two 2x2 * 2x2 multiplications (for the two column blocks).
    So rank 7+7=14.
    Total for 2x4 * 4x8:
    Split inner dim 4 into 2+2.
    Term 1: A_L (2x2) * B_T (2x8).
    Split B_T columns into 4+4? Or 2*4 blocks?
    B_T is 2x8. We can view it as two 2x4 blocks.
    A_L * B_T = A_L * [B_T1, B_T2] = [A_L*B_T1, A_L*B_T2].
    A_L*B_T1 is 2x2 * 2x4. Rank 14.
    A_L*B_T2 is 2x2 * 2x4. Rank 14.
    Total for Term 1: 28.
    Term 2: A_R (2x2) * B_B (2x8). Similarly 28.
    Total rank: 56.
    
    This matches the strategy description.
    """
    U_str, V_str, W_str = _strassen_2x2()
    
    # We need to construct U, V, W for the whole 2x4x8 problem.
    # U: 8 x 56
    # V: 32 x 56
    # W: 16 x 56
    
    U_total = np.zeros((8, 56))
    V_total = np.zeros((32, 56))
    W_total = np.zeros((16, 56))
    
    # Term 1: A_L * B_T
    # A_L indices in U: 0,1,4,5.
    # B_T indices in V: 0..15.
    # C indices in W: 0..15.
    
    # Term 1 splits into two 2x2*2x4 mults?
    # Actually, simpler:
    # A_L (2x2) * B_T (2x8).
    # We can decompose B_T into 4 blocks of 2x2: B_T = [B_T1, B_T2, B_T3, B_T4].
    # A_L * B_T = [A_L*B_T1, A_L*B_T2, A_L*B_T3, A_L*B_T4].
    # Each A_L*B_Tj is 2x2 * 2x2 -> Rank 7.
    # Total 4 * 7 = 28.
    
    # Similarly for Term 2: A_R * B_B.
    # A_R indices: 2,3,6,7.
    # B_B indices: 16..31.
    # B_B = [B_B1, B_B2, B_B3, B_B4].
    # Total 28.
    
    # Total 56.
    
    # Let's map the indices.
    
    def add_block(U_dest, V_dest, W_dest, U_src, V_src, W_src, u_offset, v_offset, w_offset):
        """
        Add a 2x2*2x2 Strassen block.
        U_src, V_src, W_src are 4x7.
        u_offset: start row in U_dest for the 4 A elements.
        v_offset: start row in V_dest for the 4 B elements.
        w_offset: start row in W_dest for the 4 C elements.
        """
        # Map U_src rows (0..3) to U_dest rows (u_offset..u_offset+3)
        # But U_src corresponds to A_flat [A11, A12, A21, A22].
        # We need to map these to the correct global U indices.
        
        pass # We will do it manually for clarity
        
    # Term 1, Block 1: A_L * B_T1
    # A_L global indices: 0,1,4,5.
    # A_L local indices: 0->0, 1->1, 2->4, 3->5.
    # B_T1 global indices: 0,1,8,9. (Rows 0,1 of B_T, Cols 0,1 of B_T1).
    # B_T1 is 2x2. B_T rows 0,1. Cols 0,1.
    # B flat index = j*8 + k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 8
    # j=1, k=1 -> 9
    # Local B_T1 indices: 0->0, 1->1, 2->8, 3->9.
    # C_T1 global indices: k=0,1. i=0,1.
    # C flat index = k*2 + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # Local C_T1 indices: 0->0, 1->1, 2->2, 3->3.
    
    # We will write a helper to add a Strassen block with index mappings.
    
    def add_strassen_block(u_map, v_map, w_map, offset):
        """
        u_map: [u_0, u_1, u_2, u_3] global indices for A11, A12, A21, A22
        v_map: [v_0, v_1, v_2, v_3] global indices for B11, B12, B21, B22
        w_map: [w_0, w_1, w_2, w_3] global indices for C11, C12, C21, C22
        offset: start column in U_total, V_total, W_total
        """
        for r in range(7):
            for local_u, global_u in enumerate(u_map):
                if U_str[local_u, r] != 0:
                    U_total[global_u, offset + r] += U_str[local_u, r]
            for local_v, global_v in enumerate(v_map):
                if V_str[local_v, r] != 0:
                    V_total[global_v, offset + r] += V_str[local_v, r]
            for local_w, global_w in enumerate(w_map):
                if W_str[local_w, r] != 0:
                    W_total[global_w, offset + r] += W_str[local_w, r]

    # Term 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A_L global: [0, 1, 4, 5]
    u_map_L = [0, 1, 4, 5]
    
    # B_T rows 0,1. Cols 0,1,2,3,4,5,6,7.
    # Split into 4 blocks of 2 cols.
    # Block 1: Cols 0,1. V indices: [0, 1, 8, 9]
    # Block 2: Cols 2,3. V indices: [2, 3, 10, 11]
    # Block 3: Cols 4,5. V indices: [4, 5, 12, 13]
    # Block 4: Cols 6,7. V indices: [6, 7, 14, 15]
    
    # C blocks:
    # Block 1: k=0,1. W indices: [0, 1, 2, 3]
    # Block 2: k=2,3. W indices: [4, 5, 6, 7]
    # Block 3: k=4,5. W indices: [8, 9, 10, 11]
    # Block 4: k=6,7. W indices: [12, 13, 14, 15]
    
    v_maps_T = [
        [0, 1, 8, 9],
        [2, 3, 10, 11],
        [4, 5, 12, 13],
        [6, 7, 14, 15]
    ]
    w_maps_T = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    offset = 0
    for b in range(4):
        add_strassen_block(u_map_L, v_maps_T[b], w_maps_T[b], offset)
        offset += 7
        
    # Term 2: A_R (cols 2,3) * B_B (rows 2,3)
    # A_R global: [2, 3, 6, 7]
    u_map_R = [2, 3, 6, 7]
    
    # B_B rows 2,3.
    # Block 1: Cols 0,1. V indices: [16, 17, 24, 25]
    # Block 2: Cols 2,3. V indices: [18, 19, 26, 27]
    # Block 3: Cols 4,5. V indices: [20, 21, 28, 29]
    # Block 4: Cols 6,7. V indices: [22, 23, 30, 31]
    
    # C blocks: Same as before, but this is added to the same C.
    # Wait, C = A_L B_T + A_R B_B.
    # So the W indices are the same!
    # We just add the contributions.
    
    v_maps_B = [
        [16, 17, 24, 25],
        [18, 19, 26, 27],
        [20, 21, 28, 29],
        [22, 23, 30, 31]
    ]
    
    for b in range(4):
        add_strassen_block(u_map_R, v_maps_B[b], w_maps_T[b], offset)
        offset += 7
        
    return U_total, V_total, W_total


def _decompose_1x4x8():
    """
    Decompose 1x4 x 4x8 -> 1x8.
    Schoolbook: rank 32.
    U: 4 x 32
    V: 32 x 32
    W: 8 x 32
    """
    U = np.zeros((4, 32))
    V = np.zeros((32, 32))
    W = np.zeros((8, 32))
    
    col = 0
    for j in range(4):
        for k in range(8):
            U[j, col] = 1.0
            V[j * 8 + k, col] = 1.0
            W[k, col] = 1.0
            col += 1
            
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # 1. Decompose 2x4x8
    try:
        U_248, V_248, W_248 = _decompose_2x4x8()
    except Exception:
        return _build_schoolbook()
        
    # 2. Decompose 1x4x8
    try:
        U_148, V_148, W_148 = _decompose_1x4x8()
    except Exception:
        return _build_schoolbook()
        
    # 3. Assemble 3x4x8
    # U_total: 12 x (56+32=88)
    # V_total: 32 x 88
    # W_total: 24 x 88
    
    U_total = np.zeros((12, 88))
    V_total = np.zeros((32, 88))
    W_total = np.zeros((24, 88))
    
    # Part 1: Rows i=0,1 (U rows 0..7)
    U_total[0:8, 0:56] = U_248
    
    # Part 2: Row i=2 (U row 8..11)
    # U_148 is 4x32. Maps to U rows 8,9,10,11.
    U_total[8:12, 56:88] = U_148
    
    # V is shared, so V_total is the same for both parts?
    # No, V is (m*p) x rank.
    # For Part 1, V_248 is 32x56.
    # For Part 2, V_148 is 32x32.
    # We stack them horizontally.
    V_total[:, 0:56] = V_248
    V_total[:, 56:88] = V_148
    
    # W
    # Part 1: W_248 is 16x56.
    # C rows for i=0,1.
    # W index = k*n + i. n=3.
    # i=0: k*3 + 0 -> 0, 3, 6, 9, 12, 15, 18, 21
    # i=1: k*3 + 1 -> 1, 4, 7, 10, 13, 16, 19, 22
    # W_248 rows: 0..15.
    # W_248 row 0 (k=0, i=0) -> W_total[0]
    # W_248 row 1 (k=0, i=1) -> W_total[1]
    # W_248 row 2 (k=1, i=0) -> W_total[3]
    # W_248 row 3 (k=1, i=1) -> W_total[4]
    # ...
    # Pattern: W_248 row 2k -> W_total[3k]
    # W_248 row 2k+1 -> W_total[3k+1]
    
    for k in range(8):
        W_total[3*k, 0:56] = W_248[2*k, :]
        W_total[3*k+1, 0:56] = W_248[2*k+1, :]
        
    # Part 2: W_148 is 8x32.
    # C rows for i=2.
    # i=2: k*3 + 2 -> 2, 5, 8, 11, 14, 17, 20, 23
    # W_148 rows: 0..7 (k=0..7).
    for k in range(8):
        W_total[3*k+2, 56:88] = W_148[k, :]
        
    # Verify
    if time.time() < deadline:
        # Check a few points
        # C = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Pick i=0, j=1, k=2
        val = 0
        for r in range(88):
            val += U_total[0*4+1, r] * V_total[1*8+2, r] * W_total[2*3+0, r]
        # A[0,1] * B[1,2] = 1*1 = 1
        # Strassen should give 1.
        # If not 1, something is wrong.
        if abs(val - 1.0) > 1e-9:
            return _build_schoolbook()
            
        # Pick i=2, j=1, k=2
        val = 0
        for r in range(88):
            val += U_total[2*4+1, r] * V_total[1*8+2, r] * W_total[2*3+2, r]
        if abs(val - 1.0) > 1e-9:
            return _build_schoolbook()
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
