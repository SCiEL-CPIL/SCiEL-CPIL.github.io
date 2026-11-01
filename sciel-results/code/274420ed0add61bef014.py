import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin
    
    # Dimensions: A (2x4), B (4x4), C (2x4)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    # W index convention: W[k*n + i, r]
    
    n, m, p = 2, 4, 4
    
    # The multiplication tensor T is defined by:
    # C_{i, k} = sum_{j=0}^{3} A_{i, j} B_{j, k}
    # Indices:
    # u = i*m + j (i in 0..1, j in 0..3)
    # v = j*p + k (j in 0..3, k in 0..3)
    # w = k*n + i (k in 0..3, i in 0..1)
    
    # We aim for Rank 28.
    # The standard 2x2x2 Strassen algorithm has rank 7.
    # We can view 2x4x4 as a combination of Strassen steps.
    # Specifically, we can split the 4 columns of A and 4 rows of B.
    # A = [A_left, A_right] (2x2 each)
    # B = [[B_top_left, B_top_right], [B_bot_left, B_bot_right]] (2x2 each)
    # C = A B = [A_left B_top_left + A_right B_bot_left, A_left B_top_right + A_right B_bot_right]
    # C = [C_left, C_right]
    
    # This structure allows us to use Strassen-like combinations.
    # A known rank-28 decomposition for 2x4x4 exists.
    # Let's construct it using the "Strassen Block" method.
    
    # The 7 Strassen products for 2x2 matrices X, Y, Z (Z = XY) are:
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y21 - Y22)
    # M4 = X22(Y12 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # For 2x4x4, we have 4 "columns" of interaction.
    # Let's define the 28 terms.
    # We can group the 4 columns of B into 2 pairs? Or use a direct lookup for the 28 terms.
    # Given the complexity of deriving the exact 28 coefficients on the fly without error,
    # and the fact that Rank 32 is the safe baseline, I will implement a robust Rank 32 solution first.
    # However, the prompt asks for a strategy change.
    # The inspirations suggest Rank 28 is achievable.
    
    # Let's try to construct Rank 28 using a known decomposition.
    # Reference: "The rank of the 2x4x4 matrix multiplication tensor is 28" (or similar).
    # Actually, R(2,4,4) is known to be 28? Or 25?
    # Literature says R(2,4,4) <= 28 via Strassen.
    # Let's build the Rank 28 decomposition explicitly.
    
    # We will use 28 columns.
    # U: 8x28, V: 16x28, W: 8x28.
    
    # Let's define the 28 terms based on splitting the 4x4 B into four 2x2 blocks.
    # B = [[B00, B01], [B10, B11]] where Bxy is 2x2.
    # A = [A0, A1] where A0, A1 are 2x2.
    # C = A0 B00 + A1 B10 (left part) and A0 B01 + A1 B11 (right part).
    
    # This is 4 multiplications of 2x2 by 2x2.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank 4*7 = 28.
    # Since the inputs A0, A1 are disjoint parts of A, and B00, B01, B10, B11 are disjoint parts of B,
    # and the outputs C_left, C_right are disjoint parts of C,
    # the tensor is the direct sum of four 2x2x2 tensors.
    # Therefore, the rank is exactly 7 + 7 + 7 + 7 = 28.
    
    # We just need to map the Strassen coefficients to the global indices.
    
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    # Strassen coefficients for 2x2 multiplication Z = XY
    # M1: X11+X22, Y11+Y22 -> Z11+Z22+...
    # Let's define the 7 vectors for U, V, W for a single 2x2 multiplication.
    # Local indices for X (2x2): x00, x01, x10, x11
    # Local indices for Y (2x2): y00, y01, y10, y11
    # Local indices for Z (2x2): z00, z01, z10, z11
    
    # Strassen Products:
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y10 - y11)
    # M4 = x11(y01 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # Z reconstruction:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # Define local vectors (length 4) for U (X), V (Y), W (Z) for each M
    # Note: In global tensor, U corresponds to A, V to B, W to C.
    # A is 2x4. We split A into A0 (cols 0,1) and A1 (cols 2,3).
    # A0 indices in global U:
    # A0_00 -> U[0*4+0] = U[0]
    # A0_01 -> U[0*4+1] = U[1]
    # A0_10 -> U[1*4+0] = U[4]
    # A0_11 -> U[1*4+1] = U[5]
    # A1 indices in global U:
    # A1_00 -> U[0*4+2] = U[2]
    # A1_01 -> U[0*4+3] = U[3]
    # A1_10 -> U[1*4+2] = U[6]
    # A1_11 -> U[1*4+3] = U[7]
    
    # B is 4x4. We split into B00 (rows 0,1; cols 0,1), B01 (rows 0,1; cols 2,3),
    # B10 (rows 2,3; cols 0,1), B11 (rows 2,3; cols 2,3).
    # Global V indices:
    # B_row, col -> V[row*4 + col]
    # B00: rows 0,1, cols 0,1
    # B00_00 -> V[0]
    # B00_01 -> V[1]
    # B00_10 -> V[4]
    # B00_11 -> V[5]
    # B01: rows 0,1, cols 2,3
    # B01_00 -> V[2]
    # B01_01 -> V[3]
    # B01_10 -> V[6]
    # B01_11 -> V[7]
    # B10: rows 2,3, cols 0,1
    # B10_00 -> V[8]
    # B10_01 -> V[9]
    # B10_10 -> V[12]
    # B10_11 -> V[13]
    # B11: rows 2,3, cols 2,3
    # B11_00 -> V[10]
    # B11_01 -> V[11]
    # B11_10 -> V[14]
    # B11_11 -> V[15]
    
    # C is 2x4. We split into C0 (cols 0,1) and C1 (cols 2,3).
    # C0 comes from A0 B00 + A1 B10.
    # C1 comes from A0 B01 + A1 B11.
    # Global W indices: W[k*n + i] where k is col, i is row.
    # C0 (cols 0,1):
    # C0_00 (row 0, col 0) -> W[0*2+0] = W[0]
    # C0_01 (row 0, col 1) -> W[1*2+0] = W[2]
    # C0_10 (row 1, col 0) -> W[0*2+1] = W[1]
    # C0_11 (row 1, col 1) -> W[1*2+1] = W[3]
    # C1 (cols 2,3):
    # C1_00 (row 0, col 2) -> W[2*2+0] = W[4]
    # C1_01 (row 0, col 3) -> W[3*2+0] = W[6]
    # C1_10 (row 1, col 2) -> W[2*2+1] = W[5]
    # C1_11 (row 1, col 3) -> W[3*2+1] = W[7]
    
    # We have 4 independent 2x2 multiplications:
    # 1. A0 * B00 -> C_part_00 (contributes to C0)
    # 2. A1 * B10 -> C_part_10 (contributes to C0)
    # 3. A0 * B01 -> C_part_01 (contributes to C1)
    # 4. A1 * B11 -> C_part_11 (contributes to C1)
    
    # For each multiplication, we use 7 Strassen terms.
    # Total 28 terms.
    
    # Define the 7 Strassen vectors (local)
    # Local U indices: 0:00, 1:01, 2:10, 3:11 (row-major 2x2)
    # Local V indices: 0:00, 1:01, 2:10, 3:11
    # Local W indices: 0:00, 1:01, 2:10, 3:11
    
    # M1
    u1 = np.array([1, 0, 0, 1]) # x00+x11
    v1 = np.array([1, 0, 0, 1]) # y00+y11
    w1 = np.array([1, 0, 0, 1]) # z00+z11 (part of M1 in z00 and z11)
    # Actually, let's use the explicit linear combinations for W.
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # So for each M_i, we define w_i as the vector of coefficients in the linear combination.
    # M1: w1 = [1, 0, 0, 1]
    # M2: w2 = [0, 0, 1, -1]
    # M3: w3 = [0, 1, 0, 1]
    # M4: w4 = [1, 0, 1, 0]
    # M5: w5 = [-1, 1, 0, 0]
    # M6: w6 = [0, 0, 0, 1]
    # M7: w7 = [1, 0, 0, 0]
    
    # Let's verify:
    # z00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7 = M1 + M4 - M5 + M7. Correct.
    # z01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7 = M3 + M5. Correct.
    # z10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7 = M2 + M4. Correct.
    # z11 = 1*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7 = M1 - M2 + M3 + M6. Correct.
    
    # U, V vectors for M1..M7:
    # M1: u=[1,0,0,1], v=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0]
    # M3: u=[1,0,0,0], v=[0,0,1,-1]
    # M4: u=[0,0,0,1], v=[0,1,-1,0]
    # M5: u=[1,1,0,0], v=[0,0,0,1]
    # M6: u=[0,-1,1,0], v=[1,1,0,0]
    # M7: u=[0,1,0,-1], v=[0,0,1,1]
    
    U_local = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    # Rows are local indices 0,1,2,3. Cols are M1..M7.
    
    V_local = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Now map to global indices for the 4 blocks.
    
    # Block 1: A0 * B00 -> C0_part
    # A0 global U indices: [0, 1, 4, 5] corresponding to local [0, 1, 2, 3]
    # B00 global V indices: [0, 1, 4, 5]
    # C0 global W indices: [0, 2, 1, 3] (z00->W0, z01->W2, z10->W1, z11->W3)
    # Note: W_local row 0 is z00, row 1 is z01, row 2 is z10, row 3 is z11.
    
    # Block 2: A1 * B10 -> C0_part
    # A1 global U indices: [2, 3, 6, 7]
    # B10 global V indices: [8, 9, 12, 13]
    # C0 global W indices: [0, 2, 1, 3] (Same C0 output)
    
    # Block 3: A0 * B01 -> C1_part
    # A0 global U indices: [0, 1, 4, 5]
    # B01 global V indices: [2, 3, 6, 7]
    # C1 global W indices: [4, 6, 5, 7] (z00->W4, z01->W6, z10->W5, z11->W7)
    
    # Block 4: A1 * B11 -> C1_part
    # A1 global U indices: [2, 3, 6, 7]
    # B11 global V indices: [10, 11, 14, 15]
    # C1 global W indices: [4, 6, 5, 7]
    
    blocks = [
        {
            'u_map': [0, 1, 4, 5],
            'v_map': [0, 1, 4, 5],
            'w_map': [0, 2, 1, 3],
            'offset': 0
        },
        {
            'u_map': [2, 3, 6, 7],
            'v_map': [8, 9, 12, 13],
            'w_map': [0, 2, 1, 3],
            'offset': 7
        },
        {
            'u_map': [0, 1, 4, 5],
            'v_map': [2, 3, 6, 7],
            'w_map': [4, 6, 5, 7],
            'offset': 14
        },
        {
            'u_map': [2, 3, 6, 7],
            'v_map': [10, 11, 14, 15],
            'w_map': [4, 6, 5, 7],
            'offset': 21
        }
    ]
    
    for block in blocks:
        off = block['offset']
        u_map = block['u_map']
        v_map = block['v_map']
        w_map = block['w_map']
        
        for local_idx in range(4):
            # Fill U
            for m_idx in range(7):
                U[u_map[local_idx], off + m_idx] = U_local[local_idx, m_idx]
                
            # Fill V
            for m_idx in range(7):
                V[v_map[local_idx], off + m_idx] = V_local[local_idx, m_idx]
                
            # Fill W
            for m_idx in range(7):
                W[w_map[local_idx], off + m_idx] = W_local[local_idx, m_idx]
                
    # Verify the decomposition
    T = np.zeros((8, 16, 8))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 2 + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
    
    if not np.allclose(T_calc, T, atol=1e-6):
        # Fallback to Rank 32 if construction failed (should not happen)
        U32 = np.zeros((8, 32))
        V32 = np.zeros((16, 32))
        W32 = np.zeros((8, 32))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    u_idx = i * 4 + j
                    v_idx = j * 4 + k
                    w_idx = k * 2 + i
                    U32[u_idx, col] = 1
                    V32[v_idx, col] = 1
                    W32[w_idx, col] = 1
                    col += 1
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
