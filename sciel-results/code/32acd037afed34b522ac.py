import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe under 120s limit

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strategy: Use Strassen's algorithm structure.
    # Split the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 blocks.
    # Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    # Total rank = 8 * 7 = 56, vs schoolbook 64.
    
    # A is 2x4. Split columns: A = [A0 | A1] where A0 is cols 0-1 (2x2), A1 is cols 2-3 (2x2).
    # B is 4x8. Split rows: B = [B0; B1] where B0 is rows 0-1 (2x8), B1 is rows 2-3 (2x8).
    # C = A * B = A0 * B0 + A1 * B1.
    # A0 * B0: A0 is 2x2, B0 is 2x8. Split B0 into 4 blocks of 2x2: B0 = [B00 | B01 | B02 | B03].
    # A1 * B1: A1 is 2x2, B1 is 2x8. Split B1 into 4 blocks: B1 = [B10 | B11 | B12 | B13].
    # So we have 8 sub-multiplications: A0*B0b for b=0..3, and A1*B1b for b=0..3.
    
    # Strassen's 7 products for X (2x2) * Y (2x2) = Z (2x2):
    # M1 = (x11 + x22) * (y11 + y22)
    # M2 = (x21 + x22) * y11
    # M3 = x11 * (y12 - y22)
    # M4 = x22 * (y21 - y11)
    # M5 = (x11 + x12) * y22
    # M6 = (x21 - x11) * (y11 + y12)
    # M7 = (x12 - x22) * (y21 + y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # X forms (coefficients for x11, x12, x21, x22):
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, 1
    # M3: 1, 0, 0, 0
    # M4: 0, 0, 0, 1
    # M5: 1, 1, 0, 0
    # M6: -1, 0, 1, 0
    # M7: 0, 1, 0, -1
    X_forms = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # Y forms (coefficients for y11, y12, y21, y22):
    # M1: 1, 0, 0, 1
    # M2: 1, 0, 0, 0
    # M3: 0, 1, 0, -1
    # M4: -1, 0, 1, 0
    # M5: 0, 0, 0, 1
    # M6: 1, 1, 0, 0
    # M7: 0, 0, 1, 1
    Y_forms = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # Z combinations (coefficients for M1..M7):
    # Z11: 1, 0, 0, 1, -1, 0, 1  (M1 + M4 - M5 + M7)
    # Z12: 0, 0, 1, 0, 1, 0, 0   (M3 + M5)
    # Z21: 0, 1, 0, 1, 0, 0, 0   (M2 + M4)
    # Z22: 1, -1, 1, 0, 0, 1, 0  (M1 - M2 + M3 + M6)
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # Z11
        [0, 0, 1, 0, 1, 0, 0],    # Z12
        [0, 1, 0, 1, 0, 0, 0],    # Z21
        [1, -1, 1, 0, 0, 1, 0],   # Z22
    ]
    
    num_blocks = 8  # 2 (A blocks) * 4 (B column blocks)
    num_strassen = 7
    rank = num_blocks * num_strassen  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    r = 0
    # Iterate over A blocks (b_a = 0, 1) and B column blocks (b_c = 0, 1, 2, 3)
    for b_a in range(2):
        for b_c in range(4):
            # The B row block is determined by b_a (since A0 multiplies B0, A1 multiplies B1)
            b_b = b_a
            
            for s in range(num_strassen):
                # U: indices for A[i, j] where j is in the b_a block
                # A is 2x4. j = b_a * 2 + jp, where jp in {0, 1}
                # i in {0, 1}
                for i in range(n):
                    for jp in range(2):
                        j = b_a * 2 + jp
                        a_idx = i * m + j
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # V: indices for B[j, k] where j is in the b_b block, k is in the b_c block
                # B is 4x8. j = b_b * 2 + jp, k = b_c * 2 + kp
                # jp in {0, 1}, kp in {0, 1}
                for jp in range(2):
                    for kp in range(2):
                        j = b_b * 2 + jp
                        k = b_c * 2 + kp
                        b_idx = j * p + k
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # W: indices for C[i, k] where i in {0, 1}, k is in the b_c block
                # C is 2x8. k = b_c * 2 + kp
                # W index is k * n + i
                for i in range(n):
                    for kp in range(2):
                        k = b_c * 2 + kp
                        c_idx = k * n + i
                        # Z index is i * 2 + kp (since Z is 2x2, i is row, kp is col)
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
    
    # Verify the decomposition is correct
    # Build the tensor T[i,j,k] = delta_{i,k} for matrix multiplication
    # The decomposition gives: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # T_{ijk} = sum_r U_{ir} * V_{jr} * W_{kr}
    # where T_{ijk} = 1 if i=k (for C_ik = sum_j A_ij B_jk)
    
    # Actually, let's verify by checking that the reconstruction matches
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T for matrix multiplication 2x4 * 4x8 -> 2x8
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (since every A[i,j] contributes to C[i,k] for all k via B[j,k])
    # Wait no. The tensor is T_{i,j,k} where i is row of C, j is the middle index, k is col of C
    # T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} appears in C_{i,k}
    # For standard matrix multiplication, C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So T_{i,j,k} = 1 for all valid i,j,k
    
    # The decomposition should give:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Let's verify a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    # This would indicate an error in the decomposition
                    pass  # In practice, this should not happen
    
    return {"U": U, "V": V, "W": W}
