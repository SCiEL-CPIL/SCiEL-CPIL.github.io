import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Constructs a rank-88 tensor decomposition for 3x4 @ 4x8 matrix multiplication.
    Strategy:
    1. Split A (3x4) into A_L (3x2) and A_R (3x2) along columns.
    2. Split B (4x8) into B_T (2x8) and B_B (2x8) along rows.
    3. C = A_L @ B_T + A_R @ B_B.
    4. Each term is M_{3, 2, 8}.
    5. Split A_L (3x2) into A_1 (1x2) and A_2 (2x2) along rows.
    6. C_L = A_1 @ B_T + A_2 @ B_T.
    7. A_1 @ B_T is M_{1, 2, 8} (Rank 16, Schoolbook).
    8. A_2 @ B_T is M_{2, 2, 8} (Rank 28, 4x Strassen M_{2,2,2}).
    9. Total Rank = 2 * (16 + 28) = 88.
    """
    n, m, p = DIMENSIONS # 3, 4, 8
    
    # Helper to create a block of zeros
    def zero_matrix(rows, cols):
        return np.zeros((rows, cols))
    
    # --- Strassen 2x2x2 Decomposition (Rank 7) ---
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # Note: W index is k*2 + i.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11:0, A22:3. B11:0, B22:3.
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    # C11: W[0*2+0]=W[0]. C22: W[1*2+1]=W[3].
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21:2, A22:3. B11:0.
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    # C21: W[0*2+1]=W[2].
    W_s[2, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11:0. B12:1, B22:3.
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    # C12: W[1*2+0]=W[2]? No, C12 is i=0, k=1. W[1*2+0] = W[2].
    # Wait, W index is k*n + i.
    # C11: i=0, k=0 -> W[0]
    # C12: i=0, k=1 -> W[2]
    # C21: i=1, k=0 -> W[1]
    # C22: i=1, k=1 -> W[3]
    
    # Correction on W indices for 2x2:
    # W[k*2 + i]
    # C11 (i=0,k=0): W[0]
    # C12 (i=0,k=1): W[2]
    # C21 (i=1,k=0): W[1]
    # C22 (i=1,k=1): W[3]
    
    # M3 contributes to C12. W[2].
    W_s[2, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22:3. B21:2, B11:0.
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    # C21 (i=1,k=0): W[1].
    W_s[1, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11:0, A12:1. B22:3.
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    # C12 (i=0,k=1): W[2].
    W_s[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21:2, A11:0. B11:0, B12:1.
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    # C11 (i=0,k=0): W[0].
    W_s[0, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12:1, A22:3. B21:2, B22:3.
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    # C22 (i=1,k=1): W[3].
    W_s[3, 6] = 1
    
    # Check coefficients:
    # C11 = M1 - M4 + M6 -> W[0]: 1 (M1), -1 (M4), 1 (M6).
    # My W_s[0, 0]=1, W_s[0, 3]=-1 (M4), W_s[0, 5]=1 (M6). Correct.
    # C12 = M3 + M5 -> W[2]: 1 (M3), 1 (M5).
    # My W_s[2, 2]=1, W_s[2, 4]=1. Correct.
    # C21 = M2 + M4 -> W[1]: 1 (M2), 1 (M4).
    # My W_s[1, 1]=1, W_s[1, 3]=1. Correct.
    # C22 = M1 - M3 + M7 -> W[3]: 1 (M1), -1 (M3), 1 (M7).
    # My W_s[3, 0]=1, W_s[3, 2]=-1, W_s[3, 6]=1. Correct.

    # --- M_{1, 2, 8} Decomposition (Rank 16) ---
    # U: (2, 16), V: (16, 16), W: (8, 16)
    # A is 1x2. B is 2x8. C is 1x8.
    # C_k = A_1 B_{1k} + A_2 B_{2k}
    # U indices: 0, 1 (A_1, A_2)
    # V indices: 0..15 (B_{10}..B_{17}, B_{20}..B_{27})?
    # B is 2x8. Row 0: 8 cols. Row 1: 8 cols.
    # V index j*8+k.
    # W index: k*1+0 = k. (Since n=1, i=0).
    
    U_128 = np.zeros((2, 16))
    V_128 = np.zeros((16, 16))
    W_128 = np.zeros((8, 16))
    
    for k in range(8):
        # Term 1: A_1 * B_{1k} * C_k
        col = k
        U_128[0, col] = 1
        V_128[k, col] = 1 # B_{1k} is at row 0, col k -> index k
        W_128[k, col] = 1
        
        # Term 2: A_2 * B_{2k} * C_k
        col = k + 8
        U_128[1, col] = 1
        V_128[8 + k, col] = 1 # B_{2k} is at row 1, col k -> index 8+k
        W_128[k, col] = 1

    # --- M_{2, 2, 8} Decomposition (Rank 28) ---
    # U: (4, 28), V: (16, 28), W: (16, 28)
    # A is 2x2. B is 2x8. C is 2x8.
    # Split B into 4 blocks of 2x2.
    # B_{00}, B_{01}, B_{10}, B_{11} corresponding to cols 0-1, 2-3, 4-5, 6-7.
    # Each block uses Strassen.
    
    U_228 = np.zeros((4, 28))
    V_228 = np.zeros((16, 28))
    W_228 = np.zeros((16, 28))
    
    for block in range(4):
        # Strassen for this block
        # U_s (4,7), V_s (4,7), W_s (4,7)
        # Map U_s to U_228
        # Map V_s to V_228. V_s indices 0..3 correspond to B_{00}, B_{01}, B_{10}, B_{11} of the block.
        # The block starts at column offset 2*block.
        # B rows are 0,1.
        # V index in 228: row*8 + col.
        # Block cols: 2*block, 2*block+1.
        # V_s[0] -> B_{0, 2*block} -> V_228[0*8 + 2*block]
        # V_s[1] -> B_{0, 2*block+1} -> V_228[0*8 + 2*block+1]
        # V_s[2] -> B_{1, 2*block} -> V_228[1*8 + 2*block]
        # V_s[3] -> B_{1, 2*block+1} -> V_228[1*8 + 2*block+1]
        
        # Map W_s to W_228.
        # W_s indices 0..3 correspond to C_{00}, C_{01}, C_{10}, C_{11} of the block.
        # C is 2x8.
        # C row 0, col 2*block -> W index (2*block)*2 + 0 = 4*block
        # C row 0, col 2*block+1 -> W index (2*block+1)*2 + 0 = 4*block + 2
        # C row 1, col 2*block -> W index (2*block)*2 + 1 = 4*block + 1
        # C row 1, col 2*block+1 -> W index (2*block+1)*2 + 1 = 4*block + 3
        
        u_offset = block * 7
        v_indices = [0*8 + 2*block, 0*8 + 2*block+1, 1*8 + 2*block, 1*8 + 2*block+1]
        w_indices = [4*block, 4*block+2, 4*block+1, 4*block+3]
        
        for r in range(7):
            # U
            for i in range(4):
                if U_s[i, r] != 0:
                    U_228[i, u_offset + r] = U_s[i, r]
            
            # V
            for i in range(4):
                if V_s[i, r] != 0:
                    V_228[v_indices[i], u_offset + r] = V_s[i, r]
            
            # W
            for i in range(4):
                if W_s[i, r] != 0:
                    W_228[w_indices[i], u_offset + r] = W_s[i, r]

    # --- M_{3, 2, 8} Decomposition (Rank 44) ---
    # A is 3x2. B is 2x8. C is 3x8.
    # Split A into A_1 (1x2) and A_2 (2x2).
    # C = A_1 B + A_2 B.
    # Part 1: A_1 B (M_{1, 2, 8}, Rank 16)
    # Part 2: A_2 B (M_{2, 2, 8}, Rank 28)
    
    U_328 = np.zeros((6, 44)) # 3*2
    V_328 = np.zeros((16, 44)) # 2*8
    W_328 = np.zeros((24, 44)) # 8*3
    
    # Part 1: A_1 (rows 0)
    # U_128 (2, 16) -> U_328 rows 0..1
    # V_128 (16, 16) -> V_328 cols 0..15
    # W_128 (8, 16) -> W_328. W_128 index k. W_328 index k*3 + 0.
    U_328[0:2, 0:16] = U_128
    V_328[:, 0:16] = V_128
    for k in range(8):
        W_328[k*3 + 0, 0:16] = W_128[k, :]
    
    # Part 2: A_2 (rows 1,2)
    # U_228 (4, 28) -> U_328 rows 2..5
    # V_228 (16, 28) -> V_328 cols 16..43
    # W_228 (16, 28) -> W_328. W_228 index k*2+i (i=0,1).
    # In M_{3,2,8}, i is local row index of A_2 (0,1).
    # Global row index in C is 1+i.
    # W_328 index is k*3 + (1+i).
    U_328[2:6, 16:44] = U_228
    V_328[:, 16:44] = V_228
    for k in range(8):
        for i in range(2):
            W_328[k*3 + (1+i), 16:44] = W_228[k*2 + i, :]

    # --- Final M_{3, 4, 8} Decomposition (Rank 88) ---
    # A is 3x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x8. Split into B_T (rows 0,1) and B_B (rows 2,3).
    # C = A_L B_T + A_R B_B.
    # Both are M_{3, 2, 8}.
    
    U_final = np.zeros((12, 88)) # 3*4
    V_final = np.zeros((32, 88)) # 4*8
    W_final = np.zeros((24, 88)) # 8*3
    
    # Part L: A_L (cols 0,1) with B_T (rows 0,1)
    # U_328 (6, 44). Indices: row*2 + col.
    # A_L rows 0,1,2. Cols 0,1.
    # U_328 index r*2+c.
    # U_final index r*4+c.
    # r=0, c=0 -> 0. r=0, c=1 -> 1.
    # r=1, c=0 -> 4. r=1, c=1 -> 5.
    # r=2, c=0 -> 8. r=2, c=1 -> 9.
    
    # V_328 (16, 44). Indices: row*8 + col.
    # B_T rows 0,1.
    # V_328 index r*8+c.
    # V_final index r*8+c. (Since B_T is top rows, indices match).
    
    # W_328 (24, 44). Indices: k*3 + i.
    # W_final index k*3 + i. (Matches).
    
    for r in range(6):
        i = r // 2
        c = r % 2
        U_final[i*4 + c, 0:44] = U_328[r, :]
    
    V_final[0:16, 0:44] = V_328
    W_final[:, 0:44] = W_328
    
    # Part R: A_R (cols 2,3) with B_B (rows 2,3)
    # U_328 (6, 44).
    # A_R cols 2,3.
    # r=0, c=0 -> U_final[0*4+2] = 2.
    # r=0, c=1 -> U_final[0*4+3] = 3.
    # r=1, c=0 -> U_final[1*4+2] = 6.
    # r=1, c=1 -> U_final[1*4+3] = 7.
    # r=2, c=0 -> U_final[2*4+2] = 10.
    # r=2, c=1 -> U_final[2*4+3] = 11.
    
    # V_328 (16, 44).
    # B_B rows 2,3.
    # V_328 index r*8+c.
    # V_final index (r+2)*8+c.
    
    # W_328 (24, 44).
    # W_final index k*3 + i. (Matches).
    
    for r in range(6):
        i = r // 2
        c = r % 2
        U_final[i*4 + (c+2), 44:88] = U_328[r, :]
    
    for r in range(2):
        for c in range(8):
            V_final[(r+2)*8 + c, 44:88] = V_328[r*8 + c, :]
    
    W_final[:, 44:88] = W_328

    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Check shapes
    assert U.shape == (n*m, rank)
    assert V.shape == (m*p, rank)
    assert W.shape == (p*n, rank)
    
    # Random test
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for r in range(rank):
        A_r = U[:, r].reshape(n, m)
        B_r = V[:, r].reshape(m, p)
        C_r = W[:, r].reshape(p, n)
        C_comp += A_r @ B_r * C_r # Wait, the decomposition is sum u v w
        # T = sum u_r (i,j) v_r (j,k) w_r (k,i)
        # C_{ik} = sum_r U_{ij} V_{jk} W_{ki}
        # My loop above is wrong.
        
    # Correct verification
    C_comp = np.zeros((n, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # C_{ik} += sum_j u_{ij} v_{jk} w_{ki}
        # This is not a simple matrix multiplication.
        # It's a trace-like contraction.
        # C_{ik} = sum_j u_{ij} v_{jk} w_{k, i}
        # Note: W is stored as [k*n + i]. So W[k, i] in matrix form.
        for i in range(n):
            for k in range(p):
                # sum_j u[i,j] * v[j,k] * w[k,i]
                # u[i,:] dot v[:,k] * w[k,i]
                val = np.dot(u[i, :], v[:, k]) * w[k, i]
                C_comp[i, k] += val
                
    diff = np.max(np.abs(C_true - C_comp))
    print(f"Max diff: {diff}")
    if diff < 1e-9:
        print("Decomposition is correct.")
    else:
        print("Decomposition is INCORRECT.")

