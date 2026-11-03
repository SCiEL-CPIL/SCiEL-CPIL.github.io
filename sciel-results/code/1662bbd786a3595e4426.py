import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm.
    # Rank for 4x4x4 is 49.
    # We need 4x4x7. Pad to 4x4x8.
    # 4x4x8 is two 4x4x4 multiplications.
    # Total rank 98.
    
    # We will build U, V, W for the two 4x4x4 blocks.
    # Block 1: B columns 0-3
    # Block 2: B columns 4-7 (cols 4-6 are real, col 7 is 0)
    
    # First, define the 4x4x4 Strassen decomposition.
    # Strassen for 4x4x4 uses 7 multiplications of 2x2x2.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank 49.
    
    # We need to generate the U, V, W matrices for a single 4x4x4 multiplication.
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    
    # Strassen 2x2x2 base case:
    # U_2: 4x7, V_2: 4x7, W_2: 4x7
    # Indices for 2x2: i, j in 0..1.
    
    # Let's generate the base 2x2x2 decomposition.
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # We need to map these to U, V, W.
    # U indices: i*2+j.
    # V indices: j*2+k.
    # W indices: k*2+i.
    
    # M1:
    # A parts: A00 (0,0), A11 (1,1)
    # B parts: B00 (0,0), B11 (1,1)
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    # W[0,0]=1 (C00), W[1,1]=1 (C11) ?
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    
    # Let's define the 7 vectors for U, V, W for 2x2.
    # c=0: M1
    U2[0,0] = 1; U2[3,0] = 1 # A00, A11
    V2[0,0] = 1; V2[3,0] = 1 # B00, B11
    W2[0,0] = 1 # C00 part? No, W is for C.
    # W index k*2+i.
    # C00 is i=0, k=0 -> idx 0.
    # C11 is i=1, k=1 -> idx 3.
    # M1 contributes to C00 and C11.
    # So W2[0,0]=1, W2[3,0]=1.
    
    # c=1: M2
    # A: A10 (2,0), A11 (3,0) -> idx 2, 3
    # B: B00 (0,0) -> idx 0
    # C: C10 (i=1, k=0 -> idx 1), C11 (i=1, k=1 -> idx 3)
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[1,1] = 1; W2[3,1] = -1 # C11 = M1 - M2 ...
    
    # c=2: M3
    # A: A00 (0,0) -> idx 0
    # B: B10 (1,0) -> idx 2, B11 (1,1) -> idx 3. B10-B11.
    # C: C01 (i=0, k=1 -> idx 2), C11 (i=1, k=1 -> idx 3).
    U2[0,2] = 1
    V2[2,2] = 1; V2[3,2] = -1
    W2[2,2] = 1; W2[3,2] = 1 # C01=M3+M5, C11=M1-M2+M3+M6
    
    # c=3: M4
    # A: A11 (3,0) -> idx 3
    # B: B01 (0,1) -> idx 1, B00 (0,0) -> idx 0. B01-B00.
    # C: C00 (i=0, k=0 -> idx 0), C10 (i=1, k=0 -> idx 1).
    U2[3,3] = 1
    V2[1,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[1,3] = 1 # C00=M1+M4-M5+M7, C10=M2+M4
    
    # c=4: M5
    # A: A00 (0,0), A01 (0,1) -> idx 0, 1
    # B: B11 (1,1) -> idx 3
    # C: C00, C01.
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[2,4] = 1 # C00=M1+M4-M5+M7, C01=M3+M5
    
    # c=5: M6
    # A: A10 (2,0), A00 (0,0) -> idx 2, 0. A10-A00.
    # B: B00 (0,0), B10 (1,0) -> idx 0, 2. B00+B10.
    # C: C11.
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[2,5] = 1
    W2[3,5] = 1 # C11=M1-M2+M3+M6
    
    # c=6: M7
    # A: A01 (0,1), A11 (1,1) -> idx 1, 3. A01-A11.
    # B: B11 (1,1), B01 (0,1) -> idx 3, 1. B11+B01.
    # C: C00.
    U2[1,6] = 1; U2[3,6] = -1
    V2[3,6] = 1; V2[1,6] = 1
    W2[0,6] = 1 # C00=M1+M4-M5+M7
    
    # Now build 4x4x4.
    # Split A, B, C into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # We apply Strassen formulas on the blocks.
    # Each "multiplication" is a 2x2x2 multiplication.
    # So we use the U2, V2, W2 for each of the 7 products.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Block indices:
    # A00: rows 0-1, cols 0-1 -> U idx 0-3
    # A01: rows 0-1, cols 2-3 -> U idx 4-7
    # A10: rows 2-3, cols 0-1 -> U idx 8-11
    # A11: rows 2-3, cols 2-3 -> U idx 12-15
    
    # B00: rows 0-1, cols 0-1 -> V idx 0-3
    # B01: rows 0-1, cols 2-3 -> V idx 4-7
    # B10: rows 2-3, cols 0-1 -> V idx 8-11
    # B11: rows 2-3, cols 2-3 -> V idx 12-15
    
    # C00: rows 0-1, cols 0-1 -> W idx 0-3 (k=0,1; i=0,1)
    # C01: rows 0-1, cols 2-3 -> W idx 4-7 (k=2,3; i=0,1)
    # C10: rows 2-3, cols 0-1 -> W idx 8-11 (k=0,1; i=2,3)
    # C11: rows 2-3, cols 2-3 -> W idx 12-15 (k=2,3; i=2,3)
    
    # Helper to add a 2x2x2 block
    def add_block(u_offset, v_offset, w_offset, sign=1):
        # u_offset, v_offset, w_offset are the starting indices for the 2x2 blocks in U, V, W
        # We add the 7 rank-1 terms of U2, V2, W2
        for c in range(7):
            r = c
            # Map local U2 indices to global U4
            for i in range(4):
                if U2[i, c] != 0:
                    U4[u_offset + i, r] += sign * U2[i, c]
            for j in range(4):
                if V2[j, c] != 0:
                    V4[v_offset + j, r] += sign * V2[j, c]
            for k in range(4):
                if W2[k, c] != 0:
                    W4[w_offset + k, r] += sign * W2[k, c]
            
    # M1 = (A00+A11)(B00+B11)
    # A00+A11: U offset 0 and 12
    # B00+B11: V offset 0 and 12
    # C00: W offset 0, C11: W offset 12
    # This requires two calls? No, Strassen combines them.
    # The 2x2x2 tensor handles the addition.
    # We just need to point U to the right blocks.
    # But U2 is a single matrix. We need to sum U2 for A00 and A11.
    # So we call add_block for A00, then for A11?
    # No, the rank is 7. We can't just call it twice (rank 14).
    # The trick is that U2, V2, W2 are fixed.
    # For M1, we need U to represent A00+A11.
    # So we add U2 to U4 at offset 0, and U2 to U4 at offset 12.
    # This adds 14 non-zero columns? No.
    # The column r in U4 will have U2[:, r] at offset 0 AND U2[:, r] at offset 12.
    # This is a single rank-1 term?
    # No, $U_{ij, r} = U2_{i, r} + U2_{i, r}$ (if indices align).
    # Yes, this is valid.
    
    # M1
    add_block(0, 0, 0, 1)   # A00, B00 -> C00
    add_block(12, 12, 12, 1) # A11, B11 -> C11
    
    # M2 = (A10+A11)B00
    add_block(8, 0, 8, 1)   # A10, B00 -> C10
    add_block(12, 0, 12, -1) # A11, B00 -> C11 (M1-M2)
    
    # M3 = A00(B10-B11)
    add_block(0, 8, 4, 1)   # A00, B10 -> C01
    add_block(0, 12, 12, 1) # A00, B11 -> C11 (M1-M2+M3)
    
    # M4 = A11(B01-B00)
    add_block(12, 4, 0, 1)  # A11, B01 -> C00
    add_block(12, 0, 8, 1)  # A11, B00 -> C10 (M2+M4)
    
    # M5 = (A00+A01)B11
    add_block(0, 12, 0, -1) # A00, B11 -> C00 (M1+M4-M5)
    add_block(4, 12, 4, 1)  # A01, B11 -> C01 (M3+M5)
    
    # M6 = (A10-A00)(B00+B10)
    add_block(8, 0, 12, 1)  # A10, B00 -> C11
    add_block(0, 8, 12, -1) # A00, B10 -> C11
    add_block(8, 8, 12, 1)  # A10, B10 -> C11
    add_block(0, 0, 12, -1) # A00, B00 -> C11
    
    # M7 = (A01-A11)(B11+B01)
    add_block(4, 12, 0, 1)  # A01, B11 -> C00
    add_block(4, 4, 0, 1)   # A01, B01 -> C00
    add_block(12, 12, 0, -1) # A11, B11 -> C00
    add_block(12, 4, 0, -1) # A11, B01 -> C00
    
    # This approach increases rank?
    # No, each add_block call adds to the SAME column r.
    # So U4[:, r] is the sum of the contributions.
    # This is correct.
    
    # Now we have U4, V4, W4 for 4x4x4.
    # We need 4x4x7.
    # Pad B to 4x8.
    # B = [B_left (4x4), B_right (4x4)]
    # C = [C_left (4x4), C_right (4x4)]
    
    # U is the same for both.
    # V is different for left and right.
    # W is different for left and right.
    
    U_final = np.zeros((16, 98), dtype=int)
    V_final = np.zeros((28, 98), dtype=int)
    W_final = np.zeros((28, 98), dtype=int)
    
    # Left block (cols 0-3 of B, 0-3 of C)
    # V indices: 0-15
    # W indices: 0-15
    U_final[:, 0:49] = U4
    V_final[0:16, 0:49] = V4
    W_final[0:16, 0:49] = W4
    
    # Right block (cols 4-7 of B, 4-7 of C)
    # V indices: 16-31
    # W indices: 16-31
    U_final[:, 49:98] = U4
    V_final[16:32, 49:98] = V4
    W_final[16:32, 49:98] = W4
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
