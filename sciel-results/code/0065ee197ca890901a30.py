import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2x2 Strassen factors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 2. Construct 4x4x4 Strassen factors (Rank 49)
    # U4, V4, W4 are 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block mappings for Strassen steps
    # Each step uses specific A, B, C blocks
    # Block indices: 00, 01, 10, 11
    # Global index mapping:
    # Block br, bc -> global rows br*2..br*2+1, cols bc*2..bc*2+1
    
    steps = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (br, bc, sign)
        # B_blocks: list of (br, bc, sign)
        # C_blocks: list of (br, bc, sign)
        # We need to map the 2x2 factors to these blocks.
        # Actually, it's easier to just embed U2, V2, W2 into the 4x4 matrices
        # by adding them at the correct positions for each of the 7 sub-steps.
        # But the 7 Strassen steps themselves are combinations of 2x2 blocks.
        # So for each of the 7 Strassen steps (M1..M7), we have a 2x2x2 tensor.
        # We need to place this 2x2x2 tensor into the 4x4x4 space.
        
        # M1: A00+A11, B00+B11 -> C00+C11
        # M2: A10+A11, B00 -> C10-C11
        # M3: A00, B10-B11 -> C01+C11
        # M4: A11, B01-B00 -> C00+C10
        # M5: A00+A01, B11 -> C00-C01 (Wait, C00 - M5 + ... so -1 for C00, +1 for C01)
        # M6: A10-A00, B00+B01 -> C11
        # M7: A01-A11, B10+B11 -> C00
    ]
    
    # Helper to embed 2x2 factors into 4x4 matrices
    def embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, offset, a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (br, bc, sign)
        for lr in range(2):
            for lc in range(2):
                val_u = U2[lr, lc]
                val_v = V2[lr, lc]
                val_w = W2[lr, lc]
                
                # Embed U
                for br, bc, sign in a_blocks:
                    if val_u != 0:
                        gi = br * 2 + lr
                        gj = bc * 2 + lc
                        U4[gi * 4 + gj, offset] += sign * val_u
                        
                # Embed V
                for br, bc, sign in b_blocks:
                    if val_v != 0:
                        gi = br * 2 + lr
                        gj = bc * 2 + lc
                        V4[gi * 4 + gj, offset] += sign * val_v
                        
                # Embed W
                for br, bc, sign in c_blocks:
                    if val_w != 0:
                        gi = br * 2 + lr
                        gj = bc * 2 + lc
                        W4[gi * 4 + gj, offset] += sign * val_w

    # Define block configurations for each Strassen step
    # Note: The signs in W2 are already accounted for in the 2x2x2 definition?
    # No, W2 contains the signs for the C blocks.
    # So we just need to map the 2x2 indices to the 4x4 blocks.
    
    # Step 1: M1
    # A: 00, 11
    # B: 00, 11
    # C: 00, 11
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 0, 
                        [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
                        
    # Step 2: M2
    # A: 10, 11
    # B: 00
    # C: 10, 11 (W2 has signs)
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 7, 
                        [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)])
                        
    # Step 3: M3
    # A: 00
    # B: 10, 11
    # C: 01, 11
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 14, 
                        [(0,0,1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)])
                        
    # Step 4: M4
    # A: 11
    # B: 01, 00
    # C: 00, 10
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 21, 
                        [(1,1,1)], [(0,1,1), (0,0,1)], [(0,0,1), (1,0,1)])
                        
    # Step 5: M5
    # A: 00, 01
    # B: 11
    # C: 00, 01
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 28, 
                        [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)])
                        
    # Step 6: M6
    # A: 10, 00
    # B: 00, 01
    # C: 11
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 35, 
                        [(1,0,1), (0,0,1)], [(0,0,1), (0,1,1)], [(1,1,1)])
                        
    # Step 7: M7
    # A: 01, 11
    # B: 10, 11
    # C: 00
    embed_2x2_into_4x4(U2, V2, W2, U4, V4, W4, 42, 
                        [(0,1,1), (1,1,1)], [(1,0,1), (1,1,1)], [(0,0,1)])

    # 3. Construct 4x4x8 factors (Rank 98)
    # U: 16x98
    # V: 32x98
    # W: 32x98
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half (cols 0-3 of B and C)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4  # V4 is 16x49, maps to B cols 0-3
    W[0:16, 0:49] = W4  # W4 is 16x49, maps to C cols 0-3
    
    # Second half (cols 4-7 of B and C)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4  # Maps to B cols 4-7
    W[16:32, 49:98] = W4  # Maps to C cols 4-7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
