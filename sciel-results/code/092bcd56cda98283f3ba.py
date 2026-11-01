import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    
    # Strategy: Block Strassen decomposition.
    # We decompose the multiplication C = A @ B where A is 2x4, B is 4x5.
    # Split A into A1 (2x2) and A2 (2x2) horizontally? No, A is 2x4.
    # Let's split the inner dimension m=4 into two blocks of size 2.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [B_L; B_R] where B_L, B_R are 2x5.
    # C = A_L @ B_L + A_R @ B_R.
    
    # We can use Strassen for the 2x2x2 parts if we treat the 5 columns carefully.
    # Strassen for 2x2x2 requires 7 multiplications.
    # Let A_L = [[a, b], [c, d]], A_R = [[e, f], [g, h]].
    # B_L = [[b11, b12, b13, b14, b15], [b21, b22, b23, b24, b25]].
    # B_R = [[b31, b32, b33, b34, b35], [b41, b42, b43, b44, b45]].
    
    # Standard Strassen for X(2x2) * Y(2x2) -> Z(2x2):
    # M1 = (X11 + X22) * (Y11 + Y22)
    # M2 = (X21 + X22) * Y11
    # M3 = X11 * (Y12 - Y22)
    # M4 = X22 * (Y21 - Y11)
    # M5 = (X11 + X12) * Y22
    # M6 = (X21 - X11) * (Y11 + Y12)
    # M7 = (X12 - X22) * (Y21 + Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Here, our "matrices" are 2x2 by 2x5.
    # Let's apply Strassen logic to the 2x2 blocks of A multiplied by the 2x5 blocks of B.
    # However, Strassen is for square matrices. We can pad or just apply the linear combinations.
    # The rank of 2x2x5 multiplication using Strassen-like splitting:
    # If we split the 5 columns into 2+3, it gets messy.
    # Better: Split the 4x5 B matrix and 2x4 A matrix such that we can use a 2x2x2 Strassen core.
    
    # Let's try a different split.
    # A is 2x4. B is 4x5.
    # Split inner dimension 4 into 2 and 2.
    # A = [A1, A2] (each 2x2). B = [B1; B2] (each 2x5).
    # C = A1 B1 + A2 B2.
    
    # We need to compute A1 B1 (2x2 @ 2x5) and A2 B2 (2x2 @ 2x5).
    # Rank of 2x2x5 is at most 10 (schoolbook).
    # Can we do better?
    # Known rank of 2x2x5:
    # If we use Strassen on 2x2x2, we get 7 mults for the first 2 cols.
    # The remaining 3 cols?
    # Actually, there is a known decomposition for 2x2x5 with rank 9 or 10?
    # Let's look at the total rank.
    # If we just do schoolbook for 2x4x5, rank is 40.
    # If we split into two 2x2x5 problems, and if rank(2x2x5) <= 9, total rank 18.
    # Is rank(2x2x5) <= 9?
    # Lower bound for 2x2x5 is ceil(2*2*5 / something)?
    # Actually, let's use a known result: rank(2,2,5) is 9.
    # Wait, is it?
    # Rank(2,2,2) = 7.
    # Rank(2,2,3) = 8?
    # Rank(2,2,4) = 9?
    # Rank(2,2,5) = 10? Or 9?
    # If rank(2,2,5) = 9, then total rank = 18.
    # If rank(2,2,5) = 10, then total rank = 20.
    # Both are much better than 40.
    
    # Let's construct the rank 9 or 10 decomposition for 2x2x5.
    # Reference: "The rank of the 2x2x5 matrix multiplication tensor"
    # It is known that r(2,2,5) = 9.
    
    # Let's find the explicit factors for 2x2x5 with rank 9.
    # Indices: i in {0,1} (rows of A), j in {0,1} (cols of A / rows of B), k in {0,1,2,3,4} (cols of B).
    # T_{ijk} = 1 if i,j,k match in matmul sense? No, T is the multiplication tensor.
    # U_{i,j, r}, V_{j,k, r}, W_{k,i, r}.
    # Sum_r U_{ij,r} V_{jk,r} W_{ki,r} = delta_{i,k}? No.
    # Matmul: C_{ik} = Sum_j A_{ij} B_{jk}.
    # Tensor T_{i,j,k} = 1.
    # Decomposition: T_{i,j,k} = Sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # Note: The task defines W[k*n+i, r]. Here n=2, so W[k*2+i, r].
    # In my local 2x2x5 problem, n=2, m=2, p=5.
    # U shape (2*2, R) = (4, R).
    # V shape (2*5, R) = (10, R).
    # W shape (5*2, R) = (10, R).
    
    # We need to find U, V, W for 2x2x5 with R=9.
    # Since I don't have the explicit 9-rank factors memorized perfectly for 2x2x5, 
    # and constructing them on the fly is risky, I will use a safe fallback:
    # Use Strassen for the first 2 columns (rank 7) and Schoolbook for the last 3 columns (rank 6).
    # Total rank for 2x2x5 = 7 + 6 = 13.
    # Total rank for 2x4x5 = 13 + 13 = 26.
    # 26 is significantly better than 40.
    # Score: 40/26 = 1.53.
    
    # Let's refine: Can we do better than 13 for 2x2x5?
    # If I use Strassen for 2x2x2 (7) and schoolbook for 2x2x3 (6), total 13.
    # Is there a 10-rank for 2x2x5?
    # If I just hardcode the schoolbook for the whole thing, rank is 40.
    # If I use the 26-rank construction, I am safe.
    
    # Let's construct the 26-rank decomposition.
    # Split B (4x5) into B_top (2x5) and B_bot (2x5).
    # Split A (2x4) into A_left (2x2) and A_right (2x2).
    # C = A_left @ B_top + A_right @ B_bot.
    
    # For each 2x2x5 block, we use:
    # - Strassen for the first 2 columns of the 5.
    # - Schoolbook for the remaining 3 columns of the 5.
    
    # Let's build the factors for one 2x2x5 block (A_block 2x2, B_block 2x5).
    # Indices for this block: i in {0,1}, j in {0,1}, k in {0,1,2,3,4}.
    # We want U_block (4x13), V_block (10x13), W_block (10x13).
    
    U_block = np.zeros((4, 13), dtype=int)
    V_block = np.zeros((10, 13), dtype=int)
    W_block = np.zeros((10, 13), dtype=int)
    
    # --- Part 1: Strassen for k=0,1 ---
    # A_block = [[a00, a01], [a10, a11]]
    # B_block_top = [[b00, b01, ...], [b10, b11, ...]]
    # We only care about k=0,1 for Strassen.
    # Strassen factors for 2x2x2 (rank 7):
    # r=0: U=(1,0,0,1), V=(1,0,0,1) for B part? 
    # Let's map indices carefully.
    # U index: i*m + j. m=2.
    # V index: j*p + k. p=5. But for Strassen part, we only use k=0,1.
    # W index: k*n + i. n=2.
    
    # Strassen 2x2x2 factors (standard):
    # M1: (a00+a11)(b00+b11) -> U: a00,a11; V: b00,b11; W: c00,c11
    # M2: (a10+a11)b00 -> U: a10,a11; V: b00; W: c10
    # M3: a00(b01-b11) -> U: a00; V: b01, -b11; W: c01
    # M4: a11(b10-b00) -> U: a11; V: -b00, b10; W: c11
    # M5: (a00+a01)b11 -> U: a00,a01; V: b11; W: c01,c11
    # M6: (a10-a00)(b00+b01) -> U: -a00, a10; V: b00, b01; W: c10
    # M7: (a01-a11)(b10+b11) -> U: a01, -a11; V: b10, b11; W: c01
    
    # Let's fill U_block, V_block, W_block for r=0..6 corresponding to Strassen.
    # Note: V and W indices for k=0,1 are in the first 2 rows of their respective k-blocks?
    # V index = j*5 + k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 5. j=1, k=1 -> 6.
    # W index = k*2 + i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    
    # r=0 (M1)
    U_block[0, 0] = 1; U_block[3, 0] = 1 # a00, a11 (indices 0, 3 in U)
    V_block[0, 0] = 1; V_block[5, 0] = 1 # b00, b10 (indices 0, 5 in V)
    W_block[0, 0] = 1; W_block[3, 0] = 1 # c00, c11 (indices 0, 3 in W)
    
    # r=1 (M2)
    U_block[2, 1] = 1; U_block[3, 1] = 1 # a10, a11 (indices 2, 3)
    V_block[0, 1] = 1 # b00 (index 0)
    W_block[1, 1] = 1 # c10 (index 1)
    
    # r=2 (M3)
    U_block[0, 2] = 1 # a00
    V_block[1, 2] = 1; V_block[6, 2] = -1 # b01, -b11 (indices 1, 6)
    W_block[2, 2] = 1 # c01 (index 2)
    
    # r=3 (M4)
    U_block[3, 3] = 1 # a11
    V_block[0, 3] = -1; V_block[5, 3] = 1 # -b00, b10
    W_block[3, 3] = 1 # c11 (index 3)
    
    # r=4 (M5)
    U_block[0, 4] = 1; U_block[1, 4] = 1 # a00, a01 (indices 0, 1)
    V_block[5, 4] = 1 # b10? No, M5 is (a00+a01)*b11. b11 is index 6.
    # Correction: M5 = (a00+a01) * b11.
    V_block[6, 4] = 1 # b11
    W_block[2, 4] = 1; W_block[3, 4] = 1 # c01, c11 (indices 2, 3)
    
    # r=5 (M6)
    U_block[0, 5] = -1; U_block[2, 5] = 1 # -a00, a10
    V_block[0, 5] = 1; V_block[1, 5] = 1 # b00, b01
    W_block[1, 5] = 1 # c10 (index 1)
    
    # r=6 (M7)
    U_block[1, 6] = 1; U_block[3, 6] = -1 # a01, -a11
    V_block[5, 6] = 1; V_block[6, 6] = 1 # b10, b11
    W_block[2, 6] = 1 # c01 (index 2)
    
    # --- Part 2: Schoolbook for k=2,3,4 ---
    # 3 columns. 2x2x3 schoolbook rank is 6.
    # For each k in [2,3,4], and each i,j:
    # U_{ij, r} = 1, V_{jk, r} = 1, W_{ki, r} = 1.
    
    r_idx = 7
    for k in [2, 3, 4]:
        for i in [0, 1]:
            for j in [0, 1]:
                U_idx = i * 2 + j
                V_idx = j * 5 + k
                W_idx = k * 2 + i
                U_block[U_idx, r_idx] = 1
                V_block[V_idx, r_idx] = 1
                W_block[W_idx, r_idx] = 1
                r_idx += 1
    
    # Now assemble the full 2x4x5 decomposition.
    # Total rank = 13 (for A_left @ B_top) + 13 (for A_right @ B_bot) = 26.
    
    U_full = np.zeros((8, 26), dtype=int) # 2*4
    V_full = np.zeros((20, 26), dtype=int) # 4*5
    W_full = np.zeros((10, 26), dtype=int) # 5*2
    
    # A_left is columns 0,1 of A. A_right is columns 2,3 of A.
    # B_top is rows 0,1 of B. B_bot is rows 2,3 of B.
    
    # Block 1: A_left (i in 0,1; j in 0,1) @ B_top (j in 0,1; k in 0..4)
    # U indices for A_left: i*4 + j. j=0,1.
    # V indices for B_top: j*5 + k. j=0,1.
    # W indices: k*2 + i.
    
    for r in range(13):
        for u_idx in range(4): # i in 0,1; j in 0,1
            val = U_block[u_idx, r]
            if val != 0:
                i = u_idx // 2
                j = u_idx % 2
                U_full[i*4 + j, r] = val
        
        for v_idx in range(10): # j in 0,1; k in 0..4
            val = V_block[v_idx, r]
            if val != 0:
                j = v_idx // 5
                k = v_idx % 5
                V_full[j*5 + k, r] = val
        
        for w_idx in range(10): # k in 0..4; i in 0,1
            val = W_block[w_idx, r]
            if val != 0:
                k = w_idx // 2
                i = w_idx % 2
                W_full[k*2 + i, r] = val
    
    # Block 2: A_right (i in 0,1; j in 2,3) @ B_bot (j in 2,3; k in 0..4)
    # U indices for A_right: i*4 + j. j=2,3.
    # V indices for B_bot: j*5 + k. j=2,3.
    # W indices: k*2 + i.
    
    for r in range(13):
        for u_idx in range(4): # i in 0,1; j in 0,1 (local)
            val = U_block[u_idx, r]
            if val != 0:
                i = u_idx // 2
                j_local = u_idx % 2
                j_global = 2 + j_local
                U_full[i*4 + j_global, r + 13] = val
        
        for v_idx in range(10): # j in 0,1 (local); k in 0..4
            val = V_block[v_idx, r]
            if val != 0:
                j_local = v_idx // 5
                k = v_idx % 5
                j_global = 2 + j_local
                V_full[j_global*5 + k, r + 13] = val
        
        for w_idx in range(10): # k in 0..4; i in 0,1
            val = W_block[w_idx, r]
            if val != 0:
                k = w_idx // 2
                i = w_idx % 2
                W_full[k*2 + i, r + 13] = val
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
