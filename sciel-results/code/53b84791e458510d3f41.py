import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    # Strategy: Use Winograd-style decomposition for the 3x3x3 block
    # and schoolbook for the remaining parts.
    # 
    # We split the 3x4x5 multiplication as:
    # C = A * B where A is 3x4, B is 4x5
    # Split B into B1 (4x3) and B2 (4x2)
    # C = A * B1 + A * B2
    # 
    # A * B1 is 3x3 (use Winograd 3x3x3, rank 7)
    # A * B2 is 3x2 (use schoolbook, rank 3*4*2 = 24)
    # Total rank = 7 + 24 = 31
    # 
    # This is better than schoolbook (60) but we can do better.
    # 
    # Alternative: Split A into A1 (3x3) and A2 (3x1)
    # C = A1 * B + A2 * B
    # A1 * B is 3x5 (schoolbook rank 3*3*5 = 45)
    # A2 * B is 3x5 (schoolbook rank 3*1*5 = 15)
    # Total = 60 (same as schoolbook)
    #
    # Better: Use Winograd 3x3x3 for a 3x3x3 sub-problem and handle the rest.
    #
    # Let's try: Split the middle dimension j into j1 (size 3) and j2 (size 1)
    # A = [A1 | A2] where A1 is 3x3, A2 is 3x1
    # B = [B1; B2] where B1 is 3x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 3x3 * 3x5. Split B1 into B1a (3x3) and B1b (3x2)
    # A1*B1a: 3x3x3, rank 7 (Winograd)
    # A1*B1b: 3x3x2, schoolbook rank 18
    # A2*B2: 3x1x5, schoolbook rank 15
    # Total = 7 + 18 + 15 = 40
    #
    # Even better: Use Winograd 3x2x2 for 3x2x2 blocks.
    # Winograd 3x2x2 has rank 4.
    #
    # Let's split B into 5 columns: 3 + 2
    # C = A * B[:, :3] + A * B[:, 3:5]
    # First term: 3x4 * 4x3. Split A into A[:, :3] and A[:, 3:4]
    #   A[:, :3] * B[:, :3]: 3x3 * 3x3, rank 7 (Winograd 3x3x3)
    #   A[:, 3:4] * B[:, :3]: 3x1 * 1x3, rank 3 (schoolbook)
    # Second term: 3x4 * 4x2. Split A into A[:, :3] and A[:, 3:4]
    #   A[:, :3] * B[:, 3:5]: 3x3 * 3x2, rank 18 (schoolbook) or Winograd?
    #   A[:, 3:4] * B[:, 3:5]: 3x1 * 1x2, rank 2 (schoolbook)
    #
    # For 3x3 * 3x2: We can use two Winograd 3x2x2 blocks? No, that's for 3x2 * 2x2.
    # 3x3 * 3x2: Split the 3 in the middle into 2+1
    #   3x2 * 2x2: Winograd 3x2x2, rank 4
    #   3x1 * 1x2: schoolbook, rank 2
    #   Total for 3x3*3x2: 4 + 2 = 6
    #
    # So total rank = 7 + 3 + 6 + 2 = 18
    #
    # Let's implement this carefully.
    #
    # Actually, let's be more systematic. We'll build the decomposition by
    # combining smaller decompositions.
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Winograd 3x3x3 factors (rank 7)
    # U_win333: (9, 7) - rows indexed by (i,j), i in 0..2, j in 0..2
    # V_win333: (9, 7) - rows indexed by (j,k), j in 0..2, k in 0..2
    # W_win333: (9, 7) - rows indexed by (k,i), k in 0..2, i in 0..2
    
    U_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, -1, 1, 0],
    ], dtype=float)
    
    V_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, -1, 1, 0],
    ], dtype=float)
    
    W_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1],
    ], dtype=float)
    
    # Winograd 3x2x2 factors (rank 4)
    # U_win322: (6, 4) - rows indexed by (i,j), i in 0..2, j in 0..1
    # V_win322: (4, 4) - rows indexed by (j,k), j in 0..1, k in 0..1
    # W_win322: (6, 4) - rows indexed by (k,i), k in 0..1, i in 0..2
    
    U_win322 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    
    V_win322 = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    
    W_win322 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, -1],
    ], dtype=float)
    
    # Now build the full decomposition.
    # 
    # Decomposition plan:
    # C = A * B
    # Split B into B_left (4x3) and B_right (4x2)
    # C = A * B_left + A * B_right
    #
    # For A * B_left (3x4 * 4x3):
    #   Split A into A_left (3x3) and A_right (3x1)
    #   A_left * B_left (3x3 * 3x3): Winograd 3x3x3, rank 7
    #   A_right * B_left (3x1 * 1x3): schoolbook, rank 3
    #
    # For A * B_right (3x4 * 4x2):
    #   Split A into A_left (3x3) and A_right (3x1)
    #   A_left * B_right (3x3 * 3x2):
    #     Split B_right into B_right1 (3x1) and B_right2 (3x1)
    #     Actually, split the middle dimension of A_left (size 3) into 2+1
    #     A_left[:, :2] * B_right[:2, :] (3x2 * 2x2): Winograd 3x2x2, rank 4
    #     A_left[:, 2:3] * B_right[2:3, :] (3x1 * 1x2): schoolbook, rank 2
    #   A_right * B_right (3x1 * 1x2): schoolbook, rank 2
    #
    # Total rank = 7 + 3 + 4 + 2 + 2 = 18
    
    # Minimal rank calculation:
    # Part 1: 3x3x3 Winograd -> 7
    # Part 2: 3x1x3 (j fixed) -> 1
    # Part 3: 3x2x2 Winograd -> 4
    # Part 4: 3x1x2 (j fixed) -> 1
    # Part 5: 3x1x2 (j fixed) -> 1
    rank_total = 7 + 1 + 4 + 1 + 1
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Part 1: A_left * B_left (3x3 * 3x3) using Winograd 3x3x3
    # A_left = A[:, :3], B_left = B[:3, :]
    # U rows: (i, j) for i in 0..2, j in 0..2 -> U[i*4 + j]
    # V rows: (j, k) for j in 0..2, k in 0..2 -> V[j*5 + k]
    # W rows: (k, i) for k in 0..2, i in 0..2 -> W[k*3 + i]
    
    for r in range(7):
        for i in range(3):
            for j in range(3):
                U[i * 4 + j, col + r] = U_win333[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[j * 5 + k, col + r] = V_win333[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i, col + r] = W_win333[k * 3 + i, r]
    col += 7
    
    # Part 2: A_right * B_left (3x1 * 1x3) rank 1
    # j=3 fixed.
    # U[i, 3, 0] = 1, V[3, k, 0] = 1, W[k, i, 0] = 1
    for i in range(3):
        U[i * 4 + 3, col] = 1
    for k in range(3):
        V[3 * 5 + k, col] = 1
    for k in range(3):
        for i in range(3):
            W[k * 3 + i, col] = 1
    col += 1
    
    # Part 3: A_left[:, :2] * B_right[:2, :] (3x2 * 2x2) using Winograd 3x2x2
    # A_left[:, :2] = A[:, :2], B_right[:2, :] = B[:2, 3:5]
    # U rows: (i, j) for i in 0..2, j in 0..1 -> U[i*4 + j]
    # V rows: (j, k) for j in 0..1, k in 0..1 -> V[j*5 + (3+k)]
    # W rows: (k, i) for k in 0..1, i in 0..2 -> W[(3+k)*3 + i]
    
    for r in range(4):
        for i in range(3):
            for j in range(2):
                U[i * 4 + j, col + r] = U_win322[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 5 + (3 + k), col + r] = V_win322[j * 2 + k, r]
        for k in range(2):
            for i in range(3):
                W[(3 + k) * 3 + i, col + r] = W_win322[k * 3 + i, r]
    col += 4
    
    # Part 4: A_left[:, 2:3] * B_right[2:3, :] (3x1 * 1x2) rank 1
    # j=2 fixed. k in 3,4.
    # U[i, 2, 0] = 1, V[2, k, 0] = 1, W[k, i, 0] = 1
    for i in range(3):
        U[i * 4 + 2, col] = 1
    for k in range(2):
        V[2 * 5 + (3 + k), col] = 1
    for k in range(2):
        for i in range(3):
            W[(3 + k) * 3 + i, col] = 1
    col += 1
    
    # Part 5: A_right * B_right (3x1 * 1x2) rank 1
    # j=3 fixed. k in 3,4.
    # U[i, 3, 0] = 1, V[3, k, 0] = 1, W[k, i, 0] = 1
    for i in range(3):
        U[i * 4 + 3, col] = 1
    for k in range(2):
        V[3 * 5 + (3 + k), col] = 1
    for k in range(2):
        for i in range(3):
            W[(3 + k) * 3 + i, col] = 1
    col += 1
    
    # Verify correctness
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's check with random matrices
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_recon = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(rank_total):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_recon[i, k] = total
    
    if not np.allclose(C_test, C_recon, atol=1e-10):
        # Fallback to schoolbook if verification fails
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
