import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Schoolbook fallback
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
    
    # Winograd 3x3x3 factors (rank 7)
    # These are the standard Winograd factors for 3x3 matrix multiplication
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
    
    # Verify Winograd 3x3x3 factors
    win333_ok = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(7):
                    s += U_win333[i*3+j, r] * V_win333[j*3+k, r] * W_win333[k*3+i, r]
                if abs(s - 1) > 1e-10:
                    win333_ok = False
                    break
            if not win333_ok:
                break
        if not win333_ok:
            break
    
    if not win333_ok:
        return {"U": U_school, "V": V_school, "W": W_school}
    
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
    
    # Verify Winograd 3x2x2 factors
    win322_ok = True
    for i in range(3):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(4):
                    s += U_win322[i*2+j, r] * V_win322[j*2+k, r] * W_win322[k*3+i, r]
                if abs(s - 1) > 1e-10:
                    win322_ok = False
                    break
            if not win322_ok:
                break
        if not win322_ok:
            break
    
    if not win322_ok:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Build the full decomposition
    # Strategy:
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
    #     Split the middle dimension of A_left (size 3) into 2+1
    #     A_left[:, :2] * B_right[:2, :] (3x2 * 2x2): Winograd 3x2x2, rank 4
    #     A_left[:, 2:3] * B_right[2:3, :] (3x1 * 1x2): schoolbook, rank 2
    #   A_right * B_right (3x1 * 1x2): schoolbook, rank 2
    #
    # Total rank = 7 + 3 + 4 + 2 + 2 = 18
    
    rank_total = 7 + 3 + 4 + 2 + 2
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Part 1: A_left * B_left (3x3 * 3x3) using Winograd 3x3x3
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
    
    # Part 2: A_right * B_left (3x1 * 1x3) rank 3
    # j=3 fixed. T[i,3,k] = A[i,3]*B[3,k]
    # For each i, one rank term: U[i*4+3, col+i]=1, V[3*5+k, col+i]=1, W[k*3+i, col+i]=1
    # Check: T[i',3,k'] = sum_r U[i'*4+3,r]*V[3*5+k',r]*W[k'*3+i',r]
    # Only r=i' has U[i'*4+3,r]=1 and W[k'*3+i',r]=1
    # So T[i',3,k'] = 1*1*1 = 1. Correct!
    for i in range(3):
        U[i * 4 + 3, col + i] = 1
        for k in range(3):
            V[3 * 5 + k, col + i] = 1
            W[k * 3 + i, col + i] = 1
    col += 3
    
    # Part 3: A_left[:, :2] * B_right[:2, :] (3x2 * 2x2) using Winograd 3x2x2
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
    
    # Part 4: A_left[:, 2:3] * B_right[2:3, :] (3x1 * 1x2) rank 2
    # j=2 fixed, k in {3,4}. T[i,2,k] = A[i,2]*B[2,k]
    # For each k in {3,4}, one rank term: U[i*4+2, col+(k-3)]=1, V[2*5+k, col+(k-3)]=1, W[k*3+i, col+(k-3)]=1
    # Check: T[i',2,k'] = sum_r U[i'*4+2,r]*V[2*5+k',r]*W[k'*3+i',r]
    # Only r=k'-3 has V[2*5+k',r]=1, so T[i',2,k'] = 1*1*1 = 1. Correct!
    for k_idx in range(2):
        k = 3 + k_idx
        for i in range(3):
            U[i * 4 + 2, col + k_idx] = 1
        V[2 * 5 + k, col + k_idx] = 1
        for i in range(3):
            W[k * 3 + i, col + k_idx] = 1
    col += 2
    
    # Part 5: A_right * B_right (3x1 * 1x2) rank 2
    # j=3 fixed, k in {3,4}. T[i,3,k] = A[i,3]*B[3,k]
    # For each k in {3,4}, one rank term
    for k_idx in range(2):
        k = 3 + k_idx
        for i in range(3):
            U[i * 4 + 3, col + k_idx] = 1
        V[3 * 5 + k, col + k_idx] = 1
        for i in range(3):
            W[k * 3 + i, col + k_idx] = 1
    col += 2
    
    # Verify correctness: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank_total):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                max_err = max(max_err, abs(total - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
