import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    # Strassen factors for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0], # A11
        [0, 0, 0, 0, 1, 1, 1], # A12
        [0, 1, 0, 0, 0, -1, 0], # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0], # B11
        [1, 0, 1, 0, 0, 1, 0], # B12
        [0, 0, 0, 1, 0, 0, 1], # B21
        [1, -1, -1, -1, 1, 0, 1] # B22
    ]
    W_strassen = [
        [1, 0, 0, 0, 0, 0, 0], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 1, 1], # C21
        [1, 1, 0, 0, -1, 0, 1] # C22
    ]
    
    # Verify Strassen
    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # i,j,k in 0..3
    # i: A idx, j: B idx, k: C idx
    # A idx: 0=A11, 1=A12, 2=A21, 3=A22
    # B idx: 0=B11, 1=B12, 2=B21, 3=B22
    # C idx: 0=C11, 1=C12, 2=C21, 3=C22
    
    # Check C11 (k=0)
    # C11 = A11 B11 + A12 B21
    # Sum_c U[i,c] V[j,c] W[0,c]
    # W[0,c] is non-zero only for c=0 (val 1)
    # So term is U[i,0] V[j,0] * 1
    # U[:,0] = [1, 0, 0, 1] (A11, A22)
    # V[:,0] = [1, 1, 0, 1] (B11, B12, B22)
    # Product: A11 B11 + A11 B12 + A22 B11 + A22 B12
    # This is NOT C11.
    
    # My Strassen factors might be wrong or I am misinterpreting.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M5 + M7
    
    # Let's re-derive U, V, W from these equations.
    # Term 1 (M1):
    # U1: A11=1, A22=1 -> [1, 0, 0, 1]
    # V1: B11=1, B22=1 -> [1, 0, 0, 1]
    # W1: C11=1, C22=1 -> [1, 0, 0, 1]
    
    # Term 2 (M2):
    # U2: A21=1, A22=1 -> [0, 0, 1, 1]
    # V2: B11=1 -> [1, 0, 0, 0]
    # W2: C21=1 -> [0, 0, 1, 0]
    
    # Term 3 (M3):
    # U3: A11=1 -> [1, 0, 0, 0]
    # V3: B12=1, B22=-1 -> [0, 1, 0, -1]
    # W3: C12=1 -> [0, 1, 0, 0]
    
    # Term 4 (M4):
    # U4: A22=1 -> [0, 0, 0, 1]
    # V4: B21=1, B11=-1 -> [-1, 0, 1, 0]
    # W4: C11=-1, C21=1 -> [-1, 0, 1, 0]
    
    # Term 5 (M5):
    # U5: A11=1, A12=1 -> [1, 1, 0, 0]
    # V5: B22=1 -> [0, 0, 0, 1]
    # W5: C12=1, C22=-1 -> [0, 1, 0, -1]
    
    # Term 6 (M6):
    # U6: A21=1, A11=-1 -> [-1, 0, 1, 0]
    # V6: B11=1, B12=1 -> [1, 1, 0, 0]
    # W6: C11=-1, C21=1 -> [-1, 0, 1, 0]
    
    # Term 7 (M7):
    # U7: A12=1, A22=-1 -> [0, 1, 0, -1]
    # V7: B21=1, B22=1 -> [0, 0, 1, 1]
    # W7: C11=1, C22=1 -> [1, 0, 0, 1]

    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, -1, 0, -1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, -1, 0, 1]
    ]

    # Construct rank 28 decomposition
    rank = 28
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(8)]
    
    # Products:
    # 1: A11 B11 -> C11 (cols 0-6)
    # 2: A11 B12 -> C12 (cols 7-13)
    # 3: A12 B21 -> C11 (cols 14-20)
    # 4: A12 B22 -> C12 (cols 21-27)
    
    # Index mappings
    # A: 0-3 (A11), 4-7 (A12)
    # B: 0-3 (B11), 4-7 (B12), 8-11 (B21), 12-15 (B22)
    # C: 0-3 (C11), 4-7 (C12)
    
    # Wait, B indices in Strassen are 0-3 corresponding to B11, B12, B21, B22.
    # B11 idx: 0,1,4,5? No.
    # B is 4x4. B11 is top-left 2x2.
    # B[0,0], B[0,1], B[1,0], B[1,1].
    # Flattened: 0, 1, 4, 5.
    
    def map_B(idx_block, idx_in_block):
        # idx_block: 0=B11, 1=B12, 2=B21, 3=B22
        # idx_in_block: 0..3
        # B11: 0,1,4,5
        # B12: 2,3,6,7
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        offsets = [0, 2, 8, 10]
        # idx_in_block 0->0, 1->1, 2->4, 3->5
        # Wait, 2x2 block indices:
        # 0 1
        # 2 3
        # Flat: 0, 1, 4, 5
        # So mapping is not linear.
        # 0->0, 1->1, 2->4, 3->5
        local_map = [0, 1, 4, 5]
        return offsets[idx_block] + local_map[idx_in_block]

    def map_C(idx_block, idx_in_block):
        # C11: 0,1,4,5
        # C12: 2,3,6,7
        offsets = [0, 2]
        local_map = [0, 1, 4, 5]
        return offsets[idx_block] + local_map[idx_in_block]

    def map_A(idx_block, idx_in_block):
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        offsets = [0, 2]
        local_map = [0, 1, 4, 5]
        return offsets[idx_block] + local_map[idx_in_block]

    # Fill U, V, W
    for prod_idx in range(4):
        base_col = prod_idx * 7
        # Determine which A block, B block, C block
        if prod_idx == 0:
            a_block = 0 # A11
            b_block = 0 # B11
            c_block = 0 # C11
        elif prod_idx == 1:
            a_block = 0 # A11
            b_block = 1 # B12
            c_block = 1 # C12
        elif prod_idx == 2:
            a_block = 1 # A12
            b_block = 2 # B21
            c_block = 0 # C11
        else:
            a_block = 1 # A12
            b_block = 3 # B22
            c_block = 1 # C12
            
        for c in range(7):
            col = base_col + c
            # U
            for i in range(4):
                val = U_s[i][c]
                if val != 0:
                    a_idx = map_A(a_block, i)
                    U[a_idx][col] += val
                    
            # V
            for j in range(4):
                val = V_s[j][c]
                if val != 0:
                    b_idx = map_B(b_block, j)
                    V[b_idx][col] += val
                    
            # W
            for k in range(4):
                val = W_s[k][c]
                if val != 0:
                    # Only add if k corresponds to c_block
                    # W_s indices: 0=C11, 1=C12, 2=C21, 3=C22
                    if k == c_block:
                        c_idx = map_C(c_block, k)
                        W[c_idx][col] += val
                    # else ignore (C21, C22 not in output)

    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Reshape
    # U: (2, 4, 28)
    # V: (4, 4, 28)
    # W: (4, 2, 28) -> W[k*n+i] -> W[k, i]
    # W is (p, n, rank) = (4, 2, 28)
    
    U_r = U_arr.reshape(2, 4, 28)
    V_r = V_arr.reshape(4, 4, 28)
    W_r = W_arr.reshape(4, 2, 28)
    
    # Check: sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
    # i in 0..1, j in 0..3, k in 0..3
    
    valid = True
    for i in range(2):
        for j in range(4):
            for k in range(4):
                val = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 28

    return best_result
