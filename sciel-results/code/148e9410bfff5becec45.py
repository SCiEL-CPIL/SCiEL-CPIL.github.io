"""Matrix multiplication 5x5x5 with Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, each 4x7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A indices: 0->00, 1->01, 2->10, 3->11
    # B indices: 0->00, 1->01, 2->10, 3->11
    # C indices: 0->00, 1->01, 2->10, 3->11
    
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00 (+1), C11 (+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (A10+A11)*B00 -> contributes to C10 (+1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00*(B01-B11) -> contributes to C01 (+1), C11 (-1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = -1
    
    # M4 = A11*(B10-B00) -> contributes to C00 (+1), C10 (+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5 = (A00+A01)*B11 -> contributes to C00 (-1), C01 (+1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B10) -> contributes to C11 (+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B01+B11) -> contributes to C00 (+1), C01 (-1)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    W[1][6] = -1
    
    return U, V, W

def construct():
    deadline = time.time() + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Partition: 5 = 2+2+1
    sizes = [2, 2, 1]
    
    # Compute offsets
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    all_U = []  # List of (row_index, value, term_index)
    all_V = []
    all_W = []
    
    term_count = 0
    
    for r1 in range(3):
        for r2 in range(3):
            for c1 in range(3):
                s_r1 = sizes[r1]
                s_r2 = sizes[r2]
                s_c1 = sizes[c1]
                
                if s_r1 == 2 and s_r2 == 2 and s_c1 == 2:
                    # Use Strassen, rank 7
                    U_block, V_block, W_block = strassen_2x2x2()
                    for r in range(7):
                        # U_block has 4 rows (local A indices 0,1,2,3 -> 00,01,10,11)
                        for li in range(4):
                            if U_block[li][r] != 0:
                                # Global A index: i = offsets[r1] + li//2, j = offsets[r2] + li%2
                                i_global = offsets[r1] + li // 2
                                j_global = offsets[r2] + li % 2
                                all_U.append((i_global * m + j_global, U_block[li][r], term_count))
                        # V_block has 4 rows (local B indices 0,1,2,3 -> 00,01,10,11)
                        for lj in range(4):
                            if V_block[lj][r] != 0:
                                # Global B index: j = offsets[r2] + lj//2, k = offsets[c1] + lj%2
                                j_global = offsets[r2] + lj // 2
                                k_global = offsets[c1] + lj % 2
                                all_V.append((j_global * p + k_global, V_block[lj][r], term_count))
                        # W_block has 4 rows (local C indices 0,1,2,3 -> 00,01,10,11)
                        for lk in range(4):
                            if W_block[lk][r] != 0:
                                # Global W index: k = offsets[c1] + lk//2, i = offsets[r1] + lk%2
                                # W is indexed as W[k*n + i, r]
                                k_global = offsets[c1] + lk // 2
                                i_global = offsets[r1] + lk % 2
                                all_W.append((k_global * n + i_global, W_block[lk][r], term_count))
                        term_count += 1
                elif s_r1 == 2 and s_r2 == 2 and s_c1 == 1:
                    # 2x2x1: C[i,0] = sum_j A[i,j] B[j,0], rank 4
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = offsets[r1] + i_local
                            j_global = offsets[r2] + j_local
                            k_global = offsets[c1] + 0
                            all_U.append((i_global * m + j_global, 1, term_count))
                            all_V.append((j_global * p + k_global, 1, term_count))
                            all_W.append((k_global * n + i_global, 1, term_count))
                            term_count += 1
                elif s_r1 == 2 and s_r2 == 1 and s_c1 == 2:
                    # 2x1x2: C[i,k] = A[i,0]*B[0,k], rank 4
                    for i_local in range(2):
                        for k_local in range(2):
                            i_global = offsets[r1] + i_local
                            j_global = offsets[r2] + 0
                            k_global = offsets[c1] + k_local
                            all_U.append((i_global * m + j_global, 1, term_count))
                            all_V.append((j_global * p + k_global, 1, term_count))
                            all_W.append((k_global * n + i_global, 1, term_count))
                            term_count += 1
                elif s_r1 == 2 and s_r2 == 1 and s_c1 == 1:
                    # 2x1x1: C[i,0] = A[i,0]*B[0,0], rank 2
                    for i_local in range(2):
                        i_global = offsets[r1] + i_local
                        j_global = offsets[r2] + 0
                        k_global = offsets[c1] + 0
                        all_U.append((i_global * m + j_global, 1, term_count))
                        all_V.append((j_global * p + k_global, 1, term_count))
                        all_W.append((k_global * n + i_global, 1, term_count))
                        term_count += 1
                elif s_r1 == 1 and s_r2 == 2 and s_c1 == 2:
                    # 1x2x2: C[0,k] = sum_j A[0,j]*B[j,k], rank 4
                    for j_local in range(2):
                        for k_local in range(2):
                            i_global = offsets[r1] + 0
                            j_global = offsets[r2] + j_local
                            k_global = offsets[c1] + k_local
                            all_U.append((i_global * m + j_global, 1, term_count))
                            all_V.append((j_global * p + k_global, 1, term_count))
                            all_W.append((k_global * n + i_global, 1, term_count))
                            term_count += 1
                elif s_r1 == 1 and s_r2 == 2 and s_c1 == 1:
                    # 1x2x1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], rank 2
                    for j_local in range(2):
                        i_global = offsets[r1] + 0
                        j_global = offsets[r2] + j_local
                        k_global = offsets[c1] + 0
                        all_U.append((i_global * m + j_global, 1, term_count))
                        all_V.append((j_global * p + k_global, 1, term_count))
                        all_W.append((k_global * n + i_global, 1, term_count))
                        term_count += 1
                elif s_r1 == 1 and s_r2 == 1 and s_c1 == 2:
                    # 1x1x2: C[0,k] = A[0,0]*B[0,k], rank 2
                    for k_local in range(2):
                        i_global = offsets[r1] + 0
                        j_global = offsets[r2] + 0
                        k_global = offsets[c1] + k_local
                        all_U.append((i_global * m + j_global, 1, term_count))
                        all_V.append((j_global * p + k_global, 1, term_count))
                        all_W.append((k_global * n + i_global, 1, term_count))
                        term_count += 1
                elif s_r1 == 1 and s_r2 == 1 and s_c1 == 1:
                    # 1x1x1: C[0,0] = A[0,0]*B[0,0], rank 1
                    i_global = offsets[r1] + 0
                    j_global = offsets[r2] + 0
                    k_global = offsets[c1] + 0
                    all_U.append((i_global * m + j_global, 1, term_count))
                    all_V.append((j_global * p + k_global, 1, term_count))
                    all_W.append((k_global * n + i_global, 1, term_count))
                    term_count += 1
    
    rank = term_count
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, val, col in all_U:
        U[row][col] = val
    for row, val, col in all_V:
        V[row][col] = val
    for row, val, col in all_W:
        W[row][col] = val
    
    # Verify the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's check a few random samples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    raise ValueError(f"Decomposition failed at i={i}, j={j}, k={k}: expected {expected}, got {actual}")
    
    return {"U": U, "V": V, "W": W}
