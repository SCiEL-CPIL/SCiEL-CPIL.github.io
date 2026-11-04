import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook baseline (just in case)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def generate_strassen_4x4x4():
        """
        Generates U, V, W for 4x4 * 4x4 multiplication using Strassen's algorithm.
        Rank 28.
        U: (16, 28), V: (16, 28), W: (16, 28)
        A indices: i*4 + j
        B indices: j*4 + k
        C indices: i*4 + k
        """
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        def block_a_indices(bi, bj):
            return [(bi*2 + di)*4 + (bj*2 + dj) for di in range(2) for dj in range(2)]
        
        def block_b_indices(bi, bj):
            return [(bi*2 + dj)*4 + (bj*2 + dk) for dj in range(2) for dk in range(2)]
        
        def block_c_indices(bi, bj):
            return [(bi*2 + di)*4 + (bj*2 + dk) for di in range(2) for dk in range(2)]
        
        col = 0
        for bi in range(2):
            for bk in range(2):
                for bj in range(2):
                    a_idx = block_a_indices(bi, bj)
                    b_idx = block_b_indices(bj, bk)
                    c_idx = block_c_indices(bi, bk)
                    
                    a11, a12, a21, a22 = a_idx
                    b11, b12, b21, b22 = b_idx
                    c11, c12, c21, c22 = c_idx
                    
                    # M1 = (A11+A22)(B11+B22)
                    U[a11][col] += 1; U[a22][col] += 1
                    V[b11][col] += 1; V[b22][col] += 1
                    W[c11][col] += 1; W[c22][col] += 1
                    col += 1
                    
                    # M2 = (A21+A22)B11
                    U[a21][col] += 1; U[a22][col] += 1
                    V[b11][col] += 1
                    W[c21][col] += 1
                    col += 1
                    
                    # M3 = A11(B12-B22)
                    U[a11][col] += 1
                    V[b12][col] += 1; V[b22][col] -= 1
                    W[c12][col] += 1
                    col += 1
                    
                    # M4 = A22(B21-B11)
                    U[a22][col] += 1
                    V[b21][col] += 1; V[b11][col] -= 1
                    W[c11][col] += 1; W[c21][col] += 1
                    col += 1
                    
                    # M5 = (A11+A12)B22
                    U[a11][col] += 1; U[a12][col] += 1
                    V[b22][col] += 1
                    W[c11][col] -= 1; W[c12][col] += 1
                    col += 1
                    
                    # M6 = (A21-A11)(B11+B12)
                    U[a21][col] += 1; U[a11][col] -= 1
                    V[b11][col] += 1; V[b12][col] += 1
                    W[c22][col] += 1
                    col += 1
                    
                    # M7 = (A12-A22)(B21+B22)
                    U[a12][col] += 1; U[a22][col] -= 1
                    V[b21][col] += 1; V[b22][col] += 1
                    W[c11][col] += 1
                    col += 1
        
        return U, V, W

    try:
        U_str, V_str, W_str = generate_strassen_4x4x4()
        
        # Construct full 4x4*4x7
        # Split B into B_left (4x4) and B_right (4x3)
        # We treat B_right as 4x4 with last col zero.
        # C_left is 4x4, C_right is 4x3 (we take first 3 cols of the 4x4 result)
        
        total_rank = 28 + 28
        U_full = [[0] * total_rank for _ in range(16)]
        V_full = [[0] * total_rank for _ in range(28)] # m*p = 4*7
        W_full = [[0] * total_rank for _ in range(28)] # p*n = 7*4
        
        # Part 1: B[:, 0:4] -> C[:, 0:4]
        # U is same
        # V: B[j, k] -> V[j*7+k]. Strassen V is [j*4+k]. k in 0..3.
        # W: C[i, k] -> W[k*4+i]. Strassen W is [i*4+k]. k in 0..3.
        
        # U is identical for both parts (since A is the same 4x4 matrix)
        for c in range(28):
            for idx in range(16):
                U_full[idx][c] = U_str[idx][c]
                U_full[idx][c + 28] = U_str[idx][c]
                
        # V mapping
        # Strassen V: 16x28. Index v_idx = j*4 + k.
        # Full V: 28x28 (rows j*7+k).
        # Part 1 (cols 0-27 of rank): maps to B[:, 0:4].
        #   j in 0..3, k in 0..3.
        #   Full index: j*7 + k.
        # Part 2 (cols 28-55 of rank): maps to B[:, 4:7] (padded to 4:8).
        #   j in 0..3, k_local in 0..3.
        #   Full index: j*7 + (4 + k_local).
        
        for c in range(28):
            for v_idx in range(16):
                val = V_str[v_idx][c]
                if val != 0:
                    j = v_idx // 4
                    k = v_idx % 4
                    # Part 1
                    V_full[j*7 + k][c] += val
                    # Part 2
                    V_full[j*7 + 4 + k][c + 28] += val
                    
        # W mapping
        # Strassen W: 16x28. Index w_idx = i*4 + k.
        # Full W: 28x28 (rows k*4+i).
        # Part 1 (cols 0-27): maps to C[:, 0:4].
        #   i in 0..3, k in 0..3.
        #   Full index: k*4 + i.
        # Part 2 (cols 28-55): maps to C[:, 4:7].
        #   i in 0..3, k_local in 0..3.
        #   We only need k_local in 0..2 for the output C[:, 4:6].
        #   Full index: (4 + k_local)*4 + i.
        
        for c in range(28):
            for w_idx in range(16):
                val = W_str[w_idx][c]
                if val != 0:
                    i = w_idx // 4
                    k = w_idx % 4
                    # Part 1
                    W_full[k*4 + i][c] += val
                    # Part 2
                    if k < 3: # Only k=0,1,2 map to valid output columns 4,5,6
                        W_full[(4 + k)*4 + i][c + 28] += val
                    # k=3 maps to column 7 which doesn't exist in 4x7 output, so we ignore it.
                    
        # Verify
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    total = 0
                    for c in range(total_rank):
                        total += U_full[i*4+j][c] * V_full[j*7+k][c] * W_full[k*4+i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {"U": U_full, "V": V_full, "W": W_full}
        else:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
