import numpy as np
import time

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 Strassen = 49
    # Part 2: 4 * (2x2x3) where 2x2x3 = 7 (Strassen 2x2x2) + 3 (rank-3 for 2x2x1) = 10
    # Total: 49 + 4*10 = 89
    total_rank = 49 + 40
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Strassen 4x4x4 outer terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j][r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j][r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final][r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 10
    # 2x2x3: Strassen 2x2x2 for cols 0,1 (rank 7) + rank-3 decomposition for col 2
    # Rank-3 decomposition of 2x2x1:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Using:
    # t1: A[0,0]*B[0] -> C[0]
    # t2: A[0,1]*B[1] -> C[0]  
    # t3: (A[1,0]+A[1,1])*(B[0]+B[1]) - A[1,0]*B[0] - A[1,1]*B[1]... 
    # Actually simpler: use the fact that 2x2x1 has rank 3:
    # M1 = (A[0,0]+A[1,0]) * B[0] -> C[0] + C[1]
    # M2 = (A[0,1]+A[1,1]) * B[1] -> C[0] + C[1]
    # M3 = A[1,0] * B[0] -> C[1] (subtract to fix)
    # Hmm, let me just use a known rank-3 decomposition:
    # For 2x2x1: C[i] = sum_j A[i,j] B[j]
    # t1: A[0,0]*B[0] -> C[0]
    # t2: A[0,1]*B[1] -> C[0]
    # t3: (A[1,0]+A[1,1]) * (B[0]+B[1]) -> C[1]
    # But t3 gives A[1,0]B[0] + A[1,0]B[1] + A[1,1]B[0] + A[1,1]B[1], 
    # we only want A[1,0]B[0] + A[1,1]B[1]
    # So we need to subtract A[1,0]B[1] + A[1,1]B[0]
    # This doesn't directly give rank 3.
    
    # Let me use a different known decomposition. The 2x2x1 tensor:
    # C[0] = A00*B0 + A01*B1
    # C[1] = A10*B0 + A11*B1
    # Rank 3 decomposition:
    # t1: A00*B0 -> C[0]
    # t2: A01*B1 -> C[0]
    # t3: (A10+A11)*(B0+B1) -> C[1] ... no
    # 
    # Actually, I'll use:
    # t1: (A00+A10)*B0 -> C[0]+C[1]
    # t2: (A01+A11)*B1 -> C[0]+C[1]
    # t3: -A10*B0 -> -C[1]  (to cancel the extra A10*B0 from t1 in C[1])
    # t4: -A11*B1 -> -C[1]  (to cancel the extra A11*B1 from t2 in C[1])
    # That's rank 4 again.
    
    # The correct rank-3 decomposition of 2x2x1:
    # t1: A00*B0 -> C[0]
    # t2: A11*B1 -> C[1]
    # t3: (A01+A10)*(B0+B1) -> C[0]+C[1]
    # Check: C[0] gets A00*B0 + A01*B0 + A10*B0... no that's wrong.
    
    # Let me just use rank 4 for safety and keep the original approach.
    # The 2x2x1 tensor actually has rank 4 in general (it's not reducible).
    # Wait, no. The matrix multiplication tensor for 2x2x1 (i.e., multiplying 2x2 by 2x1)
    # has rank 4. It's the 2x2x2 that has rank 7 (Strassen).
    
    # So 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11.
    # Total = 49 + 4*11 = 93. Same as before.
    
    # Let me just use the original rank 93 approach but with cleaner code.
    total_rank = 93
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    current_rank = 49  # After Part 1
    
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        # Strassen 2x2x2 for columns 4,5 of B -> columns 4,5 of C
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j][r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j][r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final][r] += cc
        
        # Schoolbook for column 6: C[i][6] = sum_j A[i][j]*B[j][6]
        # For 2x2x1: 4 terms
        for xi in range(2):
            for xj in range(2):
                r = current_rank
                current_rank += 1
                i_a = 2 * ar + xi
                j_a = 2 * ac + xj
                i_b = 2 * br + xj
                U[i_a * m + j_a][r] = 1.0
                V[i_b * p + 6][r] = 1.0
                W[6 * n + i_a][r] = 1.0
    
    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
