import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Strassen 2x2 multiplication terms
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6

    # We need to map A indices, B indices, C indices.
    
    # Block 1: A_L B_11
    # A_L is A[0:2, 0:2]
    # A indices: 0, 1, 2, 3 (row-major)
    # B_11 is B[0:2, 0:2]
    # B indices: 0, 1, 4, 5
    # C is C[0:2, 0:2]
    # C indices: 0, 1, 3, 4 (W is p*n, so k*n+i. C is n*p? No C is n*p. W is p*n.
    # W[k*n+i, c]. C[i, k].
    # C[0,0] -> W[0*3+0] = 0
    # C[0,1] -> W[1*3+0] = 3
    # C[1,0] -> W[0*3+1] = 1
    # C[1,1] -> W[1*3+1] = 4
    
    # Let's define a function to run Strassen for a specific block
    def run_strassen(a_base, b_base, c_base, a_dims, b_dims):
        # a_base, b_base, c_base are offsets
        # a_dims, b_dims are (rows, cols)
        # We assume 2x2 blocks for now
        # A indices relative to base
        # a00: a_base + 0
        # a01: a_base + 1
        # a10: a_base + 2
        # a11: a_base + 3
        
        # B indices
        # b00: b_base + 0
        # b01: b_base + 1
        # b10: b_base + 2
        # b11: b_base + 3
        
        # C indices
        # c00: c_base + 0
        # c01: c_base + 1
        # c10: c_base + 2
        # c11: c_base + 3
        
        # M1
        add_term(
            [a_base+0, a_base+3], [1, 1],
            [b_base+0, b_base+3], [1, 1],
            [c_base+0, c_base+1, c_base+2, c_base+3], [1, 1, 1, 1]
        )
        # M2
        add_term(
            [a_base+2, a_base+3], [1, 1],
            [b_base+0], [1],
            [c_base+2, c_base+3], [1, 1] # C21, C22?
            # C21 = M2 + M4. C22 = M1 - M2 + M3 - M6.
            # So M2 contributes to C21 and C22.
            # Wait, C21 is index 2, C22 is index 3.
            # In C21, coeff is 1. In C22, coeff is -1.
        )
        # Correction:
        # C21 = M2 + M4 -> M2 coeff 1
        # C22 = M1 - M2 + M3 - M6 -> M2 coeff -1
        # So for M2:
        # W[c_base+2] += 1
        # W[c_base+3] += -1
        
        # Let's restart the W assignments carefully.
        # M1: C11(1), C12(1), C21(1), C22(1)
        # M2: C21(1), C22(-1)
        # M3: C12(1), C22(1)
        # M4: C11(-1), C21(1)
        # M5: C11(-1), C12(1)
        # M6: C22(-1)
        # M7: C11(1)
        
        # I already added M1. Let's fix M2.
        # I need to subtract the incorrect M2 addition and add correct.
        # Or just not use the helper for partials.
        
    # Let's just write it out explicitly for safety and speed.
    # Rank 44.
    
    # 1. A_L B_11
    # A_L: 0,1,2,3. B_11: 0,1,4,5. C_11: 0,1,3,4.
    # M1
    add_term([0,3],[1,1], [0,5],[1,1], [0,1,3,4],[1,1,1,1])
    # M2
    add_term([2,3],[1,1], [0],[1], [3,4],[1,-1])
    # M3
    add_term([0],[1], [1,5],[1,-1], [1,4],[1,1])
    # M4
    add_term([3],[1], [4,0],[1,-1], [0,3],[-1,1])
    # M5
    add_term([0,1],[1,1], [5],[1], [0,1],[-1,1])
    # M6
    add_term([2,0],[1,-1], [0,1],[1,1], [4],[-1])
    # M7
    add_term([1,3],[1,-1], [4,5],[1,1], [0],[1])
    
    # 2. A_L B_12
    # A_L: 0,1,2,3. B_12: 2,3,6,7. C_12: 2,3,5,6.
    # C_12 is C[0:2, 2:4].
    # C[0,2] -> W[2*3+0]=6? No.
    # W[k*n+i]. C[i,k].
    # C[0,2] -> W[6]. C[0,3] -> W[7].
    # C[1,2] -> W[8]. C[1,3] -> W[9].
    # Wait, C is 3x4.
    # C[0,0]=W[0], C[0,1]=W[3], C[0,2]=W[6], C[0,3]=W[9]
    # C[1,0]=W[1], C[1,1]=W[4], C[1,2]=W[7], C[1,3]=W[10]
    # C[2,0]=W[2], C[2,1]=W[5], C[2,2]=W[8], C[2,3]=W[11]
    
    # So for C_12 (top right):
    # c00 (of block) = C[0,2] = W[6]
    # c01 (of block) = C[0,3] = W[9]
    # c10 (of block) = C[1,2] = W[7]
    # c11 (of block) = C[1,3] = W[10]
    
    # M1
    add_term([0,3],[1,1], [2,7],[1,1], [6,9,7,10],[1,1,1,1])
    # M2
    add_term([2,3],[1,1], [2],[1], [7,10],[1,-1])
    # M3
    add_term([0],[1], [3,7],[1,-1], [9,10],[1,1])
    # M4
    add_term([3],[1], [6,2],[1,-1], [6,7],[-1,1])
    # M5
    add_term([0,1],[1,1], [7],[1], [6,9],[-1,1])
    # M6
    add_term([2,0],[1,-1], [2,3],[1,1], [10],[-1])
    # M7
    add_term([1,3],[1,-1], [6,7],[1,1], [6],[1])
    
    # 3. A_R B_21
    # A_R: 4,5,6,7. B_21: 8,9,12,13. C_21: 0,1,3,4.
    # B_21 is B[2:4, 0:2].
    # B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13.
    # C_21 is C[0:2, 0:2]. Same as C_11.
    
    # M1
    add_term([4,7],[1,1], [8,13],[1,1], [0,1,3,4],[1,1,1,1])
    # M2
    add_term([6,7],[1,1], [8],[1], [3,4],[1,-1])
    # M3
    add_term([4],[1], [9,13],[1,-1], [1,4],[1,1])
    # M4
    add_term([7],[1], [12,8],[1,-1], [0,3],[-1,1])
    # M5
    add_term([4,5],[1,1], [13],[1], [0,1],[-1,1])
    # M6
    add_term([6,4],[1,-1], [8,9],[1,1], [4],[-1])
    # M7
    add_term([5,7],[1,-1], [12,13],[1,1], [0],[1])
    
    # 4. A_R B_22
    # A_R: 4,5,6,7. B_22: 10,11,14,15. C_22: 2,3,5,6.
    # B_22 is B[2:4, 2:4].
    # B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15.
    # C_22 is C[0:2, 2:4]. Same as C_12.
    
    # M1
    add_term([4,7],[1,1], [10,15],[1,1], [6,9,7,10],[1,1,1,1])
    # M2
    add_term([6,7],[1,1], [10],[1], [7,10],[1,-1])
    # M3
    add_term([4],[1], [11,15],[1,-1], [9,10],[1,1])
    # M4
    add_term([7],[1], [14,10],[1,-1], [6,7],[-1,1])
    # M5
    add_term([4,5],[1,1], [15],[1], [6,9],[-1,1])
    # M6
    add_term([6,4],[1,-1], [10,11],[1,1], [10],[-1])
    # M7
    add_term([5,7],[1,-1], [14,15],[1,1], [6],[1])
    
    # 5-8. 1x2x2 products
    # A_2L: 8,9. B_11: 0,1,4,5. C_21_left: 2,5.
    # C[2,0]=W[2], C[2,1]=W[5].
    # A_2L = [u, v]. B_11 = [[p,q],[r,s]].
    # Res = [up+vr, uq+vs].
    # M1: u*p
    add_term([8],[1], [0],[1], [2],[1])
    # M2: v*r
    add_term([9],[1], [4],[1], [2],[1])
    # M3: u*q
    add_term([8],[1], [1],[1], [5],[1])
    # M4: v*s
    add_term([9],[1], [5],[1], [5],[1])
    
    # A_2L B_12
    # A_2L: 8,9. B_12: 2,3,6,7. C_21_right: 8,11.
    # C[2,2]=W[8], C[2,3]=W[11].
    add_term([8],[1], [2],[1], [8],[1])
    add_term([9],[1], [6],[1], [8],[1])
    add_term([8],[1], [3],[1], [11],[1])
    add_term([9],[1], [7],[1], [11],[1])
    
    # A_2R B_21
    # A_2R: 10,11. B_21: 8,9,12,13. C_21_left: 2,5.
    add_term([10],[1], [8],[1], [2],[1])
    add_term([11],[1], [12],[1], [2],[1])
    add_term([10],[1], [9],[1], [5],[1])
    add_term([11],[1], [13],[1], [5],[1])
    
    # A_2R B_22
    # A_2R: 10,11. B_22: 10,11,14,15. C_21_right: 8,11.
    add_term([10],[1], [10],[1], [8],[1])
    add_term([11],[1], [14],[1], [8],[1])
    add_term([10],[1], [11],[1], [11],[1])
    add_term([11],[1], [15],[1], [11],[1])
    
    return {"U": U, "V": V, "W": W}
