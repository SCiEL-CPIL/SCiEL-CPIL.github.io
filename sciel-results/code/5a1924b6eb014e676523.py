import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
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
                
    def build_3x3x3_laderman():
        # Laderman's 23-term decomposition for 3x3x3
        # Returns list of (u, v, w) dictionaries
        terms = []
        
        # M1-M9: Diagonal terms
        terms.append(( {(0,0):1}, {(0,0):1}, {(0,0):1} ))
        terms.append(( {(0,1):1}, {(1,0):1}, {(0,0):1} )) # Wait, M2 is a01*b10 -> c00? No.
        # Laderman terms:
        # M1 = a00 b00 -> c00
        # M2 = a01 b10 -> c00? No, a01*b10 contributes to c00? No.
        # c_ik = sum_j a_ij b_jk.
        # a01 * b10 contributes to c_00? No. i=0, j=1, k=0. c_00. Yes.
        # So M2 contributes to (0,0).
        # Let's list contributions to c_ik.
        
        # Standard Laderman 23 terms:
        # 1. a00 b00 -> c00
        # 2. a01 b10 -> c00
        # 3. a02 b20 -> c00
        # 4. a10 b01 -> c01
        # 5. a11 b11 -> c11
        # 6. a12 b21 -> c11
        # 7. a20 b02 -> c02
        # 8. a21 b12 -> c12
        # 9. a22 b22 -> c22
        
        # This is just schoolbook. Laderman uses sums.
        
        # Let's use the explicit 23 terms from a reliable source.
        # Source: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Laderman's_algorithm
        # The 23 products are:
        # M1 = a00 b00
        # M2 = a01 b10
        # M3 = a02 b20
        # M4 = a10 b01
        # M5 = a11 b11
        # M6 = a12 b21
        # M7 = a20 b02
        # M8 = a21 b12
        # M9 = a22 b22
        # M10 = (a01 + a11)(b10 - b11)
        # M11 = (a10 + a11)(b01 - b11)
        # M12 = (a02 + a12)(b20 - b21)
        # M13 = (a10 + a20)(b01 - b02)
        # M14 = (a11 + a21)(b12 - b11)
        # M15 = (a12 + a22)(b20 - b22)
        # M16 = (a00 + a01 + a02)(b10 + b11 + b12)
        # M17 = (a00 + a10 + a20)(b01 + b11 + b21)
        # M18 = (a00 + a20 + a22)(b02 + b12 + b22)
        # M19 = (a00 + a11 + a22)(b00 + b11 + b22)
        # M20 = (a00 + a01 + a11)(b00 + b10 + b11)
        # M21 = (a00 + a02 + a12)(b00 + b20 + b22)
        # M22 = (a10 + a11 + a12)(b01 + b11 + b21)
        # M23 = (a10 + a12 + a22)(b01 + b02 + b12)
        
        # Let's verify M10: (a01+a11)(b10-b11) = a01b10 - a01b11 + a11b10 - a11b11.
        # This contributes to c00 (a01b10), c01 (a01b11? No, a01b11 -> i=0,j=1,k=1 -> c01), c10 (a11b10 -> i=1,j=1,k=0 -> c10), c11 (a11b11 -> c11).
        
        # We will construct U, V, W for these 23 terms.
        
        def add_term(u_dict, v_dict, w_dict):
            terms.append((u_dict, v_dict, w_dict))

        # M1
        add_term({(0,0):1}, {(0,0):1}, {(0,0):1})
        # M2
        add_term({(0,1):1}, {(1,0):1}, {(0,0):1})
        # M3
        add_term({(0,2):1}, {(2,0):1}, {(0,0):1})
        # M4
        add_term({(1,0):1}, {(0,1):1}, {(0,1):1})
        # M5
        add_term({(1,1):1}, {(1,1):1}, {(1,1):1})
        # M6
        add_term({(1,2):1}, {(2,1):1}, {(1,1):1})
        # M7
        add_term({(2,0):1}, {(0,2):1}, {(0,2):1})
        # M8
        add_term({(2,1):1}, {(1,2):1}, {(1,2):1})
        # M9
        add_term({(2,2):1}, {(2,2):1}, {(2,2):1})
        
        # M10: (a01+a11)(b10-b11)
        # u: (0,1):1, (1,1):1
        # v: (1,0):1, (1,1):-1
        # w: (0,0):1, (0,1):-1, (1,0):1, (1,1):-1
        # Wait, the product is a single scalar M10.
        # It contributes to c00, c01, c10, c11.
        # c00 gets +M10 (from a01b10)
        # c01 gets -M10 (from a01b11)
        # c10 gets +M10 (from a11b10)
        # c11 gets -M10 (from a11b11)
        add_term({(0,1):1, (1,1):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (0,1):-1, (1,0):1, (1,1):-1})
        
        # M11: (a10+a11)(b01-b11)
        # u: (1,0):1, (1,1):1
        # v: (0,1):1, (1,1):-1
        # w: (0,1):1, (0,1):-1? No.
        # a10b01 -> c01 (+)
        # a10b11 -> c01? No, a10b11 -> i=1,j=0,k=1? No. a10 is i=1,j=0. b01 is j=0,k=1. c11.
        # So a10b01 -> c11? No. i=1, j=0, k=1. c_11.
        # a11b01 -> i=1, j=1, k=1. c_11.
        # a11b11 -> i=1, j=1, k=1. c_11.
        # Let's re-evaluate indices.
        # a_ij b_jk -> c_ik.
        # M11 = (a10+a11)(b01-b11)
        # a10 b01: i=1, j=0, k=1 -> c11. (+)
        # a10 b11: i=1, j=1, k=1 -> c11. (-)
        # a11 b01: i=1, j=1, k=1 -> c11. (+)
        # a11 b11: i=1, j=1, k=1 -> c11. (-)
        # So M11 contributes to c11: 1 - 1 + 1 - 1 = 0?
        # This suggests my Laderman terms are wrong or I'm misinterpreting.
        
        # Let's use the verification function to check.
        
        valid = True
        # Check diagonal
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    sum_val = 0
                    for u, v, w in terms:
                        if (i,j) in u and (j,k) in v and (i,k) in w:
                            sum_val += u[(i,j)] * v[(j,k)] * w[(i,k)]
                    if abs(sum_val - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if not valid:
            # Fallback to Strassen 2x2x2 + Schoolbook
            terms = []
            strassen_terms = [
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
                ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
                ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
                ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
            ]
            for u, v, w in strassen_terms:
                terms.append((u, v, w))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if i < 2 and j < 2 and k < 2:
                            continue
                        terms.append(( {(i,j):1}, {(j,k):1}, {(i,k):1} ))
                        
        return terms

    block1_terms = build_3x3x3_laderman()
    block2_terms = build_3x3x3_laderman()
    
    total_rank = len(block1_terms) + len(block2_terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for u, v, w in block1_terms:
        for (i, j), coeff_u in u.items():
            U[i * m + j][col] += coeff_u
        for (j, k), coeff_v in v.items():
            V[j * p + k][col] += coeff_v
        for (i, k), coeff_w in w.items():
            W[k * n + i][col] += coeff_w
        col += 1
        
    for u, v, w in block2_terms:
        for (i, j), coeff_u in u.items():
            U[i * m + j][col] += coeff_u
        for (j, k_prime), coeff_v in v.items():
            k = k_prime + 3
            V[j * p + k][col] += coeff_v
        for (i, k_prime), coeff_w in w.items():
            k = k_prime + 3
            W[k * n + i][col] += coeff_w
        col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
