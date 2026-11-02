"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Strategy: Use Strassen decomposition recursively.
    # For 4x4x4: rank 49 (Strassen^2)
    # For 4x4x6: decompose as 4x4x4 + 4x4x2
    #   4x4x4: rank 49
    #   4x4x2: decompose as 4x2x2 + 4x2x2 = 2 * rank(4x2x2)
    #   4x2x2: decompose as (2x2x2 + 2x2x2) = 2 * 7 = 14
    #   So 4x4x2: rank 28
    # Total: 49 + 28 = 77
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Build Strassen 2x2x2 decomposition
    # Returns list of (u, v, w) where u, v, w are dicts
    def strassen_2x2x2():
        terms = []
        
        # M1 = (a11+a22)(b11+b22)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # M2 = (a21+a22)b11
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1}
        terms.append((u2, v2, w2))
        
        # M3 = a11(b12-b22)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # M4 = a22(b21-b11)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        terms.append((u4, v4, w4))
        
        # M5 = (a11+a12)b22
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}
        terms.append((u5, v5, w5))
        
        # M6 = (a21-a11)(b11+b12)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # M7 = (a12-a22)(b21+b22)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms
    
    # Build 2x2x1 schoolbook decomposition
    def schoolbook_2x2x1():
        terms = []
        for i in range(2):
            for j in range(2):
                u = {(i, j): 1}
                v = {(j, 0): 1}
                w = {(i, 0): 1}
                terms.append((u, v, w))
        return terms
    
    # Build 2x2x2 decomposition (Strassen)
    def decompose_2x2x2():
        return strassen_2x2x2()
    
    # Build 2x2x3 decomposition: 2x2x2 (Strassen) + 2x2x1 (schoolbook)
    def decompose_2x2x3():
        terms = []
        
        # First 2 columns: Strassen 2x2x2
        strassen_terms = strassen_2x2x2()
        for u, v, w in strassen_terms:
            terms.append((u, v, w))
        
        # Third column: schoolbook 2x2x1
        schoolbook_terms = schoolbook_2x2x1()
        for u, v, w in schoolbook_terms:
            v_new = {(b_row, b_col + 1): coeff for (b_row, b_col), coeff in v.items()}
            w_new = {(c_row, c_col + 1): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u, v_new, w_new))
        
        return terms
    
    # Build 2x2x4 decomposition: 2x2x2 (Strassen) + 2x2x2 (Strassen)
    def decompose_2x2x4():
        terms = []
        
        # First 2 columns: Strassen 2x2x2
        strassen_terms_1 = strassen_2x2x2()
        for u, v, w in strassen_terms_1:
            terms.append((u, v, w))
        
        # Next 2 columns: Strassen 2x2x2 with shifted b_col and w_col
        strassen_terms_2 = strassen_2x2x2()
        for u, v, w in strassen_terms_2:
            v_new = {(b_row, b_col + 2): coeff for (b_row, b_col), coeff in v.items()}
            w_new = {(c_row, c_col + 2): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u, v_new, w_new))
        
        return terms
    
    # Build 2x4x2 decomposition: split A into two 2x2 blocks
    # A = [A1, A2] (2x4), B = B (4x2)
    # C = A1*B1 + A2*B2 where B1 = B[0:2,:], B2 = B[2:4,:]
    def decompose_2x4x2():
        terms = []
        
        # A1*B1: 2x2x2 (Strassen)
        terms_1 = strassen_2x2x2()
        for u, v, w in terms_1:
            terms.append((u, v, w))
        
        # A2*B2: 2x2x2 (Strassen) with shifted a_col and b_row
        terms_2 = strassen_2x2x2()
        for u, v, w in terms_2:
            u_new = {(a_row, a_col + 2): coeff for (a_row, a_col), coeff in u.items()}
            v_new = {(b_row + 2, b_col): coeff for (b_row, b_col), coeff in v.items()}
            terms.append((u_new, v_new, w))
        
        return terms
    
    # Build 2x4x4 decomposition: split B into two 4x2 blocks
    # C = A*B1 + A*B2
    def decompose_2x4x4():
        terms = []
        
        # A*B1: 2x4x2
        terms_1 = decompose_2x4x2()
        for u, v, w in terms_1:
            terms.append((u, v, w))
        
        # A*B2: 2x4x2 with shifted b_col and w_col
        terms_2 = decompose_2x4x2()
        for u, v, w in terms_2:
            v_new = {(b_row, b_col + 2): coeff for (b_row, b_col), coeff in v.items()}
            w_new = {(c_row, c_col + 2): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u, v_new, w_new))
        
        return terms
    
    # Build 4x2x2 decomposition: split A into two 2x2 blocks
    # A = [A1; A2] (4x2), B = B (2x2)
    # C = A1*B + A2*B (4x2)
    def decompose_4x2x2():
        terms = []
        
        # A1*B: 2x2x2 (Strassen)
        terms_1 = strassen_2x2x2()
        for u, v, w in terms_1:
            terms.append((u, v, w))
        
        # A2*B: 2x2x2 (Strassen) with shifted a_row and w_row
        terms_2 = strassen_2x2x2()
        for u, v, w in terms_2:
            u_new = {(a_row + 2, a_col): coeff for (a_row, a_col), coeff in u.items()}
            w_new = {(c_row + 2, c_col): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u_new, v, w_new))
        
        return terms
    
    # Build 4x4x2 decomposition: split A into two 4x2 blocks
    # A = [A1, A2] (4x4), B = B (4x2)
    # C = A1*B1 + A2*B2 where B1 = B[0:2,:], B2 = B[2:4,:]
    def decompose_4x4x2():
        terms = []
        
        # A1*B1: 4x2x2
        terms_1 = decompose_4x2x2()
        for u, v, w in terms_1:
            terms.append((u, v, w))
        
        # A2*B2: 4x2x2 with shifted a_col and b_row
        terms_2 = decompose_4x2x2()
        for u, v, w in terms_2:
            u_new = {(a_row, a_col + 2): coeff for (a_row, a_col), coeff in u.items()}
            v_new = {(b_row + 2, b_col): coeff for (b_row, b_col), coeff in v.items()}
            terms.append((u_new, v_new, w))
        
        return terms
    
    # Build 4x4x4 decomposition: use Strassen recursively
    # A = [[A11, A12], [A21, A22]] (4x4), B = [[B11, B12], [B21, B22]] (4x4)
    # C = [[C11, C12], [C21, C22]] (4x4)
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def decompose_4x4x4():
        terms = []
        
        # We need to build the U, V, W contributions for each Mi
        # Each Mi is a 2x2x2 multiplication (on 2x2 blocks)
        # So we can use strassen_2x2x2() for each
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is at (0:2, 0:2), A22 is at (2:4, 2:4)
        # B11 is at (0:2, 0:2), B22 is at (2:4, 2:4)
        # C11 gets +1, C22 gets +1
        
        # Build M1: (A11+A22)(B11+B22)
        # This is like multiplying (A11+A22) by (B11+B22)
        # A11+A22 is a 4x4 matrix with A11 in top-left and A22 in bottom-right
        # B11+B22 is a 4x4 matrix with B11 in top-left and B22 in bottom-right
        # The product is a 4x4 matrix
        # C11 (top-left 2x2) gets contribution from (A11+A22) * (B11+B22) top-left
        # C22 (bottom-right 2x2) gets contribution from (A11+A22) * (B11+B22) bottom-right
        
        # Let me build this more carefully using the 2x2x2 Strassen decomposition
        # on the block level.
        
        # For each Mi, I'll construct the U, V, W contributions
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 contributes to rows 0-1, cols 0-1; A22 contributes to rows 2-3, cols 2-3
        # V: B11 contributes to rows 0-1, cols 0-1; B22 contributes to rows 2-3, cols 2-3
        # W: C11 gets +1 (rows 0-1, cols 0-1); C22 gets +1 (rows 2-3, cols 2-3)
        
        # Use strassen_2x2x2() to get the 7 terms for 2x2x2
        # Then map them to the 4x4x4 context
        
        strassen_terms = strassen_2x2x2()
        
        # M1: (A11+A22)(B11+B22)
        # A11 is at a_row in {0,1}, a_col in {0,1}
        # A22 is at a_row in {2,3}, a_col in {2,3}
        # B11 is at b_row in {0,1}, b_col in {0,1}
        # B22 is at b_row in {2,3}, b_col in {2,3}
        # C11 is at c_row in {0,1}, c_col in {0,1}
        # C22 is at c_row in {2,3}, c_col in {2,3}
        
        for u, v, w in strassen_terms:
            # u has keys (a_row, a_col) where a_row, a_col in {0,1}
            # We need to create U contributions for both A11 and A22
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar, ac)] = coeff  # A11
                u_new[(ar+2, ac+2)] = coeff  # A22
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br, bc)] = coeff  # B11
                v_new[(br+2, bc+2)] = coeff  # B22
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr, cc)] = coeff  # C11
                w_new[(cr+2, cc+2)] = coeff  # C22
            
            terms.append((u_new, v_new, w_new))
        
        # M2 = (A21+A22)B11
        # A21 is at a_row in {2,3}, a_col in {0,1}
        # A22 is at a_row in {2,3}, a_col in {2,3}
        # B11 is at b_row in {0,1}, b_col in {0,1}
        # C21 is at c_row in {2,3}, c_col in {0,1}
        
        for u, v, w in strassen_terms:
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar+2, ac)] = coeff  # A21
                u_new[(ar+2, ac+2)] = coeff  # A22
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br, bc)] = coeff  # B11
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr+2, cc)] = coeff  # C21
            
            terms.append((u_new, v_new, w_new))
        
        # M3 = A11(B12-B22)
        # A11 is at a_row in {0,1}, a_col in {0,1}
        # B12 is at b_row in {0,1}, b_col in {2,3}
        # B22 is at b_row in {2,3}, b_col in {2,3}
        # C12 is at c_row in {0,1}, c_col in {2,3}
        # C22 is at c_row in {2,3}, c_col in {2,3}
        
        for u, v, w in strassen_terms:
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar, ac)] = coeff  # A11
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br, bc+2)] = coeff  # B12
                v_new[(br+2, bc+2)] = -coeff  # B22 (with negative sign)
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr, cc+2)] = coeff  # C12
                w_new[(cr+2, cc+2)] = coeff  # C22
            
            terms.append((u_new, v_new, w_new))
        
        # M4 = A22(B21-B11)
        # A22 is at a_row in {2,3}, a_col in {2,3}
        # B21 is at b_row in {2,3}, b_col in {0,1}
        # B11 is at b_row in {0,1}, b_col in {0,1}
        # C11 is at c_row in {0,1}, c_col in {0,1}
        # C21 is at c_row in {2,3}, c_col in {0,1}
        
        for u, v, w in strassen_terms:
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar+2, ac+2)] = coeff  # A22
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br+2, bc)] = coeff  # B21
                v_new[(br, bc)] = -coeff  # B11 (with negative sign)
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr, cc)] = coeff  # C11
                w_new[(cr+2, cc)] = coeff  # C21
            
            terms.append((u_new, v_new, w_new))
        
        # M5 = (A11+A12)B22
        # A11 is at a_row in {0,1}, a_col in {0,1}
        # A12 is at a_row in {0,1}, a_col in {2,3}
        # B22 is at b_row in {2,3}, b_col in {2,3}
        # C11 is at c_row in {0,1}, c_col in {0,1} (with -1)
        # C12 is at c_row in {0,1}, c_col in {2,3}
        
        for u, v, w in strassen_terms:
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar, ac)] = coeff  # A11
                u_new[(ar, ac+2)] = coeff  # A12
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br+2, bc+2)] = coeff  # B22
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr, cc)] = -coeff  # C11 (with negative sign)
                w_new[(cr, cc+2)] = coeff  # C12
            
            terms.append((u_new, v_new, w_new))
        
        # M6 = (A21-A11)(B11+B12)
        # A21 is at a_row in {2,3}, a_col in {0,1}
        # A11 is at a_row in {0,1}, a_col in {0,1}
        # B11 is at b_row in {0,1}, b_col in {0,1}
        # B12 is at b_row in {0,1}, b_col in {2,3}
        # C22 is at c_row in {2,3}, c_col in {2,3}
        
        for u, v, w in strassen_terms:
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar+2, ac)] = coeff  # A21
                u_new[(ar, ac)] = -coeff  # A11 (with negative sign)
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br, bc)] = coeff  # B11
                v_new[(br, bc+2)] = coeff  # B12
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr+2, cc+2)] = coeff  # C22
            
            terms.append((u_new, v_new, w_new))
        
        # M7 = (A12-A22)(B21+B22)
        # A12 is at a_row in {0,1}, a_col in {2,3}
        # A22 is at a_row in {2,3}, a_col in {2,3}
        # B21 is at b_row in {2,3}, b_col in {0,1}
        # B22 is at b_row in {2,3}, b_col in {2,3}
        # C11 is at c_row in {0,1}, c_col in {0,1}
        
        for u, v, w in strassen_terms:
            u_new = {}
            for (ar, ac), coeff in u.items():
                u_new[(ar, ac+2)] = coeff  # A12
                u_new[(ar+2, ac+2)] = -coeff  # A22 (with negative sign)
            
            v_new = {}
            for (br, bc), coeff in v.items():
                v_new[(br+2, bc)] = coeff  # B21
                v_new[(br+2, bc+2)] = coeff  # B22
            
            w_new = {}
            for (cr, cc), coeff in w.items():
                w_new[(cr, cc)] = coeff  # C11
            
            terms.append((u_new, v_new, w_new))
        
        return terms
    
    # Build 4x4x6 decomposition: 4x4x4 + 4x4x2
    def decompose_4x4x6():
        terms = []
        
        # First 4 columns: 4x4x4
        terms_1 = decompose_4x4x4()
        for u, v, w in terms_1:
            terms.append((u, v, w))
        
        # Last 2 columns: 4x4x2 with shifted b_col and w_col
        terms_2 = decompose_4x4x2()
        for u, v, w in terms_2:
            v_new = {(b_row, b_col + 4): coeff for (b_row, b_col), coeff in v.items()}
            w_new = {(c_row, c_col + 4): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u, v_new, w_new))
        
        return terms
    
    # Get all terms
    all_terms = decompose_4x4x6()
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    for r, (u, v, w) in enumerate(all_terms):
        for (a_row, a_col), coeff in u.items():
            U[a_row * m + a_col][r] = coeff
        for (b_row, b_col), coeff in v.items():
            V[b_row * p + b_col][r] = coeff
        for (c_row, c_col), coeff in w.items():
            W[c_col * n + c_row][r] = coeff
    
    # Verify the decomposition
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        error += abs(val - 1.0)
        
        if error > 1e-9:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
