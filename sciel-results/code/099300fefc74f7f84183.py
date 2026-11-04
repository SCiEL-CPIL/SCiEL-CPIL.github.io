import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 44
    
    U = np.zeros((n * m, rank))  # (8, 44)
    V = np.zeros((m * p, rank))  # (32, 44)
    W = np.zeros((p * n, rank))  # (16, 44)
    
    # 4 sub-problems: (j_group, k_pair)
    # j_group: 0 or 1 (which 2x2 block of A and which 2 rows of B)
    # k_pair: 0 or 1 (which pair of output columns)
    # Each sub-problem: A_j (2x2) * B_{j, k_pair} (2x4) -> C_{j, k_pair} (2x4)
    
    # Rank-11 decomposition for 2x2 * 2x4
    # Local A indices: 0=a, 1=b, 2=c, 3=d (row-major 2x2)
    # Local B indices: 0=e, 1=f, 2=g, 3=h, 4=i, 5=j, 6=k, 7=l (row-major 2x4)
    # Local C indices: 0=C00, 1=C01, 2=C02, 3=C03, 4=C10, 5=C11, 6=C12, 7=C13
    
    # 11 bilinear forms (u, v, w) where w is C contribution
    # Form: (u_coeff, v_coeff, c_coeff)
    # u_coeff: 4 values for A[0..3]
    # v_coeff: 8 values for B[0..7]
    # c_coeff: 8 values for C[0..7]
    
    forms = [
        # M1: (a+b)(e+i) -> C00 + C10
        ([1, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0]),
        # M2: (a+b)(f+j) -> C01 + C11
        ([1, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0]),
        # M3: (a+b)(g+k) -> C02 + C12
        ([1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0]),
        # M4: (a+b)(h+l) -> C03 + C13
        ([1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]),
        # M5: a(e-i) -> C00, -C10
        ([1, 0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0]),
        # M6: b(e-i) -> (not used in C, but needed? Let me recheck)
        # Actually M6 = b(e-i) doesn't appear in C. Let me re-derive.
        # C00 = ae + bi = (a+b)(e+i) - a(e-i) - b(e-i) + a(e-i) + b(e-i)... 
        # Let me redo: C00 = ae + bi. M1 = (a+b)(e+i) = ae + ai + be + bi. 
        # So ae + bi = M1 - ai - be. Hmm, that's not right.
        # Let me use a different decomposition.
    ]
    
    # Let me re-derive the rank-11 decomposition more carefully.
    # 2x2 * 2x4: A=[[a,b],[c,d]], B=[[e,f,g,h],[i,j,k,l]]
    # C = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
    
    # Standard approach: split B into left 2 cols and right 2 cols? No, that gives 2x2*2x2 twice.
    # Better: use the fact that we can factor the A side.
    
    # Let me use a known rank-11 scheme:
    # M1 = (a+b)(e+i)
    # M2 = (a+b)(f+j)  
    # M3 = (a+b)(g+k)
    # M4 = (a+b)(h+l)
    # M5 = a(e-i)
    # M6 = b(e-i)
    # M7 = c(e-i)
    # M8 = d(e-i)
    # M9 = (c-d)(f+j)
    # M10 = (c-d)(g+k)
    # M11 = (c-d)(h+l)
    
    # C00 = M1 - M5 - M6 + M5 = M1 - M6... no
    # C00 = ae + bi
    # M1 = ae + ai + be + bi
    # M5 = ae - ai
    # M6 = be - bi
    # M1 - M5 - M6 = ae + ai + be + bi - ae + ai - be + bi = 2ai + 2bi... wrong
    
    # Let me try: C00 = ae + bi
    # M1 = (a+b)(e+i) = ae + ai + be + bi
    # M5 = a(e-i) = ae - ai
    # M6 = b(e-i) = be - bi
    # M1 - M5 = ai + be + bi... no
    # M1 - M5 - M6 = ae+ai+be+bi - ae+ai - be+bi = 2ai + 2bi... still wrong
    
    # I think the correct decomposition is:
    # C00 = M1 - M5 where M1=(a+b)(e+i), M5=a(e-i)
    # M1 - M5 = ae+ai+be+bi - ae+ai = 2ai + be + bi... still not right
    
    # Let me just verify numerically with a simple approach.
    # Actually, the known rank for 2x2*2x4 is 11. Let me use a verified scheme.
    
    # From literature, one rank-11 scheme:
    # M1 = (a+c)(e+i)  -- no, this mixes rows differently
    
    # Let me try a different grouping. Split the 4 columns of B into 2 groups of 2.
    # B1 = [[e,f],[i,j]], B2 = [[g,h],[k,l]]
    # C1 = A*B1 (2x2), C2 = A*B2 (2x2)
    # Each 2x2*2x2 has rank 7, but they share the A factor.
    # Total rank = 7 + 7 - overlap. The overlap is the rank of the shared A coefficients.
    
    # Actually, for two 2x2*2x2 multiplications sharing A:
    # C1 = A*B1, C2 = A*B2
    # We can compute A*B1 and A*B2 separately with Strassen, but that's 14.
    # To get 11, we need to share 3 of the Strassen terms.
    
    # Strassen for 2x2:
    # M1 = (a+d)(e+i)
    # M2 = (c+d)(e)
    # M3 = a(f-i)
    # M4 = d(j-c)
    # M5 = (a+b)(i)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(j+i)
    
    # For B1: e,f,g,h -> e,f,i,j (using e,f for B1 cols 0,1; g,h for B1 cols 2,3... wait)
    # B1 = [[e,f],[i,j]] so B1[0,0]=e, B1[0,1]=f, B1[1,0]=i, B1[1,1]=j
    # B2 = [[g,h],[k,l]] so B2[0,0]=g, B2[0,1]=h, B2[1,0]=k, B2[1,1]=l
    
    # Strassen for C1 = A*B1:
    # S1 = (a+d)(e+i), S2 = (c+d)e, S3 = a(f-i), S4 = d(j-c), S5 = (a+b)i, S6 = (c-a)(e+f), S7 = (b-d)(j+i)
    # C1[0,0] = S1 + S4 - S5 + S7, C1[0,1] = S3 + S5, C1[1,0] = S2 + S4, C1[1,1] = S1 - S2 + S3 + S6
    
    # Strassen for C2 = A*B2:
    # T1 = (a+d)(g+k), T2 = (c+d)g, T3 = a(h-k), T4 = d(l-c), T5 = (a+b)k, T6 = (c-a)(g+h), T7 = (b-d)(l+k)
    # C2[0,0] = T1 + T4 - T5 + T7, C2[0,1] = T3 + T5, C2[1,0] = T2 + T4, C2[1,1] = T1 - T2 + T3 + T6
    
    # To share, note that S1 and T1 both have (a+d). We can factor:
    # Let P = a+d, Q = c+d, R = c-a, S = b-d
    # S1 = P(e+i), T1 = P(g+k) -> share P
    # S2 = Q*e, T2 = Q*g -> share Q
    # S6 = R(e+f), T6 = R(g+h) -> share R
    # S7 = S(j+i), T7 = S(l+k) -> share S
    # S3 = a(f-i), T3 = a(h-k) -> share a
    # S4 = d(j-c), T4 = d(l-c) -> share d
    # S5 = (a+b)i, T5 = (a+b)k -> share (a+b)
    
    # So we can precompute: P=a+d, Q=c+d, R=c-a, S=b-d, a, d, a+b
    # That's 7 shared factors.
    # Then: P(e+i), P(g+k), Q*e, Q*g, R(e+f), R(g+h), S(j+i), S(l+k), a(f-i), a(h-k), d(j-c), d(l-c), (a+b)i, (a+b)k
    # That's 14 products, but we can combine some.
    
    # Actually, the known result is rank 11. Let me just use a verified implementation.
    
    # I'll use the approach from the current program but with a better block structure.
    # For now, let me stick with rank 56 which is known to work, and try to optimize.
    
    # Actually, let me just return the current working solution with rank 56.
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    s_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    t_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    c_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(7):
        s = s_coeffs[t]
        tc = t_coeffs[t]
        c = c_coeffs[t]
        
        for j in range(2):
            for k in range(4):
                r = t * 8 + j * 4 + k
                
                idx = 0 * 4 + j
                U[idx, r] += s[0]
                idx = 0 * 4 + (j + 2)
                U[idx, r] += s[1]
                idx = 1 * 4 + j
                U[idx, r] += s[2]
                idx = 1 * 4 + (j + 2)
                U[idx, r] += s[3]
                
                idx = j * 8 + k
                V[idx, r] += tc[0]
                idx = j * 8 + (k + 4)
                V[idx, r] += tc[1]
                idx = (j + 2) * 8 + k
                V[idx, r] += tc[2]
                idx = (j + 2) * 8 + (k + 4)
                V[idx, r] += tc[3]
                
                idx = k * 2 + 0
                W[idx, r] += c[0]
                idx = (k + 4) * 2 + 0
                W[idx, r] += c[1]
                idx = k * 2 + 1
                W[idx, r] += c[2]
                idx = (k + 4) * 2 + 1
                W[idx, r] += c[3]
                
                if time.time() - start_time > timeout - 5:
                    break
            if time.time() - start_time > timeout - 5:
                break
        if time.time() - start_time > timeout - 5:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
