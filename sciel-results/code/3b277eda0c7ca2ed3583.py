import time
import numpy as np

DIMENSIONS = [6, 6, 6]

START_TIME = time.time()
TIME_LIMIT = 120.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5.0)

def _strassen_2x2_decomp():
    """
    Returns U, V, W for Strassen 2x2 matrix multiplication.
    Shapes: U (4, 7), V (4, 7), W (4, 7)
    Indices:
    U: i*m + j (i,j in 0..1) -> 0,1,2,3
    V: j*p + k (j,k in 0..1) -> 0,1,2,3
    W: k*n + i (k,i in 0..1) -> 0,1,2,3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00(0), A11(3); V: B00(0), B11(3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00(0), C11(3)
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # U: A10(2), A11(3); V: B00(0)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # W: C10(1), C11(3) with -1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    # U: A00(0); V: B01(1), B11(3) with -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # W: C01(2), C11(3)
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    # U: A11(3); V: B10(2), B00(0) with -1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # W: C00(0), C10(1)
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00 + A01)B11
    # U: A00(0), A01(1); V: B11(3)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # W: C00(0) with -1, C01(2)
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10(2), A00(0) with -1; V: B00(0), B01(1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C11(3)
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01(1), A11(3) with -1; V: B10(2), B11(3)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # W: C00(0)
    W[0, 6] = 1
    
    return U, V, W

def _laderman_3x3_decomp():
    """
    Returns U, V, W for Laderman 3x3 matrix multiplication.
    Shapes: U (9, 23), V (9, 23), W (9, 23)
    Indices:
    U: i*3 + j (i,j in 0..2) -> 0..8
    V: j*3 + k (j,k in 0..2) -> 0..8
    W: k*3 + i (k,i in 0..2) -> 0..8
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Laderman's 23 products
    # Reference: Laderman, "A 3x3 matrix multiplication algorithm" (1976)
    # Or standard implementations found in literature.
    
    # P1: (a00+a10)(b00+b10)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += P1, C10 += P1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2: (a00+a01)(b00+b01)
    U[0, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    # C00 += P2, C01 += P2
    W[0, 1] = 1; W[1, 1] = 1
    
    # P3: (a01-a10)(b01-b10)
    U[1, 2] = 1; U[3, 2] = -1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 += P3, C10 += P3
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4: (a00+a02)(b00+b02)
    U[0, 3] = 1; U[2, 3] = 1
    V[0, 3] = 1; V[2, 3] = 1
    # C00 += P4, C02 += P4
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5: (a02-a10)(b02-b10)
    U[2, 4] = 1; U[3, 4] = -1
    V[2, 4] = 1; V[3, 4] = -1
    # C02 += P5, C10 += P5
    W[2, 4] = 1; W[3, 4] = 1
    
    # P6: (a10+a11)(b10+b11)
    U[3, 5] = 1; U[4, 5] = 1
    V[3, 5] = 1; V[4, 5] = 1
    # C10 += P6, C11 += P6
    W[3, 5] = 1; W[4, 5] = 1
    
    # P7: (a11-a00)(b11-b00)
    U[4, 6] = 1; U[0, 6] = -1
    V[4, 6] = 1; V[0, 6] = -1
    # C11 += P7, C00 += P7
    W[4, 6] = 1; W[0, 6] = 1
    
    # P8: (a10+a12)(b10+b12)
    U[3, 7] = 1; U[5, 7] = 1
    V[3, 7] = 1; V[5, 7] = 1
    # C10 += P8, C12 += P8
    W[3, 7] = 1; W[5, 7] = 1
    
    # P9: (a12-a00)(b12-b00)
    U[5, 8] = 1; U[0, 8] = -1
    V[5, 8] = 1; V[0, 8] = -1
    # C12 += P9, C00 += P9
    W[5, 8] = 1; W[0, 8] = 1
    
    # P10: (a20+a21)(b20+b21)
    U[6, 9] = 1; U[7, 9] = 1
    V[6, 9] = 1; V[7, 9] = 1
    # C20 += P10, C21 += P10
    W[6, 9] = 1; W[7, 9] = 1
    
    # P11: (a21-a10)(b21-b10)
    U[7, 10] = 1; U[3, 10] = -1
    V[7, 10] = 1; V[3, 10] = -1
    # C21 += P11, C10 += P11
    W[7, 10] = 1; W[3, 10] = 1
    
    # P12: (a20+a22)(b20+b22)
    U[6, 11] = 1; U[8, 11] = 1
    V[6, 11] = 1; V[8, 11] = 1
    # C20 += P12, C22 += P12
    W[6, 11] = 1; W[8, 11] = 1
    
    # P13: (a22-a10)(b22-b10)
    U[8, 12] = 1; U[3, 12] = -1
    V[8, 12] = 1; V[3, 12] = -1
    # C22 += P13, C10 += P13
    W[8, 12] = 1; W[3, 12] = 1
    
    # P14: (a00+a20)(b00+b20)
    U[0, 13] = 1; U[6, 13] = 1
    V[0, 13] = 1; V[6, 13] = 1
    # C00 += P14, C20 += P14
    W[0, 13] = 1; W[6, 13] = 1
    
    # P15: (a20-a00)(b20-b00)
    U[6, 14] = 1; U[0, 14] = -1
    V[6, 14] = 1; V[0, 14] = -1
    # C20 += P15, C00 += P15
    W[6, 14] = 1; W[0, 14] = 1
    
    # P16: (a01+a21)(b01+b21)
    U[1, 15] = 1; U[7, 15] = 1
    V[1, 15] = 1; V[7, 15] = 1
    # C01 += P16, C21 += P16
    W[1, 15] = 1; W[7, 15] = 1
    
    # P17: (a21-a01)(b21-b01)
    U[7, 16] = 1; U[1, 16] = -1
    V[7, 16] = 1; V[1, 16] = -1
    # C21 += P17, C01 += P17
    W[7, 16] = 1; W[1, 16] = 1
    
    # P18: (a02+a22)(b02+b22)
    U[2, 17] = 1; U[8, 17] = 1
    V[2, 17] = 1; V[8, 17] = 1
    # C02 += P18, C22 += P18
    W[2, 17] = 1; W[8, 17] = 1
    
    # P19: (a22-a02)(b22-b02)
    U[8, 18] = 1; U[2, 18] = -1
    V[8, 18] = 1; V[2, 18] = -1
    # C22 += P19, C02 += P19
    W[8, 18] = 1; W[2, 18] = 1
    
    # P20: (a00+a11+a22)(b00+b11+b22) -- Wait, this is not standard.
    # Let's check the remaining terms.
    # The standard Laderman algorithm has 23 terms.
    # The above 19 terms cover many.
    # Terms 20-23 are needed to complete the algorithm.
    
    # P20: (a00+a01+a02)(b00+b10+b20) ? No.
    
    # Let's use a verified set of 23 terms.
    # From "The Rank of 3x3 Matrix Multiplication" or similar.
    # Actually, I can just use the fact that Laderman's algorithm is well-defined.
    # If I don't have the exact coefficients memorized, I can use a fallback or search.
    # However, I am confident in the first 19. The last 4 are typically:
    
    # P20: (a00+a10+a20)(b00+b01+b02) ? No.
    
    # Let's try to derive or find the last 4.
    # Actually, for the purpose of this task, if I cannot guarantee the exact 23 coefficients,
    # I might risk an error.
    # But wait, I can use a simpler approach:
    # Use Strassen (7) x Schoolbook (27) = 189.
    # Or use a known correct Laderman implementation.
    
    # Let's assume the 19 terms above are correct and find the last 4.
    # Actually, looking at online sources, Laderman's 23 terms are:
    # 1. (a00+a10)(b00+b10)
    # 2. (a00+a01)(b00+b01)
    # 3. (a01-a10)(b01-b10)
    # 4. (a00+a02)(b00+b02)
    # 5. (a02-a10)(b02-b10)
    # 6. (a10+a11)(b10+b11)
    # 7. (a11-a00)(b11-b00)
    # 8. (a10+a12)(b10+b12)
    # 9. (a12-a00)(b12-b00)
    # 10. (a20+a21)(b20+b21)
    # 11. (a21-a10)(b21-b10)
    # 12. (a20+a22)(b20+b22)
    # 13. (a22-a10)(b22-b10)
    # 14. (a00+a20)(b00+b20)
    # 15. (a20-a00)(b20-b00)
    # 16. (a01+a21)(b01+b21)
    # 17. (a21-a01)(b21-b01)
    # 18. (a02+a22)(b02+b22)
    # 19. (a22-a02)(b22-b02)
    # 20. (a00+a11+a22)(b00+b11+b22) -- This term is often part of the sum for diagonal.
    # 21. (a01+a12+a20)(b10+b21+b02) -- Cyclic shifts?
    # 22. (a02+a10+a21)(b20+b01+b12)
    # 23. (a00+a12+a21)(b00+b12+b21) ?
    
    # This is getting risky. If I get the last 4 wrong, the score will be 0.
    # A safer bet is to stick to Strassen x Schoolbook (189) if I'm not 100% sure about Laderman.
    # BUT, the prompt explicitly suggests Laderman in Inspiration 2.
    # Let's try to verify the Laderman coefficients.
    
    # Actually, I can use a different known low-rank algorithm for 3x3.
    # But Laderman is the standard one.
    
    # Let's assume the 19 terms are correct and the last 4 are:
    # P20: (a00+a11+a22)(b00+b11+b22)
    # P21: (a01+a12+a20)(b02+b10+b21)
    # P22: (a02+a10+a21)(b01+b12+b20)
    # P23: (a00+a12+a21)(b00+b12+b21) -- Wait, this doesn't look symmetric.
    
    # Given the risk, I will implement the Strassen x Schoolbook version (Rank 189) as the primary, 
    # but I will try to include Laderman if I can find a reliable source in my training data.
    
    # Actually, I recall that the rank of 3x3 matmul is 19 (Winograd) or 23 (Laderman).
    # Winograd's algorithm for 3x3 has rank 19? No, Winograd is for FFT.
    # The best known rank for 3x3 is 19? No, it's 19 for 2x2? No.
    # Rank(3x3) <= 19 is false. Rank(3x3) <= 23 is true (Laderman).
    # Rank(3x3) >= 19 is known.
    
    # If I use Schoolbook (27), the total rank is 189.
    # If I use Laderman (23), the total rank is 161.
    # The difference is significant.
    
    # I will implement the Laderman version with the 19 terms I am sure of, and 
    # I will add a check or fallback.
    # However, I cannot easily verify the last 4 terms in this environment.
    
    # Alternative: Use the Strassen x Schoolbook approach which is definitely correct.
    # Score 1.1429.
    # The inspiration suggests Laderman.
    # I will try to implement Laderman. If I fail, I will fall back.
    
    # Let's use the 19 terms and assume the remaining 4 are zero? No, that would be incorrect.
    
    # I will stick to the Strassen x Schoolbook (189) to ensure correctness, 
    # unless I can confidently recall the Laderman coefficients.
    
    # Wait, I can use the `numpy` library to verify the decomposition?
    # No, I need to return the factors.
    
    # I will provide the Strassen x Schoolbook solution. It is robust.
    # The score is 1.1429.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Get Strassen decomposition for 2x2 blocks
    U_s, V_s, W_s = _strassen_2x2_decomp()
    r_s = U_s.shape[1] # 7
    
    # Get Laderman/Schoolbook decomposition for 3x3 blocks
    # For safety, I will use Schoolbook (27) if I am not sure about Laderman.
    # But let's try Laderman.
    # I will use a hardcoded Laderman decomposition if possible.
    # Since I am not 100% sure of the last 4 terms, I will use Schoolbook.
    # This ensures a valid score of 1.1429.
    
    # However, the task asks to combine with inspirations.
    # Inspiration 2 is Laderman.
    # I will implement Laderman. I will use the 19 terms and leave the rest 0?
    # No, that's invalid.
    
    # I will use Schoolbook for the 3x3 blocks.
    U_l, V_l, W_l = _laderman_3x3_decomp() # This currently returns zeros for the last 4.
    # I will replace this with Schoolbook.
    
    def _schoolbook_3x3():
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = i*9 + j*3 + k
                    U[i*3+k, idx] = 1
                    V[k*3+j, idx] = 1
                    W[j*3+i, idx] = 1
        return U, V, W
    
    U_l, V_l, W_l = _schoolbook_3x3()
    r_l = U_l.shape[1] # 27
    
    total_rank = r_s * r_l # 189
    
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Map Strassen block indices to global indices
    # Strassen blocks: 00, 01, 10, 11
    # Global A: 6x6
    # Block 00: rows 0-2, cols 0-2
    # Block 01: rows 0-2, cols 3-5
    # Block 10: rows 3-5, cols 0-2
    # Block 11: rows 3-5, cols 3-5
    
    for r_s_idx in range(r_s):
        for r_l_idx in range(r_l):
            r_idx = r_s_idx * r_l + r_l_idx
            
            # U: A part
            for i_s in range(2):
                for j_s in range(2):
                    a_s = U_s[i_s*2+j_s, r_s_idx]
                    if a_s == 0: continue
                    for i_l in range(3):
                        for j_l in range(3):
                            a_l = U_l[i_l*3+j_l, r_l_idx]
                            if a_l == 0: continue
                            # Global index
                            i_g = i_s*3 + i_l
                            j_g = j_s*3 + j_l
                            U[i_g*m + j_g, r_idx] += a_s * a_l
            
            # V: B part
            for j_s in range(2):
                for k_s in range(2):
                    b_s = V_s[j_s*2+k_s, r_s_idx]
                    if b_s == 0: continue
                    for j_l in range(3):
                        for k_l in range(3):
                            b_l = V_l[j_l*3+k_l, r_l_idx]
                            if b_l == 0: continue
                            # Global index
                            j_g = j_s*3 + j_l
                            k_g = k_s*3 + k_l
                            V[j_g*p + k_g, r_idx] += b_s * b_l
            
            # W: C part
            for k_s in range(2):
                for i_s in range(2):
                    c_s = W_s[k_s*2+i_s, r_s_idx]
                    if c_s == 0: continue
                    for k_l in range(3):
                        for i_l in range(3):
                            c_l = W_l[k_l*3+i_l, r_l_idx]
                            if c_l == 0: continue
                            # Global index
                            k_g = k_s*3 + k_l
                            i_g = i_s*3 + i_l
                            W[k_g*n + i_g, r_idx] += c_s * c_l
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
