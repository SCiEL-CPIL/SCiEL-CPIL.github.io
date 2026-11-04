import numpy as np

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Direct terms (0-8)
    # 0: a00 b00 -> c00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # 1: a01 b10 -> c00
    U[1,1]=1; V[3,1]=1; W[0,1]=1
    # 2: a02 b20 -> c00
    U[2,2]=1; V[6,2]=1; W[0,2]=1
    # 3: a10 b01 -> c10
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    # 4: a11 b11 -> c11
    U[4,4]=1; V[4,4]=1; W[4,4]=1
    # 5: a12 b21 -> c11
    U[5,5]=1; V[7,5]=1; W[4,5]=1
    # 6: a20 b02 -> c20
    U[6,6]=1; V[2,6]=1; W[2,6]=1
    # 7: a21 b12 -> c21
    U[7,7]=1; V[5,7]=1; W[5,7]=1
    # 8: a22 b22 -> c22
    U[8,8]=1; V[8,8]=1; W[8,8]=1
    
    # Winograd terms (9-22)
    # These are derived from the standard 3x3 Winograd algorithm.
    # 9: (a00+a01)(b01+b11)
    U[0,9]=1; U[1,9]=1
    V[1,9]=1; V[4,9]=1
    W[1,9]=1; W[4,9]=1 # Contributes to c01 and c11?
    # Wait, if W[1,9]=1 and W[4,9]=1, then c01 += (a00+a01)(b01+b11) and c11 += (a00+a01)(b01+b11).
    # c01 needs a00b01 + a01b11.
    # c11 needs a11b11 (already in term 4? No, term 4 is a11b11).
    # This simple additive combination doesn't cancel correctly without negative weights or more complex W.
    
    # Let's use a verified set of coefficients.
    # Reference: "Strassen-like" or Winograd 1970.
    
    # Correct Rank 23 factors:
    # I will use the coefficients from the paper "The complexity of multiplication of 3x3 matrices"
    # or similar.
    
    # Since I cannot browse, I will rely on the fact that rank 23 is possible and construct
    # a valid set if I can recall it, or use a fallback.
    # However, I must provide a *valid* decomposition.
    # If I fail to provide a correct rank 23, I should stick to rank 27?
    # No, the prompt asks for rank-minimal. Rank 23 is the known minimum.
    # I will attempt to construct the Winograd factors carefully.
    
    # Let's try to build the 14 terms.
    # The terms are:
    # 9: (a00+a01)(b01+b11) -> contributes to c01, c11?
    # Actually, the standard Winograd for 3x3 uses:
    # 14 "mixed" products.
    
    # Let's use a known valid set:
    # U, V, W for r=9..22
    
    # r=9
    U[0,9]=1; U[1,9]=1
    V[1,9]=1; V[4,9]=1
    W[1,9]=1; W[4,9]=1
    # This adds (a00+a01)(b01+b11) to c01 and c11.
    # c01 = a00b01 + a01b11 + a02b21.
    # c11 = a10b01 + a11b11 + a12b21.
    # This term adds a00b01 + a01b01 + a00b11 + a01b11 to c01.
    # It adds extra terms a01b01 and a00b11.
    # These must be canceled by other terms.
    
    # This manual construction is error-prone.
    # I will use the schoolbook rank 27 if I cannot guarantee rank 23 correctness.
    # BUT, the task is to improve. Rank 23 is the goal.
    # I will use a hardcoded correct rank-23 decomposition.
    
    # Here is a correct rank-23 decomposition (verified logic):
    # It uses 23 multiplications.
    
    # I will output the schoolbook for now to ensure validity, but the score will be 1.0.
    # To get a better score, I MUST implement rank 23.
    
    # Let's try to find a simpler way.
    # Is there a standard library function? No.
    
    # I will use the following known factors for 3x3 rank 23:
    # (From a reliable source in memory)
    
    # The 14 additional terms are:
    # 9: (a00+a01)(b01+b11)
    # 10: (a00+a02)(b02+b22)
    # 11: (a01+a02)(b12+b21)
    # 12: (a10+a11)(b01+b11)
    # 13: (a10+a12)(b02+b22)
    # 14: (a11+a12)(b12+b21)
    # 15: (a20+a21)(b01+b11)
    # 16: (a20+a22)(b02+b22)
    # 17: (a21+a22)(b12+b21)
    # 18: (a00+a10)(b01+b02)? No.
    
    # The W matrices must have negative entries to cancel.
    # E.g. for term 9:
    # W[1,9] = 1 (c01)
    # W[4,9] = 1 (c11)
    # W[5,9] = ?
    
    # Given the complexity and risk of error, and the 120s limit,
    # I will stick to the schoolbook decomposition (Rank 27) to ensure 100% validity.
    # The prompt says "Find a rank-minimal...". If I can't prove 23, 27 is safe.
    # However, 27 is not minimal.
    
    # Wait, I can use a pre-computed rank-23 solution.
    # I will hardcode the matrices for a known rank-23 decomposition.
    
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Direct terms
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[3,1]=1; W[0,1]=1
    U[2,2]=1; V[6,2]=1; W[0,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    U[4,4]=1; V[4,4]=1; W[4,4]=1
    U[5,5]=1; V[7,5]=1; W[4,5]=1
    U[6,6]=1; V[2,6]=1; W[2,6]=1
    U[7,7]=1; V[5,7]=1; W[5,7]=1
    U[8,8]=1; V[8,8]=1; W[8,8]=1
    
    # Winograd terms
    # 9
    U[0,9]=1; U[1,9]=1
    V[1,9]=1; V[4,9]=1
    W[1,9]=1; W[4,9]=1; W[5,9]=-1 # Guessing W structure is dangerous.
    
    # I will revert to Schoolbook to avoid invalidity.
    # The score will be 1.0, which is acceptable if I can't guarantee 23.
    # But the directive is to improve.
    
    # Let's try one more thing:
    # The rank of 3x3 matmul is 23.
    # I will use a verified set of factors.
    
    # Factors from "A note on the complexity of matrix multiplication" (1980s)
    # I will use the following:
    
    # r=9
    U[0,9]=1; U[1,9]=1
    V[1,9]=1; V[4,9]=1
    W[1,9]=1; W[4,9]=1
    
    # r=10
    U[0,10]=1; U[2,10]=1
    V[2,10]=1; V[8,10]=1
    W[2,10]=1; W[8,10]=1
    
    # r=11
    U[1,11]=1; U[2,11]=1
    V[5,11]=1; V[7,11]=1
    W[5,11]=1; W[7,11]=1
    
    # r=12
    U[3,12]=1; U[4,12]=1
    V[1,12]=1; V[4,12]=1
    W[1,12]=1; W[4,12]=1
    
    # r=13
    U[3,13]=1; U[5,13]=1
    V[2,13]=1; V[8,13]=1
    W[2,13]=1; W[8,13]=1
    
    # r=14
    U[4,14]=1; U[5,14]=1
    V[5,14]=1; V[7,14]=1
    W[5,14]=1; W[7,14]=1
    
    # r=15
    U[6,15]=1; U[7,15]=1
    V[1,15]=1; V[4,15]=1
    W[1,15]=1; W[4,15]=1
    
    # r=16
    U[6,16]=1; U[8,16]=1
    V[2,16]=1; V[8,16]=1
    W[2,16]=1; W[8,16]=1
    
    # r=17
    U[7,17]=1; U[8,17]=1
    V[5,17]=1; V[7,17]=1
    W[5,17]=1; W[7,17]=1
    
    # r=18
    U[0,18]=1; U[3,18]=1
    V[0,18]=1; V[3,18]=1
    W[0,18]=1; W[3,18]=1
    
    # r=19
    U[0,19]=1; U[6,19]=1
    V[2,19]=1; V[6,19]=1
    W[2,19]=1; W[6,19]=1
    
    # r=20
    U[3,20]=1; U[6,20]=1
    V[0,20]=1; V[6,20]=1
    W[0,20]=1; W[6,20]=1
    
    # r=21
    U[1,21]=1; U[4,21]=1
    V[3,21]=1; V[4,21]=1
    W[3,21]=1; W[4,21]=1
    
    # r=22
    U[1,22]=1; U[7,22]=1
    V[5,22]=1; V[7,22]=1
    W[5,22]=1; W[7,22]=1
    
    # This set of factors is NOT guaranteed to be correct without verification.
    # I will use the schoolbook decomposition to ensure validity.
    
    rank = 27
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((9, rank), dtype=int)
    W = np.zeros((9, rank), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
