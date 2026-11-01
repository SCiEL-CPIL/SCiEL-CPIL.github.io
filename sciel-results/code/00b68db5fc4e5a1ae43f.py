"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    # Decompose 2x12 as 4 independent 2x3 blocks
    # Each 2x3 block uses rank-5 decomposition
    # Total rank = 4 * 5 = 20
    
    # For each block b (0..3), we multiply A (2x2) by B_block (2x3)
    # A indices: i (0,1) row, j (0,1) col -> U index = i*m + j = i*2+j
    # B_block indices: j (0,1) row, l (0,1,2) col -> V index = j*p + (b*3+l) = j*12 + b*3+l
    # Output: i (0,1) row, l_global (0..11) col -> W index = l_global*n + i = (b*3+l)*2 + i
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 4 x 20
    V = [[0] * rank for _ in range(m * p)]  # 24 x 20
    W = [[0] * rank for _ in range(p * n)]  # 24 x 20
    
    # Rank-5 decomposition for 2x2 * 2x3:
    # Outputs: C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    # 
    # Term 0: A[0,0] * B[0,0] -> C[0,0]
    # Term 1: A[0,1] * B[1,1] -> C[0,1]
    # Term 2: A[1,0] * B[0,2] -> C[1,2]
    # Term 3: A[1,1] * B[1,0] -> C[1,0]
    # Term 4: (A[0,0]+A[1,1]+A[0,1]+A[1,0]) * (B[0,1]+B[1,2]+B[0,2]+B[1,1])
    #          -> C[0,1] + C[1,2] + C[0,2] + C[1,1]
    
    # Verify: 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   From t0: A[0,0]*B[0,0]
    #   From t4: contributes A[0,0]*B[1,0]? No...
    
    # Let me use a verified decomposition:
    # t0: U=A00, V=B00, W=C00
    # t1: U=A01, V=B10, W=C01  
    # t2: U=A10, V=B01, W=C11
    # t3: U=A11, V=B11, W=C11
    # t4: U=A00+A11, V=B01+B10, W=C01+C11
    
    # Check C00 = A00*B00 + A01*B10
    #   t0 gives A00*B00, t1 gives A01*B10 -> C00 needs both. But t1's W is C01 not C00.
    
    # This is tricky. Let me use the actual known decomposition.
    # 
    # Known: M(2,2,2,3)=5. One explicit decomposition:
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = AB
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # 5 terms:
    # T1 = a*e  -> contributes to C00
    # T2 = b*h  -> contributes to C00
    # T3 = c*g  -> contributes to C12
    # T4 = d*i  -> contributes to C11
    # T5 = (a+b+c+d)*(e+f+g+h+i+j) - ... 
    #
    # Actually simpler verified decomposition:
    # T1 = a*e, T2 = b*f, T3 = c*g, T4 = d*h, T5 = (a+d+b+c)*(e+h+f+i+g+j)
    # This doesn't work either.
    
    # Let me just use the schoolbook for each 2x3 block (rank 6) to be safe,
    # giving total rank 24, same as before. But the task asks for improvement.
    
    # Alternative: use rank-5 for 2x3. The correct decomposition:
    # For 2x2 * 2x3, one rank-5 decomposition is:
    # t1: (a)(e) -> C00
    # t2: (b)(h) -> C00  
    # t3: (c)(g) -> C12
    # t4: (d)(i) -> C11
    # t5: (a+b+c+d)(e+f+g+h+i+j) with appropriate W coefficients
    
    # W for t5: C00+C01+C02+C10+C11+C12 minus what's already covered
    # This gets complicated. Let me just implement and verify numerically.
    
    # For now, use a safe approach: implement 4 blocks of rank-6 (schoolbook) = rank 24
    # But try to find rank-5 per block.
    
    # Actually, let me just hardcode a verified rank-5 decomposition.
    # After careful derivation, here's one that works:
    
    # For block with A=[[a,b],[c,d]] and B=[[e,f,g],[h,i,j]]:
    # C00 = ae + bh
    # C01 = af + bi  
    # C02 = ag + bj
    # C10 = ce + dh
    # C11 = cf + di
    # C12 = cg + dj
    
    # Decomposition:
    # t1: U=a, V=e, W=C00
    # t2: U=b, V=h, W=C00
    # t3: U=c, V=g, W=C12
    # t4: U=d, V=i, W=C11
    # t5: U=a+b+c+d, V=e+f+g+h+i+j, W=C00+C01+C02+C10+C11+C12
    
    # Verify C00: t1 gives ae, t2 gives bh, t5 gives (a+b+c+d)(e+f+g+h+i+j)
    #   = ae+af+ag+ah+be+bf+bg+bh+ce+cf+cg+ch+de+df+dg+dh+ei+ei... 
    # This is way too many terms. The W coefficients must be negative to cancel.
    
    # This approach with a single "sum" term doesn't work cleanly for 2x3.
    
    # Let me just use schoolbook rank 6 per block = total rank 24 for safety.
    # The improvement from 24 to 20 requires a valid rank-5 per block.
    
    # For safety and correctness, I'll use rank 24 (schoolbook).
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
