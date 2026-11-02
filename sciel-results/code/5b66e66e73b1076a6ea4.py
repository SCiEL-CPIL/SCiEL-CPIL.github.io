def construct():
    """
    Rank-44 tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    - Split the 4x4 B matrix into two 4x2 halves (left: cols 0-1, right: cols 2-3).
    - For each half, split the 3x4 A matrix into top 2x4 and bottom 1x4.
    - Top 2x4 * 4x2: Split into four 2x2 * 2x2 blocks. Use Strassen (rank 7) for 
      the two diagonal blocks and schoolbook (rank 4) for the two off-diagonal blocks.
      Total per half: 7 + 4 + 4 + 7 = 22.
    - Bottom 1x4 * 4x2: Schoolbook (rank 8).
    - Total per half: 22 + 8 = 30.
    - Two halves: 60. 
    
    Wait, 60 > 44. Let me re-evaluate.
    
    Alternative strategy from the current program:
    - Split A into top 2x4 and bottom 1x4.
    - Split B into left 4x2 and right 4x2.
    - For the top 2x4 * 4x2 part:
      - Split A_top into 2x2 blocks: A_TL (cols 0-1), A_TR (cols 2-3).
      - Split B_left into 2x2 blocks: B_TL (rows 0-1), B_BL (rows 2-3).
      - C_TL = A_TL * B_TL + A_TR * B_BL.
      - Use Strassen for both 2x2*2x2 multiplications: 7 + 7 = 14.
    - Similarly for the right half:
      - C_TR = A_TL * B_TR + A_TR * B_BR.
      - Use Strassen for both: 7 + 7 = 14.
    - Total for top part: 14 + 14 = 28.
    - Bottom part: 1x4 * 4x4 = 8 (schoolbook).
    - Total: 28 + 8 = 36.
    
    But the current program has rank 44. Why?
    Looking at the current code:
    - It uses Strassen for the top 2x2 * 2x2 blocks, but it seems to be doing it for each 
      2x2 block of B separately, and also including the bottom row with schoolbook.
    - Let's trace through the current code more carefully.
    
    The current code:
    - rank = 44
    - It loops over b_r in range(2) and b_c in range(2).
    - For each (b_r, b_c), it does 7 Strassen terms for the top 2x2 * 2x2.
    - Then it does 4 schoolbook terms for the bottom row.
    - So per (b_r, b_c): 7 + 4 = 11.
    - Total: 2 * 2 * 11 = 44.
    
    This is equivalent to:
    - For each 2x2 block of B (rows b_r*2 to b_r*2+1, cols b_c*2 to b_c*2+1):
      - Multiply top 2x4 of A by this 2x2 block of B using Strassen (but wait, Strassen 
        is for 2x2 * 2x2, not 2x4 * 2x2).
    
    Actually, I think the current code is treating each 2x2 block of B independently and 
    multiplying it with the corresponding 2x2 block of A. But that's not how matrix 
    multiplication works.
    
    Let me reconsider. The correct decomposition for 3x4 * 4x4:
    - C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    - C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    - C[2:3, 0:2] = A[2:3, 0:2] * B[0:2, 0:2] + A[2:3, 2:4] * B[2:4, 0:2]
    - C[2:3, 2:4] = A[2:3, 0:2] * B[0:2, 2:4] + A[2:3, 2:4] * B[2:4, 2:4]
    
    Using Strassen for the 2x2 * 2x2 multiplications:
    - C[0:2, 0:2]: 7 (Strassen) + 7 (Strassen) = 14
    - C[0:2, 2:4]: 7 + 7 = 14
    - C[2:3, 0:2]: 4 (schoolbook for 1x2 * 2x2) + 4 (schoolbook) = 8
    - C[2:3, 2:4]: 4 + 4 = 8
    Total: 14 + 14 + 8 + 8 = 44.
    
    So the current program is correct with rank 44.
    
    Can we do better? The inspiration suggests rank 36, but the code was incomplete.
    Let's try to verify if rank 36 is possible.
    
    Actually, looking at the inspiration more carefully, it seems to have made an error 
    in the bottom part calculation. The bottom part 1x4 * 4x2 has rank 8, not 4.
    
    So the current rank of 44 seems to be the correct value for this decomposition strategy.
    Let's keep the current program as is, since it's already valid and achieves rank 44.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # Strassen part: 7 terms, rows i=0,1
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m1
            col += 1
            
            # m2 = (c+d)e: A-part: A[1,0]=1, A[1,1]=1; B-part: B[0,0]=1
            U[0*m + (b_r*2+0)][col] = 0
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1   # C[1,0] += m2
            W[(b_c*2+1)*n + 1][col] = -1  # C[1,1] -= m2
            col += 1
            
            # m3 = a(f-h): A-part: A[0,0]=1; B-part: B[0,1]=1, B[1,1]=-1
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1   # C[0,1] += m3
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m3
            col += 1
            
            # m4 = d(g-e): A-part: A[1,1]=1; B-part: B[1,0]=1, B[0,0]=-1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m4
            W[(b_c*2+0)*n + 1][col] = 1   # C[1,0] += m4
            col += 1
            
            # m5 = (a+b)h: A-part: A[0,0]=1, A[0,1]=1; B-part: B[1,1]=1
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1  # C[0,0] -= m5
            W[(b_c*2+1)*n + 0][col] = 1   # C[0,1] += m5
            col += 1
            
            # m6 = (c-a)(e+f): A-part: A[1,0]=1, A[0,0]=-1; B-part: B[0,0]=1, B[0,1]=1
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m6
            col += 1
            
            # m7 = (b-d)(g+h): A-part: A[0,1]=1, A[1,1]=-1; B-part: B[1,0]=1, B[1,1]=1
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m7
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
