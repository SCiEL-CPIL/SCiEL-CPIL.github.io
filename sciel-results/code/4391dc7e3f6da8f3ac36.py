import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strategy: split output into 4 column-pairs.
    # For each pair, compute a 2x4 * 4x2 -> 2x2 multiply.
    # Use a rank-12 decomposition for 2x4 * 4x2.
    # Total rank = 4 * 12 = 48.
    #
    # Rank-12 construction for 2x4 * 4x2:
    # A: 2x4 (indices 0..7: a00,a01,a02,a03,a10,a11,a12,a13)
    # B: 4x2 (indices 0..7: b00,b01,b10,b11,b20,b21,b30,b31)
    # C: 2x2 (W convention: W[k*n+i], n=2)
    #   c00 -> W[0], c10 -> W[1], c01 -> W[2], c11 -> W[3]
    #
    # We use 12 rank-1 terms. Construction via grouping:
    # Split A into left half (cols 0,1) and right half (cols 2,3)
    # Split B into top half (rows 0,1) and bottom half (rows 2,3)
    # Each sub-product is 2x2 * 2x2 -> 2x2 (rank 7 via Strassen)
    # But we share: total = 7 + 7 = 14 without sharing.
    # 
    # Better: use a direct rank-12 construction.
    # 
    # Let me define 12 bilinear forms:
    # 
    # Group A entries:
    #   A_L = [a00, a01] (row 0, cols 0-1), A_R = [a02, a03] (row 0, cols 2-3)
    #   A_L2 = [a10, a11] (row 1, cols 0-1), A_R2 = [a12, a13] (row 1, cols 2-3)
    # 
    # Group B entries:
    #   B_T = [[b00,b01],[b10,b11]] (rows 0-1), B_B = [[b20,b21],[b30,b31]] (rows 2-3)
    #
    # c = A_L @ B_T + A_R @ B_B  (each is 1x2 * 2x2 -> 1x2, but actually 2x2*2x2)
    # Wait: A is 2x4, so A_L is 2x2, B_T is 2x2. A_L @ B_T is 2x2.
    # Similarly A_R @ B_B is 2x2.
    # c = A_L @ B_T + A_R @ B_B
    #
    # Each 2x2*2x2 has rank 7 (Strassen). Total = 14.
    # To get rank 12, we need to share 2 terms between the two products.
    #
    # Alternative: use a rank-12 construction directly.
    # The rank of the 2x4x4x2 multiplication tensor is known to be 12.
    #
    # Here's a construction using 12 rank-1 tensors:
    #
    # Let me label A flat: 0=a00,1=a01,2=a02,3=a03,4=a10,5=a11,6=a12,7=a13
    # B flat: 0=b00,1=b01,2=b10,3=b11,4=b20,5=b21,6=b30,7=b31
    # W: 0=c00, 1=c10, 2=c01, 3=c11
    #
    # I'll use the following 12 terms:
    #
    # Term 0: (a00+a01+a02+a03) * (b00+b01+b10+b11+b20+b21+b30+b31)
    #   -> contributes to all 4 c entries with coefficient 1
    #
    # Actually, let me use a verified construction. I'll build it programmatically
    # by solving for a rank-12 decomposition using the known structure.
    #
    # Simpler approach: use two Strassen decompositions (rank 14) but that's worse.
    # Let me just use the rank-14 approach per block: 4 * 14 = 56. Same as before.
    #
    # Let me try rank 12 explicitly. I'll define the 12 terms:
    #
    # Using the fact that we can write:
    # c00 = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # c01 = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # c10 = a10*b00 + a11*b10 + a12*b20 + a13*b30
    # c11 = a10*b01 + a11*b11 + a12*b21 + a13*b31
    #
    # Define:
    # s0 = a00 + a02, t0 = b00 + b20
    # s1 = a01 + a03, t1 = b01 + b21
    # s2 = a10 + a12, t2 = b10 + b30
    # s3 = a11 + a13, t3 = b11 + b31
    # u0 = a00 - a02, v0 = b00 - b20
    # u1 = a01 - a03, v1 = b01 - b21
    # u2 = a10 - a12, v2 = b10 - b30
    # u3 = a11 - a13, v3 = b11 - b31
    #
    # Then:
    # a00*b00 = (s0+t0-... ) this is getting complicated.
    #
    # Let me just use the straightforward rank-14 per block (two Strassen) = 56 total.
    # But wait, maybe I can do better with a different tiling.
    #
    # Alternative tiling: split m=4 into 4 individual columns.
    # For each column j, we have a 2x1 * 1x8 -> 2x8 contribution.
    # That's 4 such contributions, each rank = 2*8 = 16 (no sharing). Total = 64. Worse.
    #
    # Let me try: split into 2x2 * 2x4 -> 2x4 blocks.
    # A_half (2x2) * B_half (2x4) -> 2x4. Rank of 2x2*2x4: 
    # This is equivalent to two 2x2*2x2 multiplications sharing the A side.
    # Rank = 7 + 7 - (shared) ... not obvious.
    #
    # I'll stick with the known rank-56 solution but clean it up.
    # Actually, let me try to find if rank < 56 is possible.
    #
    # The tensor M_{2,4,8} has rank at most 56 (Strassen tiling).
    # Lower bound: The rank is at least max(n*m, m*p, p*n) / something... 
    # Actually, the trivial lower bound is that rank >= max(n*m, m*p, p*n) is false.
    # A better lower bound: rank >= (n*m*p) / (max(n,m,p)^2) approximately.
    # 2*4*8 = 64, max dim = 8, so 64/64 = 1. Not helpful.
    #
    # Let me just go with rank 56 (the current best known construction).

    # Strassen 2x2 local factorization (rank 7).
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    num_blocks = (p // 2) * 2  # 4 column-pair blocks * 2 A-halves = 8 groups
    total_rank = num_blocks * 7  # 56

    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    group = 0
    for jj in range(p // 2):        # column-pair of output
        for half in range(2):       # which A-half / B-half
            base_r = group * 7
            group += 1

            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    if time.time() - start_time > 115:
        pass

    return {"U": U, "V": V, "W": W}
