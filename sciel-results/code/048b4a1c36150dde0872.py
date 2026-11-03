import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2
    # Local A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # Local B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # Local C indices (W convention W[k*n+i], n=2): 
    #   0=c00, 1=c10, 2=c01, 3=c11

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

    # Total rank: 8 blocks * 7 = 56
    total_rank = 56

    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    group = 0
    for jj in range(p // 2):        # 4 column-pair blocks of output (p=8 -> 4 pairs)
        for half in range(2):       # 2 A-halves / B-halves (m=4 -> 2 halves of 2)
            base_r = group * 7
            group += 1

            # Map local 2x2 A indices to global 2x4 A indices
            # A is 2x4, flat index = row*m + col
            # half=0: cols 0,1; half=1: cols 2,3
            a_map = [
                0*m + (2*half + 0),  # a00 or a02
                0*m + (2*half + 1),  # a01 or a03
                1*m + (2*half + 0),  # a10 or a12
                1*m + (2*half + 1),  # a11 or a13
            ]

            # Map local 2x2 B indices to global 4x2 B indices
            # B is 4x2, flat index = row*p + col
            # half=0: rows 0,1; half=1: rows 2,3
            # jj=0: cols 0,1; jj=1: cols 2,3; etc.
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]

            # Map local 2x2 C indices to global 2x2 C indices (W convention)
            # C is 2x2 per block, flat index = col*n + row (W[k*n+i], k=col, i=row)
            # Actually W convention: W[k*n+i] where k is column index, i is row index
            # For 2x2 block at columns (2*jj, 2*jj+1):
            #   c00 -> (2*jj)*n + 0
            #   c10 -> (2*jj)*n + 1
            #   c01 -> (2*jj+1)*n + 0
            #   c11 -> (2*jj+1)*n + 1
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 0,
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

    # Convert to numpy arrays for cleaner output
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)

    if time.time() - start_time > 115:
        pass

    return {"U": U, "V": V, "W": W}
