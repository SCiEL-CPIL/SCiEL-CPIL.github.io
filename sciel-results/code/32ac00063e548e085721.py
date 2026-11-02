import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # The current approach uses a rank-56 Strassen tiling.
    # We want to improve this. The rank of the (n, m, p) multiplication tensor
    # is at most n*p = 16 (by fixing j, we get 4 outer products of size 2x8,
    # but they are not independent in the tensor sense).
    #
    # Actually, let's think about this differently.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed i and k, this is a linear combination of products A[i,j]*B[j,k].
    #
    # A trivial decomposition has rank m*n*p = 4*2*8 = 64 (one term per (i,j,k) triple).
    # The current code achieves rank 56.
    #
    # Can we do better?
    #
    # Consider the structure: n=2 is small.
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    #
    # For each k, we have two linear forms in the A's and B's.
    #
    # Let's try a rank-32 construction.
    # For each j (4 values), we have A[0,j] and A[1,j] as coefficients,
    # and B[j,0], B[j,1], ..., B[j,7] as coefficients.
    #
    # The product A[0,j]*B[j,k] contributes to C[0,k].
    # The product A[1,j]*B[j,k] contributes to C[1,k].
    #
    # So for each j, we have 8 products for row 0 and 8 products for row 1,
    # but these are independent. That gives 4*8 + 4*8 = 64 terms.
    #
    # To reduce rank, we need to share terms between different j or k values.
    #
    # Let's use a block approach. Group the 4 j-values into pairs.
    # Group 1: j=0,1; Group 2: j=2,3.
    #
    # For Group 1 (j=0,1):
    # C[0,k] += A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] += A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This is a 2x2 matrix multiplication:
    # [C[0,k], C[1,k]]^T += [A[0,0], A[0,1]; A[1,0], A[1,1]] * [B[0,k], B[1,k]]^T
    #
    # Using Strassen for this 2x2 multiplication (rank 7), we get 7 terms per k.
    # But k ranges over 8 values, so that's 7*8 = 56 terms for Group 1.
    # Similarly for Group 2, another 56 terms. Total 112, which is worse.
    #
    # Instead, let's treat the entire 4 j-values at once.
    # C[0,:] = A[0,:] * B (1x4 times 4x8)
    # C[1,:] = A[1,:] * B (1x4 times 4x8)
    #
    # This is a 2x8 matrix C = (2x4) * (4x8).
    #
    # Let's use a rank-16 construction.
    # For each k in 0..7, C[:,k] = A * B[:,k] (2x4 times 4x1 = 2x1).
    # This is a 2x1 matrix, which can be written as a sum of 2 rank-1 terms.
    # So for each k, we need 2 terms, giving 8*2 = 16 terms total.
    #
    # But wait, the rank of the tensor is not simply the sum of ranks of slices.
    # Let's construct it explicitly.
    #
    # Term (k, i) for i=0,1 and k=0..7:
    # M_{k,i} = (sum_j A[i,j] * something) * (sum_j B[j,k] * something)
    #
    # This doesn't work directly because A[i,j] and B[j,k] are coupled through j.
    #
    # Let's use the following rank-16 construction:
    # For each k in 0..7 and each i in 0,1:
    # M_{k,i} = A[i,:] * B[:,k]  (dot product)
    #
    # But M_{k,i} is a scalar, not a product of two linear forms.
    #
    # Actually, the tensor T_{i,j,k} = delta_{i,k} (no, that's not right).
    # T_{i,j,k} = 1 if the indices match the multiplication pattern.
    #
    # Let me just keep the current rank-56 solution since I can't find a lower-rank
    # construction quickly. The directive says to stay in the same family.
    #
    # However, let's try to optimize the current code for speed and clarity.

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

    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
