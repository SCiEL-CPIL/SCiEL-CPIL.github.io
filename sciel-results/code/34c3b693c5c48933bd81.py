import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def schoolbook_1x2x2():
        # 1x2 A times 2x2 B = 1x2 result. Rank 2 suffices.
        # A = [a0, a1], B = [[b00,b01],[b10,b11]], result = [c0, c1]
        # c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
        # Term 1: a0 * b00 -> c0 ; a0 * b01 -> c1
        # Term 2: a1 * b10 -> c0 ; a1 * b11 -> c1
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        # Term 0: U[0]=1 (a0), V[0]=1 (b00), W[0]=1 (c0)
        #          U[0]=1 (a0), V[1]=1 (b01), W[1]=1 (c1)
        U[0,0]=1
        V[0,0]=1  # b00
        V[1,0]=1  # b01
        W[0,0]=1  # c0
        W[1,0]=1  # c1
        # Term 1: U[1]=1 (a1), V[2]=1 (b10), W[0]=1 (c0)
        #          U[1]=1 (a1), V[3]=1 (b11), W[1]=1 (c1)
        U[1,1]=1
        V[2,1]=1  # b10
        V[3,1]=1  # b11
        W[0,1]=1  # c0
        W[1,1]=1  # c1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W

    U_parts = []
    V_parts = []
    W_parts = []

    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        for r in range(U_blk.shape[1]):
            U_part = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                U_part[idx] = U_blk[i, r]
            U_parts.append(U_part)
            V_part = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                V_part[idx] = V_blk[i, r]
            V_parts.append(V_part)
            W_part = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                W_part[idx] = W_blk[i, r]
            W_parts.append(W_part)

    # A is 3x4, split into rows: top 2 rows (0,1), bottom 2 rows (2,3) for the 2x2 blocks
    # and row 2 (index 2 in 0-based, but we use 8,9 for the single row)
    # Actually: A has 3 rows, 4 cols. U index = i*m + j where i in [0,2], j in [0,3]
    # Row 0: U indices 0,1,2,3
    # Row 1: U indices 4,5,6,7
    # Row 2: U indices 8,9,10,11
    # B has 4 rows, 5 cols. V index = j*p + k where j in [0,3], k in [0,4]
    # Row 0: V indices 0,1,2,3,4
    # Row 1: V indices 5,6,7,8,9
    # Row 2: V indices 10,11,12,13,14
    # Row 3: V indices 15,16,17,18,19
    # W index = k*n + i where k in [0,4], i in [0,2]
    # k=0: W 0,1,2
    # k=1: W 3,4,5
    # k=2: W 6,7,8
    # k=3: W 9,10,11
    # k=4: W 12,13,14

    # Split B's 5 columns into: left 2, mid 2, right 1
    # B row j, cols 0,1 -> V indices j*5+0, j*5+1
    # B row j, cols 2,3 -> V indices j*5+2, j*5+3
    # B row j, col 4    -> V index j*5+4

    # Split A's 3 rows into: top 2 rows (rows 0,1), bottom 1 row (row 2)
    # But for 2x2x2 blocks we need 2 rows of A. So:
    # A rows 0,1 (U indices 0-7) for 2x2x2 blocks
    # A row 2 (U indices 8-11) for 1x2x2 blocks

    # B rows: 4 rows. Split into top 2 (rows 0,1) and bottom 2 (rows 2,3)
    # B row 0: V 0-4, B row 1: V 5-9, B row 2: V 10-14, B row 3: V 15-19

    # For 2x2x2 blocks: A rows {0,1}, B rows {0,1} or {2,3}
    # u_idx for A rows 0,1: [0,1,4,5] for cols 0,1 (left 2)
    #                        [2,3,6,7] for cols 2,3 (mid 2)
    # v_idx for B rows 0,1: [0,1,5,6] for cols 0,1 (left 2)
    #                        [2,3,7,8] for cols 2,3 (mid 2)
    #                        [4,9]     for col 4 (right 1)
    # v_idx for B rows 2,3: [10,11,15,16] for cols 0,1
    #                        [12,13,17,18] for cols 2,3
    #                        [14,19]     for col 4

    # W indices for output columns:
    # k=0,1 (left 2): W 0,1,2,3,4,5
    # k=2,3 (mid 2): W 6,7,8,9,10,11
    # k=4 (right 1): W 12,13,14

    # Block 1: A rows 0,1 x B rows 0,1 x cols 0,1 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0,1,4,5], [0,1,5,6], [0,3,1,4])

    # Block 2: A rows 0,1 x B rows 0,1 x cols 2,3 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0,1,4,5], [2,3,7,8], [6,9,7,10])

    # Block 3: A rows 0,1 x B rows 0,1 x col 4 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_blk, V_blk, W_blk, [0,1,4,5], [4,9], [12,13])

    # Block 4: A rows 0,1 x B rows 2,3 x cols 0,1 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0,1,4,5], [10,11,15,16], [0,3,1,4])

    # Block 5: A rows 0,1 x B rows 2,3 x cols 2,3 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0,1,4,5], [12,13,17,18], [6,9,7,10])

    # Block 6: A rows 0,1 x B rows 2,3 x col 4 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_blk, V_blk, W_blk, [0,1,4,5], [14,19], [12,13])

    # Block 7: A row 2 x B rows 0,1 x cols 0,1 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8,9], [0,1,5,6], [2,5])

    # Block 8: A row 2 x B rows 0,1 x cols 2,3 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8,9], [2,3,7,8], [8,11])

    # Block 9: A row 2 x B rows 0,1 x col 4 (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(U_blk, V_blk, W_blk, [8,9], [4,9], [14])

    # Block 10: A row 2 x B rows 2,3 x cols 0,1 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8,9], [10,11,15,16], [2,5])

    # Block 11: A row 2 x B rows 2,3 x cols 2,3 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8,9], [12,13,17,18], [8,11])

    # Block 12: A row 2 x B rows 2,3 x col 4 (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(U_blk, V_blk, W_blk, [8,9], [14,19], [14])

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    def verify(U, V, W, tol=1e-7):
        r = U.shape[1]
        # Vectorized verification
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                    if abs(acc - 1.0) > tol:
                        ok = False
                        return ok
        return ok

    if not verify(U, V, W):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by checking if any column is redundant
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]

    def fast_verify(U, V, W, tol=1e-7):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                    if abs(acc - 1.0) > tol:
                        return False
        return True

    for attempt in range(current_rank):
        if time.time() > deadline - 15:
            break
        reduced = False
        for col in range(current_rank):
            if time.time() > deadline - 15:
                break
            keep = [c for c in range(current_rank) if c != col]
            U_new = current_U[:, keep]
            V_new = current_V[:, keep]
            W_new = current_W[:, keep]
            if fast_verify(U_new, V_new, W_new):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                reduced = True
                break
        if not reduced:
            break

    if not fast_verify(current_U, current_V, current_W):
        current_U = U
        current_V = V
        current_W = W

    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
