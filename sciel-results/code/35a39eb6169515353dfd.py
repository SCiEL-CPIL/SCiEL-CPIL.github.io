"""Structural rewrite: block-based Strassen for 2x2 x 2x14, rank 49."""
import time

DIMENSIONS = [2, 2, 14]

def _strassen_2x2():
    """
    Return U, V, W for 2x2x2 Strassen (rank 7).
    U shape (4,7): U[i*2+j, t]
    V shape (4,7): V[j*2+k, t]
    W shape (4,7): W[k*2+i, t]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # P1 = (A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[2][0]=1; W[1][0]=-1; W[3][0]=-1

    # P2 = (A10+A11)B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1

    # P3 = A00(B10-B11)
    U[0][2]=1
    V[2][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1

    # P4 = A11(B01-B00)
    U[3][3]=1
    V[1][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1

    # P5 = (A00+A01)B11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[1][4]=1; W[3][4]=1

    # P6 = (A10-A00)(B00+B01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1

    # P7 = (A01-A11)(B10+B11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1

    return U, V, W

def _verify_block(U, V, W):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for t in range(7):
                    val += U[i*2+j][t] * V[j*2+k][t] * W[k*2+i][t]
                if val != 1:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 100.0

    n, m, p = 2, 2, 14
    U_blk, V_blk, W_blk = _strassen_2x2()
    
    if not _verify_block(U_blk, V_blk, W_blk):
        # Fallback: schoolbook
        rank = 56
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(28)]
        W = [[0]*rank for _ in range(28)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    num_blocks = p // m  # 7
    rank = 7 * num_blocks  # 49

    U = [[0]*rank for _ in range(n*m)]   # 4 x 49
    V = [[0]*rank for _ in range(m*p)]   # 28 x 49
    W = [[0]*rank for _ in range(p*n)]   # 28 x 49

    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            for idx in range(n*m):
                U[idx][col] = U_blk[idx][t]
            for idx in range(m*m):
                V[idx + b*m*p // m // m * m * p // m % p]  # need proper indexing
            # Proper indexing:
            for j in range(m):
                for k in range(m):
                    V[j*p + b*m + k][col] = V_blk[j*m + k][t]
            for k in range(m):
                for i in range(n):
                    W[(b*m + k)*n + i][col] = W_blk[k*n + i][t]

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    # Fallback
                    rank2 = 56
                    U2 = [[0]*rank2 for _ in range(4)]
                    V2 = [[0]*rank2 for _ in range(28)]
                    W2 = [[0]*rank2 for _ in range(28)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U2[i2*m+j2][col] = 1
                                V2[j2*p+k2][col] = 1
                                W2[k2*n+i2][col] = 1
                                col += 1
                    return {'U': U2, 'V': V2, 'W': W2}

    return {'U': U, 'V': V, 'W': W}
