import time
import numpy as np

def construct():
    """
    Rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses three independent rank-7 Strassen blocks (columns 0-1, 2-3, 4-5)
    and a rank-4 schoolbook block (column 6).
    """
    t0 = time.time()
    DEADLINE = 150.0
    n, m, p = 2, 2, 7
    rank = 25

    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)

    def strassen_block(b0, b1, s):
        # b0 = col index of B's first column, b1 = col index of B's second column
        # s = starting rank column
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: row0col0=b0, row0col1=b1, row1col0=7+b0, row1col1=7+b1
        # C indices: row0col0=2*b0, row1col0=2*b0+1, row0col1=2*b1, row1col1=2*b1+1

        # M1 = (A00+A11)*(B00+B11)
        U[0, s] = 1; U[3, s] = 1
        V[b0, s] = 1; V[7+b1, s] = 1
        W[2*b0, s] = 1; W[2*b1+1, s] = 1

        # M2 = (A10+A11)*B00
        U[2, s+1] = 1; U[3, s+1] = 1
        V[b0, s+1] = 1
        W[2*b0+1, s+1] = 1; W[2*b1+1, s+1] = -1

        # M3 = A00*(B01-B11)
        U[0, s+2] = 1
        V[b1, s+2] = 1; V[7+b1, s+2] = -1
        W[2*b1, s+2] = 1; W[2*b1+1, s+2] = 1

        # M4 = A11*(B10-B00)
        U[3, s+3] = 1
        V[7+b0, s+3] = 1; V[b0, s+3] = -1
        W[2*b0, s+3] = 1; W[2*b0+1, s+3] = 1

        # M5 = (A00+A01)*B11
        U[0, s+4] = 1; U[1, s+4] = 1
        V[7+b1, s+4] = 1
        W[2*b0, s+4] = -1; W[2*b1, s+4] = 1

        # M6 = (A10-A00)*(B00+B01)
        U[2, s+5] = 1; U[0, s+5] = -1
        V[b0, s+5] = 1; V[b1, s+5] = 1
        W[2*b1+1, s+5] = 1

        # M7 = (A01-A11)*(B10+B11)
        U[1, s+6] = 1; U[3, s+6] = -1
        V[7+b0, s+6] = 1; V[7+b1, s+6] = 1
        W[2*b0, s+6] = 1

    # Three Strassen blocks for column pairs (0,1), (2,3), (4,5)
    strassen_block(0, 1, 0)
    strassen_block(2, 3, 7)
    strassen_block(4, 5, 14)

    # Schoolbook for column 6 (rank 4)
    k = 6
    U[0, 21] = 1; V[k, 21] = 1; W[2*k, 21] = 1       # A00*B0,6 -> C0,6
    U[1, 22] = 1; V[7+k, 22] = 1; W[2*k, 22] = 1     # A01*B1,6 -> C0,6
    U[2, 23] = 1; V[k, 23] = 1; W[2*k+1, 23] = 1     # A10*B0,6 -> C1,6
    U[3, 24] = 1; V[7+k, 24] = 1; W[2*k+1, 24] = 1   # A11*B1,6 -> C1,6

    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False

    if not valid:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
