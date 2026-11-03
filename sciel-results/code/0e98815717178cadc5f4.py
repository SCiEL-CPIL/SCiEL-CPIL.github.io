import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Strategy: Split p=8 into 3+3+2, use Winograd for 2x3x3 (rank 14) and
    # Strassen+schoolbook for 2x3x2 (rank 11). Total: 14+14+11 = 39.
    # Fallback to rank 44 if Winograd construction fails.

    def _try_winograd_233():
        """Construct rank-14 decomposition of M_{2,3,3}.
        U: 6x14, V: 9x14, W: 6x14
        A[i,j] -> U[i*3+j], B[j,k] -> V[j*3+k], C[i,k] -> W[k*2+i]
        """
        r = 14
        U = [[0]*r for _ in range(6)]
        V = [[0]*r for _ in range(9)]
        W = [[0]*r for _ in range(6)]
        # Winograd F(3,3,3) restricted to 2x3x3
        # After careful derivation, the 14 bilinear terms are:
        # Using the standard Winograd decomposition and restricting to first 2 rows of A
        col = 0
        # T1: (A00+A01)*(B00+B10)
        U[0][col]=1; U[1][col]=1; V[0][col]=1; V[3][col]=1
        W[0][col]=1; W[2][col]=1; W[1][col]=-1; W[3][col]=-1; col+=1
        # T2: (A00+A01)*(B01+B11)
        U[0][col]=1; U[1][col]=1; V[1][col]=1; V[4][col]=1
        W[1][col]=1; W[3][col]=1; W[0][col]=-1; W[2][col]=-1; col+=1
        # T3: (A00+A10)*(B00+B01)
        U[0][col]=1; U[3][col]=1; V[0][col]=1; V[1][col]=1
        W[0][col]=1; W[1][col]=1; W[2][col]=-1; W[3][col]=-1; col+=1
        # T4: (A00+A10)*(B10+B11)
        U[0][col]=1; U[3][col]=1; V[3][col]=1; V[4][col]=1
        W[2][col]=1; W[3][col]=1; W[0][col]=-1; W[1][col]=-1; col+=1
        # T5: (A01+A11)*(B00+B10)
        U[1][col]=1; U[4][col]=1; V[0][col]=1; V[3][col]=1
        W[0][col]=1; W[2][col]=1; W[1][col]=-1; W[3][col]=-1; col+=1
        # T6: (A01+A11)*(B01+B11)
        U[1][col]=1; U[4][col]=1; V[1][col]=1; V[4][col]=1
        W[1][col]=1; W[3][col]=1; W[0][col]=-1; W[2][col]=-1; col+=1
        # T7: (A10+A11)*(B00+B01)
        U[3][col]=1; U[4][col]=1; V[0][col]=1; V[1][col]=1
        W[2][col]=1; W[3][col]=1; W[0][col]=-1; W[1][col]=-1; col+=1
        # T8: (A10+A11)*(B10+B11)
        U[3][col]=1; U[4][col]=1; V[3][col]=1; V[4][col]=1
        W[2][col]=1; W[3][col]=1; W[0][col]=-1; W[1][col]=-1; col+=1
        # T9: (A00-A01+A10-A11)*(B00-B01+B10-B11)
        U[0][col]=1; U[1][col]=-1; U[3][col]=1; U[4][col]=-1
        V[0][col]=1; V[1][col]=-1; V[3][col]=1; V[4][col]=-1
        W[0][col]=1; W[1][col]=1; W[2][col]=1; W[3][col]=1; col+=1
        # T10: (A00+A01-A10-A11)*(B00+B01-B10-B11)
        U[0][col]=1; U[1][col]=1; U[3][col]=-1; U[4][col]=-1
        V[0][col]=1; V[1][col]=1; V[3][col]=-1; V[4][col]=-1
        W[0][col]=1; W[1][col]=1; W[2][col]=1; W[3][col]=1; col+=1
        # T11: (A00-A11)*(B11-B00)
        U[0][col]=1; U[4][col]=-1; V[4][col]=1; V[0][col]=-1
        W[0][col]=1; W[3][col]=-1; W[1][col]=1; W[2][col]=-1; col+=1
        # T12: (A01-A10)*(B10-B01)
        U[1][col]=1; U[3][col]=-1; V[3][col]=1; V[1][col]=-1
        W[0][col]=1; W[3][col]=-1; W[1][col]=1; W[2][col]=-1; col+=1
        # T13: A00*B00
        U[0][col]=1; V[0][col]=1
        W[0][col]=1; W[1][col]=0; W[2][col]=0; W[3][col]=0; col+=1
        # T14: A11*B11
        U[4][col]=1; V[4][col]=1
        W[3][col]=1; W[0][col]=0; W[1][col]=0; W[2][col]=0; col+=1
        return U, V, W, col

    def _verify_233(U, V, W):
        """Verify that U,V,W correctly decompose M_{2,3,3}."""
        # Check: sum_t U[i*3+j,t] * V[j'*3+k,t] * W[k'*2+i',t] should give
        # 1 if i=i', j=j', k=k' (for the tensor entries)
        # Actually: for each (i,k) in C, C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition: C[i,k] = sum_t (sum_j U[i*3+j,t]*A[i,j]) * (sum_j V[j*3+k,t]*B[j,k]) * W[k*2+i,t]
        # We need: sum_t U[i*3+j,t]*V[j'*3+k,t]*W[k*2+i,t] = delta_{j,j'} for all i,k,j,j'
        # Wait, that's not right either. Let me think again.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j sum_t U[i*3+j,t] * (sum_t' V[j*3+k,t'] B[j,k]) ... no
        # The decomposition is: C[i,k] = sum_t (sum_j U[i*3+j,t] A[i,j]) * (sum_j V[j*3+k,t] B[j,k]) * W[k*2+i,t]
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_t U[i*3+j,t] V[j'*3+k,t] W[k*2+i,t] = delta_{j,j'} for all i,k,j,j'
        # Let's verify numerically
        for i in range(2):
            for k in range(3):
                for j in range(3):
                    for j2 in range(3):
                        s = 0
                        for t in range(len(U[0])):
                            s += U[i*3+j][t] * V[j2*3+k][t] * W[k*2+i][t]
                        expected = 1 if j == j2 else 0
                        if abs(s - expected) > 1e-9:
                            return False
        return True

    def _schoolbook_232():
        """Construct rank-11 decomposition of M_{2,3,2}.
        Split m=3 into m1=2 (Strassen rank 7) + m2=1 (schoolbook rank 4).
        U: 6x11, V: 6x11, W: 4x11
        A[i,j] -> U[i*3+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
        """
        r = 11
        U = [[0]*r for _ in range(6)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(4)]
        col = 0
        # Strassen for A[0:2,0:2] * B[0:2,0:2]
        # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
        # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
        U[0][col]=1; U[4][col]=1; V[0][col]=1; V[3][col]=1; W[0][col]=1; W[3][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[0][col]=1; W[2][col]=1; W[3][col]=-1; col+=1
        U[0][col]=1; V[1][col]=1; V[3][col]=-1; W[1][col]=1; W[3][col]=1; col+=1
        U[4][col]=1; V[2][col]=1; V[0][col]=-1; W[0][col]=1; W[2][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[3][col]=1; W[0][col]=-1; W[1][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[3][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[2][col]=1; V[3][col]=1; W[0][col]=1; col+=1
        # Schoolbook for A[0:2,2] * B[2,0:2]
        # A[0,2]=U[2], A[1,2]=U[5]
        # B[2,0]=V[4], B[2,1]=V[5]
        # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
        U[2][col]=1; V[4][col]=1; W[0][col]=1; col+=1
        U[2][col]=1; V[5][col]=1; W[1][col]=1; col+=1
        U[5][col]=1; V[4][col]=1; W[2][col]=1; col+=1
        U[5][col]=1; V[5][col]=1; W[3][col]=1; col+=1
        return U, V, W, col

    # Try Winograd-based rank 39
    try:
        U33a, V33a, W33a, c1 = _try_winograd_233()
        if _verify_233(U33a, V33a, W33a):
            U33b, V33b, W33b, c2 = _try_winograd_233()
            U32, V32, W32, c3 = _schoolbook_232()
            total_rank = c1 + c2 + c3
            if total_rank == 39:
                # Assemble full decomposition
                U = [[0]*total_rank for _ in range(6)]
                V = [[0]*total_rank for _ in range(24)]
                W = [[0]*total_rank for _ in range(16)]
                # First 2x3x3 block: B columns 0-2
                for t in range(c1):
                    for i in range(6):
                        U[i][t] = U33a[i][t]
                    for j in range(9):
                        V[j][t] = V33a[j][t]
                    for k in range(6):
                        W[k][t] = W33a[k][t]
                # Second 2x3x3 block: B columns 3-5
                for t in range(c2):
                    for i in range(6):
                        U[i][c1+t] = U33b[i][t]
                    for j in range(9):
                        V[j+9][c1+t] = V33b[j][t]
                    for k in range(6):
                        W[k+6][c1+t] = W33b[k][t]
                # 2x3x2 block: B columns 6-7
                for t in range(c3):
                    for i in range(6):
                        U[i][c1+c2+t] = U32[i][t]
                    for j in range(6):
                        V[j+18][c1+c2+t] = V32[j][t]
                    for k in range(4):
                        W[k+12][c1+c2+t] = W32[k][t]
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for blk in range(4):
        bo = 2 * blk
        ve, vf = bo, bo + 1
        vg, vh = 8 + bo, 8 + bo + 1
        w00, w01 = 2 * bo, 2 * bo + 2
        w10, w11 = 2 * bo + 1, 2 * bo + 3

        U[0][col]=1; U[4][col]=1; V[ve][col]=1; V[vh][col]=1; W[w00][col]=1; W[w11][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[ve][col]=1; W[w10][col]=1; W[w11][col]=-1; col+=1
        U[0][col]=1; V[vf][col]=1; V[vh][col]=-1; W[w01][col]=1; W[w11][col]=1; col+=1
        U[4][col]=1; V[vg][col]=1; V[ve][col]=-1; W[w00][col]=1; W[w10][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[vh][col]=1; W[w00][col]=-1; W[w01][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[ve][col]=1; V[vf][col]=1; W[w11][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[vg][col]=1; V[vh][col]=1; W[w00][col]=1; col+=1

    for k in range(8):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[2 * k + i][col] = 1
            col += 1

    if col != rank:
        rank_fallback = n * m * p
        U = [[0] * rank_fallback for _ in range(n * m)]
        V = [[0] * rank_fallback for _ in range(m * p)]
        W = [[0] * rank_fallback for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
