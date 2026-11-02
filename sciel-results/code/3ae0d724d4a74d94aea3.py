"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 2, 4, 7

    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(t_val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}

    def build_rank48():
        rank = 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0

        # Strassen 2x2x2 for a 2x2 A block, 2x2 B block -> 2x2 C block
        def strassen_222(i0, j0, k0):
            nonlocal col
            a, b = i0*m+j0, i0*m+j0+1
            c_, d = (i0+1)*m+j0, (i0+1)*m+j0+1
            e, f = j0*p+k0, j0*p+k0+1
            g, h = (j0+1)*p+k0, (j0+1)*p+k0+1
            c00, c01 = k0*n+i0, (k0+1)*n+i0
            c10, c11 = k0*n+i0+1, (k0+1)*n+i0+1

            U[a,col]+=1; U[d,col]+=1; V[e,col]+=1; V[h,col]+=1; W[c00,col]+=1; W[c11,col]+=1; col+=1
            U[c_,col]+=1; U[d,col]+=1; V[e,col]+=1; W[c10,col]+=1; W[c11,col]-=1; col+=1
            U[a,col]+=1; V[f,col]+=1; V[h,col]-=1; W[c01,col]+=1; W[c11,col]+=1; col+=1
            U[d,col]+=1; V[g,col]+=1; V[e,col]-=1; W[c00,col]+=1; W[c10,col]+=1; col+=1
            U[a,col]+=1; U[b,col]+=1; V[h,col]+=1; W[c00,col]-=1; W[c01,col]+=1; col+=1
            U[c_,col]+=1; U[a,col]-=1; V[e,col]+=1; V[f,col]+=1; W[c11,col]+=1; col+=1
            U[b,col]+=1; U[d,col]-=1; V[g,col]+=1; V[h,col]+=1; W[c00,col]+=1; col+=1

        # 2x2x3 decomposition in rank 9: A(2x2) * B(2x3) -> C(2x3)
        # Strassen 2x2x2 for B[:,0:2] gives 7 terms, plus 2 terms for B[:,2]
        def block_223(i0, j0, k0):
            nonlocal col
            # Strassen 2x2x2 for k0, k0+1
            strassen_222(i0, j0, k0)
            # Schoolbook 2x2x1 for k0+2
            a, b = i0*m+j0, i0*m+j0+1
            c_, d = (i0+1)*m+j0, (i0+1)*m+j0+1
            g = j0*p+k0+2
            h = (j0+1)*p+k0+2
            c02 = (k0+2)*n+i0
            c12 = (k0+2)*n+i0+1
            U[a,col]+=1; V[g,col]+=1; W[c02,col]+=1; col+=1
            U[c_,col]+=1; V[g,col]+=1; W[c12,col]+=1; col+=1

        # Block 1: j=0,1, k=0,1,2 -> rank 9
        block_223(0, 0, 0)
        # Block 2: j=0,1, k=3,4,5 -> rank 9
        block_223(0, 0, 3)
        # Block 3: j=0,1, k=6 -> rank 4 (schoolbook 2x2x1)
        a, b = 0*m+0, 0*m+1
        c_, d = 1*m+0, 1*m+1
        e = 0*p+6; g = 1*p+6
        c00 = 6*n+0; c10 = 6*n+1
        U[a,col]+=1; V[e,col]+=1; W[c00,col]+=1; col+=1
        U[b,col]+=1; V[g,col]+=1; W[c00,col]+=1; col+=1
        U[c_,col]+=1; V[e,col]+=1; W[c10,col]+=1; col+=1
        U[d,col]+=1; V[g,col]+=1; W[c10,col]+=1; col+=1
        # Block 4: j=2,3, k=0,1,2 -> rank 9
        block_223(0, 2, 0)
        # Block 5: j=2,3, k=3,4,5 -> rank 9
        block_223(0, 2, 3)
        # Block 6: j=2,3, k=6 -> rank 4
        a, b = 0*m+2, 0*m+3
        c_, d = 1*m+2, 1*m+3
        e = 2*p+6; g = 3*p+6
        c00 = 6*n+0; c10 = 6*n+1
        U[a,col]+=1; V[e,col]+=1; W[c00,col]+=1; col+=1
        U[b,col]+=1; V[g,col]+=1; W[c00,col]+=1; col+=1
        U[c_,col]+=1; V[e,col]+=1; W[c10,col]+=1; col+=1
        U[d,col]+=1; V[g,col]+=1; W[c10,col]+=1; col+=1

        if col != rank:
            return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result = build_rank48()
    if result is not None and verify(result["U"], result["V"], result["W"]):
        return result

    return schoolbook()

# EVOLVE-BLOCK-END
