import time
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # a=A[0,j0], b=A[0,j0+1], c=A[1,j0], d=A[1,j0+1]
            # e=B[j0,k0], f=B[j0,k0+1], g=B[j0+1,k0], h=B[j0+1,k0+1]

            # p1 = (a+d)(e+h) -> C[0,k0]+=1, C[1,k0+1]+=1
            U[0*m+j0][col] += 1
            U[1*m+j0+1][col] += 1
            V[j0*p+k0][col] += 1
            V[(j0+1)*p+k0+1][col] += 1
            W[k0*n+0][col] += 1
            W[(k0+1)*n+1][col] += 1
            col += 1

            # p2 = (c+d)e -> C[1,k0]+=1, C[1,k0+1]+=-1
            U[1*m+j0][col] += 1
            U[1*m+j0+1][col] += 1
            V[j0*p+k0][col] += 1
            W[(k0+1)*n+0][col] += 1
            W[(k0+1)*n+1][col] += -1
            col += 1

            # p3 = a(f-h) -> C[0,k0+1]+=1, C[1,k0+1]+=1
            U[0*m+j0][col] += 1
            V[j0*p+k0+1][col] += 1
            V[(j0+1)*p+k0][col] += -1
            W[k0*n+1][col] += 1
            W[(k0+1)*n+1][col] += 1
            col += 1

            # p4 = d(g-e) -> C[0,k0]+=1, C[1,k0]+=1
            U[1*m+j0+1][col] += 1
            V[(j0+1)*p+k0][col] += 1
            V[j0*p+k0][col] += -1
            W[k0*n+0][col] += 1
            W[(k0+1)*n+0][col] += 1
            col += 1

            # p5 = (a+b)h -> C[0,k0]+=-1, C[0,k0+1]+=1
            U[0*m+j0][col] += 1
            U[0*m+j0+1][col] += 1
            V[(j0+1)*p+k0+1][col] += 1
            W[k0*n+0][col] += -1
            W[k0*n+1][col] += 1
            col += 1

            # p6 = (c-a)(e+f) -> C[1,k0+1]+=1
            U[1*m+j0][col] += 1
            U[0*m+j0][col] += -1
            V[j0*p+k0][col] += 1
            V[j0*p+k0+1][col] += 1
            W[(k0+1)*n+1][col] += 1
            col += 1

            # p7 = (b-d)(g+h) -> C[0,k0]+=1
            U[0*m+j0+1][col] += 1
            U[1*m+j0+1][col] += -1
            V[(j0+1)*p+k0][col] += 1
            V[(j0+1)*p+k0+1][col] += 1
            W[k0*n+0][col] += 1
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _verify():
    """Internal verification of the decomposition."""
    n, m, p = 2, 4, 8
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Test with random integer matrices
    for trial in range(50):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]

        # Schoolbook C = A * B
        C_true = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_true[i][k] = s

        # Tensor reconstruction
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, that's not right. Let me re-derive.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # C[i,k] = sum_j T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k]
        #         = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
        #         = sum_r W[k*n+i,r] * sum_j U[i*m+j,r] A[i,j] B[j,k] V[j*p+k,r]
        
        C_rec = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    wr = W[k*n+i][r]
                    if wr == 0:
                        continue
                    # sum_j U[i*m+j, r] * A[i,j] * B[j,k] * V[j*p+k, r]
                    inner = 0
                    for j in range(m):
                        inner += U[i*m+j][r] * A[i][j] * B[j][k] * V[j*p+k][r]
                    s += wr * inner
                C_rec[i][k] = s

        for i in range(n):
            for k in range(p):
                if C_true[i][k] != C_rec[i][k]:
                    raise ValueError(f"Mismatch at ({i},{k}): true={C_true[i][k]}, rec={C_rec[i][k]}")
    return True

# EVOLVE-BLOCK-END
