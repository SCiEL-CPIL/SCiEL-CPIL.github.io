import numpy as np

def check_schoolbook(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    rank = n*m*p
    U = np.zeros((nm, rank)); V = np.zeros((mp, rank)); W = np.zeros((pn, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    A = np.random.randn(n, m); B = np.random.randn(m, p)
    Ctrue = A @ B
    # Convention 1: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j])*(sum_l V[l*p+k,c]*B[l,k])*W[k*n+i,c]
    C1 = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for cc in range(rank):
                sA = sum(A[i, j]*U[i*m+j, cc] for j in range(m))
                sB = sum(B[l, k]*V[l*p+k, cc] for l in range(m))
                C1[i, k] += sA*sB*W[k*n+i, cc]
    print("Convention1 err:", np.max(np.abs(C1 - Ctrue)))
    # Convention 2: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j])*(sum_l V[l*p+k,c]*B[l,k])  (W folded, W=1)
    C2 = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for cc in range(rank):
                sA = sum(A[i, j]*U[i*m+j, cc] for j in range(m))
                sB = sum(B[l, k]*V[l*p+k, cc] for l in range(m))
                C2[i, k] += sA*sB
    print("Convention2 err:", np.max(np.abs(C2 - Ctrue)))
    # Convention 3: C[i,k] = sum_c sum_j U[i*m+j,c]*A[i,j]*V[j*p+k,c]*W[k*n+i,c]
    C3 = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for cc in range(rank):
                for j in range(m):
                    C3[i, k] += U[i*m+j, cc]*A[i, j]*V[j*p+k, cc]*W[k*n+i, cc]
    print("Convention3 err:", np.max(np.abs(C3 - Ctrue)))

check_schoolbook(2, 4, 7)
