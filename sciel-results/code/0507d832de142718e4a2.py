"""Rank-28 Strassen-based exact decomposition for 2x4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2,4,4
    # A: n x m (2x4), B: m x p (4x4), C: p x n (4x2) with C index W[k*n+i] = C[k][i]? 
    # Output factor W[k*n+i, r]: k in [0,p)=4 rows of C? Actually C is 2x4 (n x m). 
    # Trace convention W[k*n+i]: i in [0,n)=2, k in [0,p)=4. So C[i][k] = (A B)[i][k].
    # A is 2x4 (n x m), B is 4x4 (m x p). C = A B is 2 x 4 (n x p). C[i][k].
    # W index k*n+i = k*2+i. So W row = k*2+i for C[i][k]. Good.

    def strassen_cols(A_idx, B_idx, C_idx, U, V, W):
        # A_idx: [a,b,c,d] indices into A (8 entries)
        # B_idx: [e,f,g,h] indices into B (16 entries)
        # C_idx: [p,q,r,s] indices into C (8 entries)  (p=C00,q=C01,r=C10,s=C11 of the 2x2 block)
        a,b,c,d = A_idx
        e,f,g,h = B_idx
        cp,cq,cr,cs = C_idx
        X = [
            # (u_coeffs on (a,b,c,d), v_coeffs on (e,f,g,h), w_coeffs on (p,q,r,s))
            ((1,1,0,0),(1,0,0,1),(1,0,0,-1)),   # X1: p+1, s-1
            ((0,0,1,1),(1,0,0,0),(0,0,1,1)),    # X2: r+1, s+1
            ((1,0,-1,0),(0,0,1,1),(1,0,0,1)),   # X3: p+1, s+1
            ((0,1,0,-1),(0,1,0,1),(0,1,0,-1)),  # X4: q+1, s-1
            ((1,0,0,1),(0,0,0,1),(1,0,0,1)),    # X5: p+1, s+1
            ((0,0,1,1),(0,1,0,-1),(0,0,1,0)),   # X6: r+1
            ((1,-1,0,0),(1,0,0,0),(-1,0,0,0)),  # X7: p-1
        ]
        for (uc, vc, wc) in X:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            for t in range(4):
                u[A_idx[t]] += uc[t]
                v[B_idx[t]] += vc[t]
            for t in range(4):
                w[C_idx[t]] += wc[t]
            U.append(u); V.append(v); W.append(w)

    U=[]; V=[]; W=[]
    # A indices: A[i][j] -> i*m+j = i*4+j. A is 2x4.
    # A_left = cols0-1: A[0][0]=0,A[0][1]=1,A[1][0]=4,A[1][1]=5 -> (a,b,c,d)
    # A_right = cols2-3: A[0][2]=2,A[0][3]=3,A[1][2]=6,A[1][3]=7
    A_left  = [0,1,4,5]
    A_right = [2,3,6,7]
    # B indices: B[i][j] -> i*p+j = i*4+j. B is 4x4.
    # B11 rows0-1 cols0-1: B00=0,B01=1,B10=4,B11=5
    # B12 rows0-1 cols2-3: B02=2,B03=3,B12=6,B13=7
    # B21 rows2-3 cols0-1: B20=8,B21=9,B30=12,B31=13
    # B22 rows2-3 cols2-3: B22=10,B23=11,B32=14,B33=15
    B11 = [0,1,4,5]
    B12 = [2,3,6,7]
    B21 = [8,9,12,13]
    B22 = [10,11,14,15]
    # C indices: C[i][k] -> W index k*n+i = k*2+i. C is 2x4 (n=2 rows, p=4 cols).
    # C_left = cols0-1: C00=0*2+0=0, C01=0*2+1=1, C10=1*2+0=2, C11=1*2+1=3
    # C_right = cols2-3: C02=2*2+0=4, C03=2*2+1=5, C12=3*2+0=6, C13=3*2+1=7
    C_left  = [0,1,2,3]
    C_right = [4,5,6,7]

    # C_left = A_left*B11 + A_right*B21
    strassen_cols(A_left, B11, C_left, U, V, W)
    strassen_cols(A_right, B21, C_left, U, V, W)
    # C_right = A_left*B12 + A_right*B22
    strassen_cols(A_left, B12, C_right, U, V, W)
    strassen_cols(A_right, B22, C_right, U, V, W)

    rank = len(U)
    assert rank == 28, rank

    # Validate exactly with integer arithmetic (pure Python ints).
    import random
    rng = random.Random(12345)
    def dot(u, x):
        s = 0
        for a, b in zip(u, x):
            if a: s += a*b
        return s
    for _ in range(200):
        A = [rng.randint(-3,3) for _ in range(n*m)]
        B = [rng.randint(-3,3) for _ in range(m*p)]
        # standard C = A B, C[i][k] = sum_j A[i][j] B[j][k]; W index k*n+i
        Cref = [0]*(p*n)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i*m+j]*B[j*p+k]
                Cref[k*n+i] = s
        Cgot = [0]*(p*n)
        for c in range(rank):
            prod = dot(U[c], A) * dot(V[c], B)
            if prod:
                for idx, coeff in enumerate(W[c]):
                    if coeff:
                        Cgot[idx] += coeff*prod
        if Cgot != Cref:
            raise ValueError("decomposition invalid")
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
