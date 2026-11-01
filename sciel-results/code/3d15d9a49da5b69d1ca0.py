import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    # Term 1: A[:,2] * B[2,:] (2x1 by 1x4, rank 8)
    # C[i,k] = A[i,2] * B[2,k] for i in {0,1}, k in {0,1,2,3}
    for i in range(n):
        for k in range(p):
            r = i * p + k  # 0..7
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
    
    # Term 2: A[:,:2] * B[:2,:] (2x2 by 2x4, rank 14 via Strassen)
    # Split B[:2,:] into B1=B[:2,:2] and B2=B[:2,2:4]
    
    # C1 = A[:,:2] * B[:2,:2] (2x2 by 2x2, Strassen)
    # A = [[a,b],[c,d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B1 = [[e,f],[g,h]] where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # Output C1[i,k] goes to W[k*2+i, r] for k in {0,1}
    
    # M1 = (a+d)(e+h)
    r = 8
    U[0*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = 1  # C1[0,0]
    W[1*n+1][r] = 1  # C1[1,1]
    
    # M2 = (c+d)e
    r = 9
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1
    W[1*n+0][r] = 1  # C1[1,0]
    
    # M3 = a(f-h)
    r = 10
    U[0*m+0][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = -1
    W[0*n+1][r] = 1  # C1[0,1]
    
    # M4 = d(g-e)
    r = 11
    U[1*m+1][r] = 1
    V[1*p+0][r] = 1; V[0*p+0][r] = -1
    W[0*n+0][r] = 1  # C1[0,0]
    W[1*n+0][r] = 1  # C1[1,0]
    
    # M5 = (a+b)h
    r = 12
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+1][r] = 1
    W[0*n+0][r] = -1  # C1[0,0]
    W[0*n+1][r] = 1   # C1[0,1]
    
    # M6 = (c-a)(e+f)
    r = 13
    U[1*m+0][r] = 1; U[0*m+0][r] = -1
    V[0*p+0][r] = 1; V[0*p+1][r] = 1
    W[1*n+1][r] = 1  # C1[1,1]
    
    # M7 = (b-d)(g+h)
    r = 14
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[1*p+0][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = 1  # C1[0,0]
    
    # C2 = A[:,:2] * B[:2,2:4] (2x2 by 2x2, Strassen)
    # B2 = [[i,j],[k,l]] where i=B[0,2], j=B[0,3], k=B[1,2], l=B[1,3]
    # Output C2[i,k] goes to W[(k+2)*2+i, r] for k in {0,1}
    
    # N1 = (a+d)(i+l)
    r = 15
    U[0*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+2][r] = 1; V[1*p+3][r] = 1
    W[2*n+0][r] = 1  # C2[0,0] -> output k=2
    W[3*n+1][r] = 1  # C2[1,1] -> output k=3
    
    # N2 = (c+d)i
    r = 16
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+2][r] = 1
    W[2*n+1][r] = 1  # C2[1,0] -> output k=2, i=1
    
    # N3 = a(j-l)
    r = 17
    U[0*m+0][r] = 1
    V[0*p+3][r] = 1; V[1*p+3][r] = -1
    W[3*n+0][r] = 1  # C2[0,1] -> output k=3, i=0
    
    # N4 = d(k-i)
    r = 18
    U[1*m+1][r] = 1
    V[1*p+2][r] = 1; V[0*p+2][r] = -1
    W[2*n+0][r] = 1  # C2[0,0]
    W[2*n+1][r] = 1  # C2[1,0]
    
    # N5 = (a+b)l
    r = 19
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+3][r] = 1
    W[2*n+0][r] = -1  # C2[0,0]
    W[3*n+0][r] = 1   # C2[0,1]
    
    # N6 = (c-a)(i+j)
    r = 20
    U[1*m+0][r] = 1; U[0*m+0][r] = -1
    V[0*p+2][r] = 1; V[0*p+3][r] = 1
    W[3*n+1][r] = 1  # C2[1,1]
    
    # N7 = (b-d)(k+l)
    r = 21
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[1*p+2][r] = 1; V[1*p+3][r] = 1
    W[2*n+0][r] = 1  # C2[0,0]
    
    return {"U": U, "V": V, "W": W}
