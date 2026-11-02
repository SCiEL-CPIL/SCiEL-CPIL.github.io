DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    # Helper to add to U[i*m+j, c]
    def add_U(i, j, c, val):
        U[i*m+j][c] += val
    
    def add_V(j, k, c, val):
        V[j*p+k][c] += val
    
    def add_W(k, i, c, val):
        W[k*n+i][c] += val
    
    # Sub-block 1a: A[:,:2] * B[:2,:2] → C[0:2, 0:2]
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    c = 0
    add_U(0,0,c,1); add_U(1,1,c,1)
    add_V(0,0,c,1); add_V(1,1,c,1)
    add_W(0,0,c,1); add_W(1,1,c,1)
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    c = 1
    add_U(1,0,c,1); add_U(1,1,c,1)
    add_V(0,0,c,1)
    add_W(0,1,c,1); add_W(1,1,c,-1)
    
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    c = 2
    add_U(0,0,c,1)
    add_V(0,1,c,1); add_V(1,1,c,-1)
    add_W(1,0,c,1); add_W(1,1,c,1)
    
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    c = 3
    add_U(1,1,c,1)
    add_V(1,0,c,1); add_V(0,0,c,-1)
    add_W(0,0,c,1); add_W(0,1,c,1)
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    c = 4
    add_U(0,0,c,1); add_U(0,1,c,1)
    add_V(1,1,c,1)
    add_W(0,0,c,-1); add_W(1,0,c,1)
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    c = 5
    add_U(1,0,c,1); add_U(0,0,c,-1)
    add_V(0,0,c,1); add_V(0,1,c,1)
    add_W(1,1,c,1)
    
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    c = 6
    add_U(0,1,c,1); add_U(1,1,c,-1)
    add_V(1,0,c,1); add_V(1,1,c,1)
    add_W(0,0,c,1)
    
    # Sub-block 1b: A[:,2:4] * B[2:4,:2] → C[0:2, 0:2]
    # N1 = (A[0,2]+A[1,3])*(B[2,0]+B[3,1])
    c = 7
    add_U(0,2,c,1); add_U(1,3,c,1)
    add_V(2,0,c,1); add_V(3,1,c,1)
    add_W(0,0,c,1); add_W(1,1,c,1)
    
    # N2 = (A[1,2]+A[1,3])*B[2,0]
    c = 8
    add_U(1,2,c,1); add_U(1,3,c,1)
    add_V(2,0,c,1)
    add_W(0,1,c,1); add_W(1,1,c,-1)
    
    # N3 = A[0,2]*(B[2,1]-B[3,1])
    c = 9
    add_U(0,2,c,1)
    add_V(2,1,c,1); add_V(3,1,c,-1)
    add_W(1,0,c,1); add_W(1,1,c,1)
    
    # N4 = A[1,3]*(B[3,0]-B[2,0])
    c = 10
    add_U(1,3,c,1)
    add_V(3,0,c,1); add_V(2,0,c,-1)
    add_W(0,0,c,1); add_W(0,1,c,1)
    
    # N5 = (A[0,2]+A[0,3])*B[3,1]
    c = 11
    add_U(0,2,c,1); add_U(0,3,c,1)
    add_V(3,1,c,1)
    add_W(0,0,c,-1); add_W(1,0,c,1)
    
    # N6 = (A[1,2]-A[0,2])*(B[2,0]+B[2,1])
    c = 12
    add_U(1,2,c,1); add_U(0,2,c,-1)
    add_V(2,0,c,1); add_V(2,1,c,1)
    add_W(1,1,c,1)
    
    # N7 = (A[0,3]-A[1,3])*(B[3,0]+B[3,1])
    c = 13
    add_U(0,3,c,1); add_U(1,3,c,-1)
    add_V(3,0,c,1); add_V(3,1,c,1)
    add_W(0,0,c,1)
    
    # Sub-block 2a: A[:,:2] * B[:2,2:4] → C[0:2, 2:4]
    # P1 = (A[0,0]+A[1,1])*(B[0,2]+B[1,3])
    c = 14
    add_U(0,0,c,1); add_U(1,1,c,1)
    add_V(0,2,c,1); add_V(1,3,c,1)
    add_W(2,0,c,1); add_W(3,1,c,1)
    
    # P2 = (A[1,0]+A[1,1])*B[0,2]
    c = 15
    add_U(1,0,c,1); add_U(1,1,c,1)
    add_V(0,2,c,1)
    add_W(2,1,c,1); add_W(3,1,c,-1)
    
    # P3 = A[0,0]*(B[0,3]-B[1,3])
    c = 16
    add_U(0,0,c,1)
    add_V(0,3,c,1); add_V(1,3,c,-1)
    add_W(3,0,c,1); add_W(3,1,c,1)
    
    # P4 = A[1,1]*(B[1,2]-B[0,2])
    c = 17
    add_U(1,1,c,1)
    add_V(1,2,c,1); add_V(0,2,c,-1)
    add_W(2,0,c,1); add_W(2,1,c,1)
    
    # P5 = (A[0,0]+A[0,1])*B[1,3]
    c = 18
    add_U(0,0,c,1); add_U(0,1,c,1)
    add_V(1,3,c,1)
    add_W(2,0,c,-1); add_W(3,0,c,1)
    
    # P6 = (A[1,0]-A[0,0])*(B[0,2]+B[0,3])
    c = 19
    add_U(1,0,c,1); add_U(0,0,c,-1)
    add_V(0,2,c,1); add_V(0,3,c,1)
    add_W(3,1,c,1)
    
    # P7 = (A[0,1]-A[1,1])*(B[1,2]+B[1,3])
    c = 20
    add_U(0,1,c,1); add_U(1,1,c,-1)
    add_V(1,2,c,1); add_V(1,3,c,1)
    add_W(2,0,c,1)
    
    # Sub-block 2b: A[:,2:4] * B[2:4,2:4] → C[0:2, 2:4]
    # Q1 = (A[0,2]+A[1,3])*(B[2,2]+B[3,3])
    c = 21
    add_U(0,2,c,1); add_U(1,3,c,1)
    add_V(2,2,c,1); add_V(3,3,c,1)
    add_W(2,0,c,1); add_W(3,1,c,1)
    
    # Q2 = (A[1,2]+A[1,3])*B[2,2]
    c = 22
    add_U(1,2,c,1); add_U(1,3,c,1)
    add_V(2,2,c,1)
    add_W(2,1,c,1); add_W(3,1,c,-1)
    
    # Q3 = A[0,2]*(B[2,3]-B[3,3])
    c = 23
    add_U(0,2,c,1)
    add_V(2,3,c,1); add_V(3,3,c,-1)
    add_W(3,0,c,1); add_W(3,1,c,1)
    
    # Q4 = A[1,3]*(B[3,2]-B[2,2])
    c = 24
    add_U(1,3,c,1)
    add_V(3,2,c,1); add_V(2,2,c,-1)
    add_W(2,0,c,1); add_W(2,1,c,1)
    
    # Q5 = (A[0,2]+A[0,3])*B[3,3]
    c = 25
    add_U(0,2,c,1); add_U(0,3,c,1)
    add_V(3,3,c,1)
    add_W(2,0,c,-1); add_W(3,0,c,1)
    
    # Q6 = (A[1,2]-A[0,2])*(B[2,2]+B[2,3])
    c = 26
    add_U(1,2,c,1); add_U(0,2,c,-1)
    add_V(2,2,c,1); add_V(2,3,c,1)
    add_W(3,1,c,1)
    
    # Q7 = (A[0,3]-A[1,3])*(B[3,2]+B[3,3])
    c = 27
    add_U(0,3,c,1); add_U(1,3,c,-1)
    add_V(3,2,c,1); add_V(3,3,c,1)
    add_W(2,0,c,1)
    
    return {"U": U, "V": V, "W": W}
