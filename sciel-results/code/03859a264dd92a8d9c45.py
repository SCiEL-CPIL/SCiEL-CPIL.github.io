def construct():
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0]*rank for _ in range(n*m)]  # 6 rows
    V = [[0]*rank for _ in range(m*p)]  # 18 rows
    W = [[0]*rank for _ in range(p*n)]  # 12 rows
    
    col = 0
    for b in range(3):  # 3 blocks of 2 columns
        k0 = 2*b
        k1 = 2*b + 1
        
        # Strassen for 2x2x2 (A[:,0:2] * B[0:2,:])
        # m1 = (a+d)(e+h) where a=A[0,0], d=A[1,1], e=B[0,k0], h=B[1,k1]
        # U: A[0,0] and A[1,1] -> rows 0*3+0=0 and 1*3+1=4
        # V: B[0,k0] and B[1,k1] -> rows 0*6+k0 and 1*6+k1
        # W: contributes to all 4 outputs of this block
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1
        U[4][col] = 1
        V[0*6+k0][col] = 1
        V[1*6+k1][col] = 1
        # m1 contributes to: C[0,0]+=m1, C[0,1]+=0, C[1,0]+=0, C[1,1]+=m1
        # W[k0*2+0] += 1, W[k1*2+1] += 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # m2 = (c+d)*e where c=A[1,0], d=A[1,1], e=B[0,k0]
        U[3][col] = 1  # A[1,0]
        U[4][col] = 1  # A[1,1]
        V[0*6+k0][col] = 1  # B[0,k0]
        # m2 contributes to: C[1,0]+=m2, C[1,1]-=m2
        W[k0*2+1][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
        # m3 = a*(f-h) where a=A[0,0], f=B[0,k1], h=B[1,k1]
        U[0][col] = 1
        V[0*6+k1][col] = 1
        V[1*6+k1][col] = -1
        # m3 contributes to: C[0,1]+=m3, C[1,1]+=m3
        W[k1*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # m4 = d*(g-e) where d=A[1,1], g=B[1,k0], e=B[0,k0]
        U[4][col] = 1
        V[1*6+k0][col] = 1
        V[0*6+k0][col] = -1
        # m4 contributes to: C[0,0]+=m4, C[1,0]+=m4
        W[k0*2+0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # m5 = (a+b)*h where a=A[0,0], b=A[0,1], h=B[1,k1]
        U[0][col] = 1
        U[1][col] = 1
        V[1*6+k1][col] = 1
        # m5 contributes to: C[0,0]-=m5, C[0,1]+=m5
        W[k0*2+0][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # m6 = (c-a)*(e+f) where c=A[1,0], a=A[0,0], e=B[0,k0], f=B[0,k1]
        U[3][col] = 1
        U[0][col] = -1
        V[0*6+k0][col] = 1
        V[0*6+k1][col] = 1
        # m6 contributes to: C[1,1]+=m6
        W[k1*2+1][col] = 1
        col += 1
        
        # m7 = (b-d)*(g+h) where b=A[0,1], d=A[1,1], g=B[1,k0], h=B[1,k1]
        U[1][col] = 1
        U[4][col] = -1
        V[1*6+k0][col] = 1
        V[1*6+k1][col] = 1
        # m7 contributes to: C[0,0]+=m7
        W[k0*2+0][col] = 1
        col += 1
        
        # Schoolbook for 2x1x2 (A[:,2] * B[2,:])
        # p*r: A[0,2]*B[2,k0]
        U[2][col] = 1  # A[0,2]
        V[2*6+k0][col] = 1  # B[2,k0]
        W[k0*2+0][col] = 1
        col += 1
        
        # p*s: A[0,2]*B[2,k1]
        U[2][col] = 1
        V[2*6+k1][col] = 1
        W[k1*2+0][col] = 1
        col += 1
        
        # q*r: A[1,2]*B[2,k0]
        U[5][col] = 1  # A[1,2]
        V[2*6+k0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # q*s: A[1,2]*B[2,k1]
        U[5][col] = 1
        V[2*6+k1][col] = 1
        W[k1*2+1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
