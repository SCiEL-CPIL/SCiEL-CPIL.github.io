"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for left block (B cols 0,1)
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(g-h), m4=d(f-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    # C11=m1+m4-m5+m7, C12=m3+m5, C21=m2+m4, C22=m1-m2+m3+m6
    # A: a=U[0],b=U[1],c=U[2],d=U[3]; B_left: e=V[0],f=V[1],g=V[4],h=V[5]
    # W: C[0,0]=W[0],C[0,1]=W[2],C[1,0]=W[1],C[1,1]=W[3]
    
    r=0  # m1=(a+d)(e+h)
    U[0][r]=1; U[3][r]=1
    V[0][r]=1; V[5][r]=1
    W[0][r]=1; W[3][r]=1  # C11+=m1, C22+=m1
    
    r=1  # m2=(c+d)e
    U[2][r]=1; U[3][r]=1
    V[0][r]=1
    W[1][r]=1; W[3][r]=-1  # C21+=m2, C22-=m2
    
    r=2  # m3=a(g-h)
    U[0][r]=1
    V[4][r]=1; V[5][r]=-1
    W[2][r]=1; W[3][r]=1  # C12+=m3, C22+=m3
    
    r=3  # m4=d(f-e)
    U[3][r]=1
    V[1][r]=1; V[0][r]=-1
    W[0][r]=1; W[1][r]=1  # C11+=m4, C21+=m4
    
    r=4  # m5=(a+b)h
    U[0][r]=1; U[1][r]=1
    V[5][r]=1
    W[0][r]=-1; W[2][r]=1  # C11-=m5, C12+=m5
    
    r=5  # m6=(c-a)(e+f)
    U[2][r]=1; U[0][r]=-1
    V[0][r]=1; V[1][r]=1
    W[3][r]=1  # C22+=m6
    
    r=6  # m7=(b-d)(g+h)
    U[1][r]=1; U[3][r]=-1
    V[4][r]=1; V[5][r]=1
    W[0][r]=1  # C11+=m7
    
    # Right block (B cols 2,3): e=V[2],f=V[3],g=V[6],h=V[7]
    # W: C[0,2]=W[4],C[0,3]=W[6],C[1,2]=W[5],C[1,3]=W[7]
    
    r=7  # m1=(a+d)(e+h)
    U[0][r]=1; U[3][r]=1
    V[2][r]=1; V[7][r]=1
    W[4][r]=1; W[7][r]=1
    
    r=8  # m2=(c+d)e
    U[2][r]=1; U[3][r]=1
    V[2][r]=1
    W[5][r]=1; W[7][r]=-1
    
    r=9  # m3=a(g-h)
    U[0][r]=1
    V[6][r]=1; V[7][r]=-1
    W[6][r]=1; W[7][r]=1
    
    r=10  # m4=d(f-e)
    U[3][r]=1
    V[3][r]=1; V[2][r]=-1
    W[4][r]=1; W[5][r]=1
    
    r=11  # m5=(a+b)h
    U[0][r]=1; U[1][r]=1
    V[7][r]=1
    W[4][r]=-1; W[6][r]=1
    
    r=12  # m6=(c-a)(e+f)
    U[2][r]=1; U[0][r]=-1
    V[2][r]=1; V[3][r]=1
    W[7][r]=1
    
    r=13  # m7=(b-d)(g+h)
    U[1][r]=1; U[3][r]=-1
    V[6][r]=1; V[7][r]=1
    W[4][r]=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
