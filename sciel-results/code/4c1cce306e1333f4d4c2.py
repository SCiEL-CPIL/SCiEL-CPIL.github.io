"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    for jb in range(2):
        for kb in range(2):
            # Strassen 2x2x2: rows 0,1; A-cols 2jb,2jb+1; B-cols 2kb,2kb+1
            # a00=A[0][2jb], a01=A[0][2jb+1], a10=A[1][2jb], a11=A[1][2jb+1]
            # b00=B[2jb][2kb], b01=B[2jb][2kb+1], b10=B[2jb+1][2kb], b11=B[2jb+1][2kb+1]
            # c00=C[0][2kb], c01=C[0][2kb+1], c10=C[1][2kb], c11=C[1][2kb+1]
            
            # M1 = (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
            U[0*4+2*jb][col]=1; U[1*4+2*jb+1][col]=1
            V[2*jb*4+2*kb][col]=1; V[(2*jb+1)*4+2*kb+1][col]=1
            W[2*kb*3+0][col]=1; W[(2*kb+1)*3+1][col]=1
            col+=1
            
            # M2 = (a10+a11)b00 -> c11-=M2
            U[1*4+2*jb][col]=1; U[1*4+2*jb+1][col]=1
            V[2*jb*4+2*kb][col]=1
            W[(2*kb+1)*3+1][col]=-1
            col+=1
            
            # M3 = a00(b10-b11) -> c01+=M3, c11+=M3
            U[0*4+2*jb][col]=1
            V[(2*jb+1)*4+2*kb][col]=1; V[(2*jb+1)*4+2*kb+1][col]=-1
            W[(2*kb+1)*3+0][col]=1; W[(2*kb+1)*3+1][col]=1
            col+=1
            
            # M4 = a11(b01-b10) -> c00+=M4, c10+=M4
            U[1*4+2*jb+1][col]=1
            V[2*jb*4+2*kb+1][col]=1; V[(2*jb+1)*4+2*kb][col]=-1
            W[2*kb*3+0][col]=1; W[2*kb*3+1][col]=1
            col+=1
            
            # M5 = (a00+a01)b11 -> c00-=M5, c01+=M5
            U[0*4+2*jb][col]=1; U[0*4+2*jb+1][col]=1
            V[(2*jb+1)*4+2*kb+1][col]=1
            W[2*kb*3+0][col]=-1; W[(2*kb+1)*3+0][col]=1
            col+=1
            
            # M6 = (a10-a00)(b00+b10) -> c11+=M6
            U[1*4+2*jb][col]=1; U[0*4+2*jb][col]=-1
            V[2*jb*4+2*kb][col]=1; V[(2*jb+1)*4+2*kb][col]=1
            W[(2*kb+1)*3+1][col]=1
            col+=1
            
            # M7 = (a01-a11)(b01+b11) -> c00+=M7
            U[0*4+2*jb+1][col]=1; U[1*4+2*jb+1][col]=-1
            V[2*jb*4+2*kb+1][col]=1; V[(2*jb+1)*4+2*kb+1][col]=1
            W[2*kb*3+0][col]=1
            col+=1
            
            # Trivial 1x2x2: row 2
            U[2*4+2*jb][col]=1; V[2*jb*4+2*kb][col]=1; W[2*kb*3+2][col]=1; col+=1
            U[2*4+2*jb+1][col]=1; V[(2*jb+1)*4+2*kb][col]=1; W[2*kb*3+2][col]=1; col+=1
            U[2*4+2*jb][col]=1; V[2*jb*4+2*kb+1][col]=1; W[(2*kb+1)*3+2][col]=1; col+=1
            U[2*4+2*jb+1][col]=1; V[(2*jb+1)*4+2*kb+1][col]=1; W[(2*kb+1)*3+2][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
