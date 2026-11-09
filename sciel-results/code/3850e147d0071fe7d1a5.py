"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Try Strassen-based decomposition with rank 55
    # If verification fails, fall back to schoolbook rank 60
    
    def build_strassen():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for block in range(5):
            bcol = 2 * block
            
            def Uidx(i,j): return i*m+j
            def Vidx(j,k): return j*p+k
            def Widx(i,k): return k*n+i
            
            # Strassen 7 terms for A[:2,:2] @ B[:2, bcol:bcol+2]
            # P1=(a+d)(e+h): U={a:1,d:1}, V={e:1,h:1}, W={C11:1,C22:1}
            U[Uidx(0,0)][col]+=1; U[Uidx(1,1)][col]+=1
            V[Vidx(0,bcol)][col]+=1; V[Vidx(1,bcol+1)][col]+=1
            W[Widx(0,0)][col]+=1; W[Widx(1,1)][col]+=1
            col+=1
            # P2=(c+d)e: U={c:1,d:1}, V={e:1}, W={C21:1}
            U[Uidx(1,0)][col]+=1; U[Uidx(1,1)][col]+=1
            V[Vidx(0,bcol)][col]+=1
            W[Widx(1,0)][col]+=1
            col+=1
            # P3=a(f-h): U={a:1}, V={f:1,h:-1}, W={C12:1}
            U[Uidx(0,0)][col]+=1
            V[Vidx(0,bcol+1)][col]+=1; V[Vidx(1,bcol+1)][col]-=1
            W[Widx(0,1)][col]+=1
            col+=1
            # P4=d(g-e): U={d:1}, V={g:1,e:-1}, W={C21:1,C11:-1}
            U[Uidx(1,1)][col]+=1
            V[Vidx(1,bcol)][col]+=1; V[Vidx(0,bcol)][col]-=1
            W[Widx(1,0)][col]+=1; W[Widx(0,0)][col]-=1
            col+=1
            # P5=(a+b)h: U={a:1,b:1}, V={h:1}, W={C12:1,C22:1}
            U[Uidx(0,0)][col]+=1; U[Uidx(0,1)][col]+=1
            V[Vidx(1,bcol+1)][col]+=1
            W[Widx(0,1)][col]+=1; W[Widx(1,1)][col]+=1
            col+=1
            # P6=(c-a)(e+f): U={c:1,a:-1}, V={e:1,f:1}, W={C11:1}
            U[Uidx(1,0)][col]+=1; U[Uidx(0,0)][col]-=1
            V[Vidx(0,bcol)][col]+=1; V[Vidx(0,bcol+1)][col]+=1
            W[Widx(0,0)][col]+=1
            col+=1
            # P7=(b-d)(g+h): U={b:1,d:-1}, V={g:1,h:1}, W={C11:1,C22:1}
            U[Uidx(0,1)][col]+=1; U[Uidx(1,1)][col]-=1
            V[Vidx(1,bcol)][col]+=1; V[Vidx(1,bcol+1)][col]+=1
            W[Widx(0,0)][col]+=1; W[Widx(1,1)][col]+=1
            col+=1
            
            # Standard 2x1x1x2: A[:,2] @ B[2, bcol:bcol+2]
            for i in range(2):
                for k in range(2):
                    U[Uidx(i,2)][col]=1
                    V[Vidx(2,bcol+k)][col]=1
                    W[Widx(i,k)][col]=1
                    col+=1
        
        return {"U": U, "V": V, "W": W}
    
    def build_schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col]=1
                    V[j*p+k][col]=1
                    W[k*n+i][col]=1
                    col+=1
        return {"U": U, "V": V, "W": W}
    
    # Verify Strassen decomposition
    result = build_strassen()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verify: sum_c U[i,c]*V[j,c]*W[k,c] should equal delta(i,j,k) for matmul
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So tensor T[i,j,k] = 1 if this is a valid matmul term
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return result
    else:
        return build_schoolbook()
# EVOLVE-BLOCK-END
