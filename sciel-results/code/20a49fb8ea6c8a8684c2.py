"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """Rank-minimal (or near-minimal) tensor decomposition for (2,2,9) matmul.
    
    Strategy: Use Strassen's 7-rank decomposition for (2,2,2) blocks.
    Split 9 columns into four pairs (0,1),(2,3),(4,5),(6,7) and one singleton (8).
    Rank = 4*7 + 4 = 32.
    """
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(list(u))
        V_cols.append(list(v))
        W_cols.append(list(w))
    
    def z4(): return [0.0]*4
    def z18(): return [0.0]*18
    
    def strassen_block(k0, k1):
        """Strassen 7-rank for columns k0, k1."""
        # Standard Strassen formulas:
        # M1 = (A00+A11)(B00+B11) -> c00+=1, c11+=1
        # M2 = (A10+A11)B00       -> c10+=1
        # M3 = A00(B10-B11)       -> c01+=1
        # M4 = A11(B01-B00)       -> c00+=1, c10+=1
        # M5 = (A00+A01)B11       -> c00-=1, c01+=1
        # M6 = (A01-A11)(B00+B10) -> c11+=1
        # M7 = (A10-A00)(B01+B11) -> c00+=1
        # where B00=B[0,k0], B01=B[0,k1], B10=B[1,k0], B11=B[1,k1]
        # c00=C[0,k0], c01=C[0,k1], c10=C[1,k0], c11=C[1,k1]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
        u = z4(); u[0]=1; u[3]=1
        v = z18(); v[0*p+k0]=1; v[1*p+k1]=1
        w = z18(); w[k0*n+0]=1; w[k1*n+1]=1
        add_term(u,v,w)
        
        # M2 = (A[1,0]+A[1,1])*B[0,k0]
        u = z4(); u[2]=1; u[3]=1
        v = z18(); v[0*p+k0]=1
        w = z18(); w[k0*n+1]=1
        add_term(u,v,w)
        
        # M3 = A[0,0]*(B[1,k0]-B[1,k1])
        u = z4(); u[0]=1
        v = z18(); v[1*p+k0]=1; v[1*p+k1]=-1
        w = z18(); w[k1*n+0]=1
        add_term(u,v,w)
        
        # M4 = A[1,1]*(B[0,k1]-B[0,k0])
        u = z4(); u[3]=1
        v = z18(); v[0*p+k1]=1; v[0*p+k0]=-1
        w = z18(); w[k0*n+0]=1; w[k0*n+1]=1
        add_term(u,v,w)
        
        # M5 = (A[0,0]+A[0,1])*B[1,k1]
        u = z4(); u[0]=1; u[1]=1
        v = z18(); v[1*p+k1]=1
        w = z18(); w[k0*n+0]=-1; w[k1*n+0]=1
        add_term(u,v,w)
        
        # M6 = (A[0,1]-A[1,1])*(B[0,k0]+B[1,k0])
        u = z4(); u[1]=1; u[3]=-1
        v = z18(); v[0*p+k0]=1; v[1*p+k0]=1
        w = z18(); w[k1*n+1]=1
        add_term(u,v,w)
        
        # M7 = (A[1,0]-A[0,0])*(B[0,k1]+B[1,k1])
        u = z4(); u[2]=1; u[0]=-1
        v = z18(); v[0*p+k1]=1; v[1*p+k1]=1
        w = z18(); w[k0*n+0]=1
        add_term(u,v,w)
    
    def schoolbook_col(k):
        """4 rank-1 terms for single column k."""
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        u = z4(); u[0]=1
        v = z18(); v[0*p+k]=1
        w = z18(); w[k*n+0]=1
        add_term(u,v,w)
        
        u = z4(); u[1]=1
        v = z18(); v[1*p+k]=1
        w = z18(); w[k*n+0]=1
        add_term(u,v,w)
        
        u = z4(); u[2]=1
        v = z18(); v[0*p+k]=1
        w = z18(); w[k*n+1]=1
        add_term(u,v,w)
        
        u = z4(); u[3]=1
        v = z18(); v[1*p+k]=1
        w = z18(); w[k*n+1]=1
        add_term(u,v,w)
    
    # Build decomposition
    strassen_block(0, 1)
    strassen_block(2, 3)
    strassen_block(4, 5)
    strassen_block(6, 7)
    schoolbook_col(8)
    
    rank = len(U_cols)
    U = [U_cols[c] for c in range(rank)]
    V = [V_cols[c] for c in range(rank)]
    W = [W_cols[c] for c in range(rank)]
    
    # Verify with random test
    np.random.seed(12345)
    A_t = np.random.randn(2, 2)
    B_t = np.random.randn(2, 9)
    C_t = A_t @ B_t
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Verify: C[i,k] = sum_c sum_j A[i,j]*U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # More precisely, the trilinear form check:
    # sum_{i,j,k} T[i,j,k]*A[i,j]*B[j,k]*C[i,k] where T[i,j,k]=1
    # should equal sum_c (sum_{i,j} A[i,j]*U[i*m+j,c]) * (sum_{j,k} B[j,k]*V[j*p+k,c]) * (sum_{k,i} C[i,k]*W[k*n+i,c])
    
    lhs = np.sum(A_t * 1) * 0  # placeholder
    # Actually: lhs = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
    lhs = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(9):
                lhs += A_t[i,j] * B_t[j,k] * C_t[i,k]
    
    rhs = 0.0
    for c in range(rank):
        a_c = sum(A_t[i,j] * U_arr[i*2+j, c] for i in range(2) for j in range(2))
        b_c = sum(B_t[j,k] * V_arr[j*9+k, c] for j in range(2) for k in range(9))
        c_c = sum(C_t[i,k] * W_arr[k*2+i, c] for i in range(2) for k in range(9))
        rhs += a_c * b_c * c_c
    
    if abs(lhs - rhs) > 1e-8:
        # Fallback to schoolbook
        r = 36
        U_s = [[0.0]*r for _ in range(4)]
        V_s = [[0.0]*r for _ in range(18)]
        W_s = [[0.0]*r for _ in range(18)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_s[i*2+j][col] = 1.0
                    V_s[j*9+k][col] = 1.0
                    W_s[k*2+i][col] = 1.0
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
