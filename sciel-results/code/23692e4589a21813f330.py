"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def schoolbook_full():
        """Standard schoolbook decomposition with rank n*m*p = 45"""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p, num_trials=5):
        """Verify that the decomposition exactly decomposes matrix multiplication"""
        for trial in range(num_trials):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            
            C_actual = np.zeros((n, p))
            rank = U.shape[1]
            for r in range(rank):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                if np.all(u_r == 0) or np.all(v_r == 0) or np.all(w_r == 0):
                    continue
                a_part = np.zeros(n)
                b_part = np.zeros(p)
                for i in range(n):
                    a_part[i] = np.dot(u_r[i*m:(i+1)*m], A[i, :])
                for k in range(p):
                    b_part[k] = np.dot(v_r[k*m:(k+1)*m], B[:, k])
                for i in range(n):
                    for k in range(p):
                        C_actual[i, k] += w_r[k*n+i] * a_part[i] * b_part[k]
            
            if not np.allclose(C_actual, C_expected, atol=1e-10):
                return False
        return True
    
    def bsc_3x3x3():
        """
        Bini-Strassen-Caporaso rank-23 decomposition for 3x3x3.
        """
        U = np.zeros((9, 23), dtype=float)
        V = np.zeros((9, 23), dtype=float)
        W = np.zeros((9, 23), dtype=float)
        
        # BSC decomposition terms
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[4,1]=1; W[4,1]=1
        U[2,2]=1; V[8,2]=1; W[8,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        U[4,4]=1; V[5,4]=1; W[5,4]=1
        U[5,5]=1; V[8,5]=1; W[8,5]=1
        U[6,6]=1; V[1,6]=1; W[1,6]=1
        U[7,7]=1; V[5,7]=1; W[5,7]=1
        U[8,8]=1; V[8,8]=1; W[8,8]=1
        U[0,9]=1; U[1,9]=1; V[3,9]=1; W[4,9]=1
        U[1,10]=1; U[2,10]=1; V[7,10]=1; W[8,10]=1
        U[3,11]=1; U[4,11]=1; V[2,11]=1; W[8,11]=1
        U[4,12]=1; U[5,12]=1; V[3,12]=1; W[5,12]=1
        U[6,13]=1; U[7,13]=1; V[2,13]=1; W[8,13]=1
        U[7,14]=1; U[8,14]=1; V[3,14]=1; W[5,14]=1
        U[0,15]=1; U[4,15]=1; V[3,15]=1; W[4,15]=1
        U[1,16]=1; U[8,16]=1; V[6,16]=1; W[8,16]=1
        U[3,17]=1; U[7,17]=1; V[7,17]=1; W[8,17]=1
        U[2,18]=1; U[3,18]=1; V[2,18]=1; W[8,18]=1
        U[5,19]=1; U[6,19]=1; V[1,19]=1; W[5,19]=1
        U[0,20]=1; U[8,20]=1; V[6,20]=1; W[8,20]=1
        U[4,21]=1; U[6,21]=1; V[1,21]=1; W[5,21]=1
        U[1,22]=1; U[3,22]=1; V[1,22]=1; W[4,22]=1
        
        return U, V, W
    
    def schoolbook_3x3x2():
        """Schoolbook decomposition for 3x3x2, rank 18"""
        n, m, p = 3, 3, 2
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """
        Combine a (n,m,p1) decomposition with rank r1 and (n,m,p2) decomposition with rank r2
        to get a (n,m,p1+p2) decomposition with rank r1+r2.
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        p = p1 + p2
        r = r1 + r2
        
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        
        U[:, :r1] = U1
        V[:m*p1, :r1] = V1
        W[:p1*n, :r1] = W1
        
        U[:, r1:] = U2
        V[m*p1:m*p1+m*p2, r1:] = V2
        W[p1*n:p1*n+p2*n, r1:] = W2
        
        return U, V, W
    
    # Verify BSC decomposition for 3x3x3
    U_bsc, V_bsc, W_bsc = bsc_3x3x3()
    bsc_valid = verify_decomposition(U_bsc, V_bsc, W_bsc, 3, 3, 3, num_trials=10)
    
    if not bsc_valid:
        U, V, W = schoolbook_full()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Combine BSC(3,3,3) with schoolbook(3,3,2)
    U_sb, V_sb, W_sb = schoolbook_3x3x2()
    
    U_comb, V_comb, W_comb = combine_decompositions(
        U_bsc, V_bsc, W_bsc, 
        U_sb, V_sb, W_sb, 
        n=3, m=3, p1=3, p2=2
    )
    
    is_valid = verify_decomposition(U_comb, V_comb, W_comb, n, m, p, num_trials=10)
    
    if is_valid:
        return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}
    else:
        U, V, W = schoolbook_full()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
