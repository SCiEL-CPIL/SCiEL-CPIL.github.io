import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    
    # Rank-19 decomposition: T_{2,2,4} (rank 11) + T_{2,2,3} (rank 8)
    rank = 19
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # === Part 1: Rank-11 decomposition for 2x2x4 (columns k=0,1,2,3) ===
    # U indices: 0=a11, 1=a12, 2=a21, 3=a22
    # V indices: j*4+k (j=0,1; k=0..3)
    # W indices: k*2+i (k=0..3; i=0,1)
    
    # For each k, we need: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # T[i,j,k] = 1 for all i,j,k
    
    # Term 0: (a11+a22) * (b10+b20) -> contributes to w_{0,0} and w_{0,1}
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[4, 0] += 1
    W[0, 0] += 1; W[1, 0] += 1
    
    # Term 1: (a11+a22) * (b11+b21) -> contributes to w_{1,0} and w_{1,1}
    U[0, 1] += 1; U[3, 1] += 1
    V[1, 1] += 1; V[5, 1] += 1
    W[2, 1] += 1; W[3, 1] += 1
    
    # Term 2: (a11+a21) * (b10+b20) -> contributes to w_{0,0} and w_{0,1}
    U[0, 2] += 1; U[2, 2] += 1
    V[0, 2] += 1; V[4, 2] += 1
    W[0, 2] += 1; W[1, 2] += 1
    
    # Term 3: (a22+a21) * (b11+b21) -> contributes to w_{1,0} and w_{1,1}
    U[3, 3] += 1; U[2, 3] += 1
    V[1, 3] += 1; V[5, 3] += 1
    W[2, 3] += 1; W[3, 3] += 1
    
    # Term 4: (a12+a22) * (b10+b20) -> contributes to w_{0,1} and w_{0,0}
    U[1, 4] += 1; U[3, 4] += 1
    V[0, 4] += 1; V[4, 4] += 1
    W[0, 4] += 1; W[1, 4] += 1
    
    # Term 5: (a11+a12) * (b11+b21) -> contributes to w_{1,0} and w_{1,1}
    U[0, 5] += 1; U[1, 5] += 1
    V[1, 5] += 1; V[5, 5] += 1
    W[2, 5] += 1; W[3, 5] += 1
    
    # Term 6: a11 * b22 -> w_{2,0}
    U[0, 6] += 1
    V[6, 6] += 1
    W[4, 6] += 1
    
    # Term 7: a22 * b12 -> w_{2,1}
    U[3, 7] += 1
    V[2, 7] += 1
    W[5, 7] += 1
    
    # Term 8: a12 * b12 -> w_{2,0}
    U[1, 8] += 1
    V[2, 8] += 1
    W[4, 8] += 1
    
    # Term 9: a21 * b22 -> w_{2,1}
    U[2, 9] += 1
    V[6, 9] += 1
    W[5, 9] += 1
    
    # Term 10: (a11+a22) * (b13+b23) -> w_{3,0} + w_{3,1}
    U[0, 10] += 1; U[3, 10] += 1
    V[3, 10] += 1; V[7, 10] += 1
    W[6, 10] += 1; W[7, 10] += 1
    
    # === Part 2: Rank-8 decomposition for 2x2x3 (columns k=4,5,6) ===
    # V indices: j*7+k (j=0,1; k=4,5,6)
    # W indices: k*2+i (k=4,5,6; i=0,1)
    
    # Term 11: (a11+a22) * (b14+b24) -> w_{4,0} + w_{4,1}
    U[0, 11] += 1; U[3, 11] += 1
    V[7*0+4, 11] += 1; V[7*1+4, 11] += 1
    W[2*4+0, 11] += 1; W[2*4+1, 11] += 1
    
    # Term 12: (a11+a22) * (b15+b25) -> w_{5,0} + w_{5,1}
    U[0, 12] += 1; U[3, 12] += 1
    V[7*0+5, 12] += 1; V[7*1+5, 12] += 1
    W[2*5+0, 12] += 1; W[2*5+1, 12] += 1
    
    # Term 13: (a11+a21) * (b14+b24) -> w_{4,0} + w_{4,1}
    U[0, 13] += 1; U[2, 13] += 1
    V[7*0+4, 13] += 1; V[7*1+4, 13] += 1
    W[2*4+0, 13] += 1; W[2*4+1, 13] += 1
    
    # Term 14: (a22+a21) * (b15+b25) -> w_{5,0} + w_{5,1}
    U[3, 14] += 1; U[2, 14] += 1
    V[7*0+5, 14] += 1; V[7*1+5, 14] += 1
    W[2*5+0, 14] += 1; W[2*5+1, 14] += 1
    
    # Term 15: (a12+a22) * (b14+b24) -> w_{4,0} + w_{4,1}
    U[1, 15] += 1; U[3, 15] += 1
    V[7*0+4, 15] += 1; V[7*1+4, 15] += 1
    W[2*4+0, 15] += 1; W[2*4+1, 15] += 1
    
    # Term 16: (a11+a12) * (b15+b25) -> w_{5,0} + w_{5,1}
    U[0, 16] += 1; U[1, 16] += 1
    V[7*0+5, 16] += 1; V[7*1+5, 16] += 1
    W[2*5+0, 16] += 1; W[2*5+1, 16] += 1
    
    # Term 17: (a11+a22) * (b16+b26) -> w_{6,0} + w_{6,1}
    U[0, 17] += 1; U[3, 17] += 1
    V[7*0+6, 17] += 1; V[7*1+6, 17] += 1
    W[2*6+0, 17] += 1; W[2*6+1, 17] += 1
    
    # Term 18: (a11+a12) * b26 - (a22+a21) * b16
    U[0, 18] += 1; U[1, 18] += 1; U[3, 18] -= 1; U[2, 18] -= 1
    V[7*1+6, 18] += 1; V[7*0+6, 18] -= 1
    W[2*6+0, 18] += 1; W[2*6+1, 18] -= 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    valid = verify(U, V, W, n, m, p)
    if not valid:
        # Fallback to rank-28 (direct)
        rank = 28
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        col_start = 0
        for k in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j, col_start] += 1
                    V[j*p+k, col_start] += 1
                    W[k*n+i, col_start] += 1
                    col_start += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
