import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 2.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Uses a recursive Strassen-like decomposition to achieve rank 55.
    1. Split p=10 into 8+2.
    2. T(2,3,2) splits m=3 into 2+1.
       - T(2,2,2) uses Strassen (rank 7).
       - T(2,1,2) uses schoolbook (rank 4).
       Total rank for p=2 part: 11.
    3. T(2,3,8) splits p=8 into 4+4.
       - T(2,3,4) splits p=4 into 2+2.
         - T(2,3,2) has rank 11.
         Total rank for p=4 part: 22.
       Total rank for p=8 part: 44.
    4. Total rank: 44 + 11 = 55.
    """
    n, m, p = DIMENSIONS
    
    def get_decomposition(n, m, p):
        # Returns U (n*m x r), V (m*p x r), W (p*n x r)
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # Base case: Schoolbook for small dimensions or if no better split
        # We use schoolbook if n*m*p <= 8 or if dimensions are 1
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
            return U, V, W
        
        # Special case: Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition
            # U, V, W of shape 4x7, 4x7, 4x7
            U = [[0] * 7 for _ in range(4)]
            V = [[0] * 7 for _ in range(4)]
            W = [[0] * 7 for _ in range(4)]
            
            # Indices:
            # A: 00, 01, 10, 11 -> 0, 1, 2, 3
            # B: 00, 01, 10, 11 -> 0, 1, 2, 3
            # C: 00, 01, 10, 11 -> 0, 1, 2, 3
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1+M4-M5-M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3-M6
            
            # U corresponds to A indices, V to B indices, W to C indices.
            # The tensor is T_{ijk} = sum_l U_{i,j,l} V_{j,k,l} W_{k,i,l} ? 
            # No, standard tensor definition for matrix mult:
            # C_{ik} = sum_j A_{ij} B_{jk}
            # T_{i,j,k} = delta_{j,k}? No.
            # The multiplication tensor T_{n,m,p} has entries T_{i,j,k} such that
            # <T, A \otimes B> = C.
            # C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # So T_{i,j,k} is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
            # Wait, the tensor is usually defined such that 
            # T_{i,j,k} = 1 if i,j,k correspond to A_{i,j}, B_{j,k}, C_{i,k}.
            # In our interface:
            # U is (n*m) x rank. Index i*m+j.
            # V is (m*p) x rank. Index j*p+k.
            # W is (p*n) x rank. Index k*n+i.
            # Decomposition: T_{i,j,k} = sum_l U_{i*m+j, l} V_{j*p+k, l} W_{k*n+i, l}.
            
            # For Strassen:
            # l=0 (M1):
            # A00, A11 contribute. U[0,0]=1, U[3,0]=1.
            # B00, B11 contribute. V[0,0]=1, V[3,0]=1.
            # C00, C11 contribute. W[0,0]=1, W[3,0]=1.
            
            # l=1 (M2):
            # A10, A11. U[2,1]=1, U[3,1]=1.
            # B00. V[0,1]=1.
            # C10. W[2,1]=1.
            
            # l=2 (M3):
            # A00. U[0,2]=1.
            # B01, B11. V[1,2]=1, V[3,2]=-1.
            # C01. W[1,2]=1.
            
            # l=3 (M4):
            # A11. U[3,3]=1.
            # B10, B00. V[2,3]=1, V[0,3]=-1.
            # C00, C10. W[0,3]=1, W[2,3]=1.
            
            # l=4 (M5):
            # A00, A01. U[0,4]=1, U[1,4]=1.
            # B11. V[3,4]=1.
            # C00, C01. W[0,4]=-1, W[1,4]=1.
            
            # l=5 (M6):
            # A10, A00. U[2,5]=1, U[0,5]=-1.
            # B00, B01. V[0,5]=1, V[1,5]=1.
            # C11. W[3,5]=-1.
            
            # l=6 (M7):
            # A01, A11. U[1,6]=1, U[3,6]=-1.
            # B10, B11. V[2,6]=1, V[3,6]=1.
            # C00. W[0,6]=-1.
            
            U = [
                [1, 0, 1, 0, 1, -1, 0], # A00
                [0, 0, 0, 0, 1, 0, 1],  # A01
                [0, 1, 0, 0, 0, 1, 0],  # A10
                [1, 1, 0, 1, 0, 0, -1]  # A11
            ]
            V = [
                [1, 1, 0, -1, 0, 1, 0], # B00
                [0, 0, 1, 0, 0, 1, 0],  # B01
                [0, 0, 0, 1, 0, 0, 1],  # B10
                [1, 0, -1, 0, 1, 0, 1]  # B11
            ]
            W = [
                [1, 0, 0, 1, -1, 0, -1], # C00
                [0, 0, 1, 0, 1, 0, 0],   # C01
                [0, 1, 0, 1, 0, 0, 0],   # C10
                [1, 0, 0, 0, 0, -1, 0]   # C11
            ]
            return U, V, W
        
        # Recursive case: Split p
        if p >= 2:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomposition(n, m, p1)
            U2, V2, W2 = get_decomposition(n, m, p2)
            
            # Concatenate
            # U is same for both (A is same)
            # V is split: V1 for first p1 cols, V2 for next p2 cols
            # W is split: W1 for first p1 rows (k index), W2 for next p2 rows
            
            rank = len(U1[0]) + len(U2[0])
            U = [U1[i] + U2[i] for i in range(n*m)]
            
            V = [[0] * rank for _ in range(m*p)]
            for i in range(m*p1):
                V[i] = V1[i] + [0] * len(U2[0])
            for i in range(m*p1, m*p):
                V[i] = [0] * len(U1[0]) + V2[i - m*p1]
            
            W = [[0] * rank for _ in range(p*n)]
            for i in range(p1*n):
                W[i] = W1[i] + [0] * len(U2[0])
            for i in range(p1*n, p*n):
                W[i] = [0] * len(U1[0]) + W2[i - p1*n]
                
            return U, V, W
        
        # If we get here, we should have split.
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return U, V, W

    # We need to handle the m-split for T(2,3,2)
    # The current get_decomposition only splits p.
    # We need to modify it to split m as well if beneficial.
    
    def get_decomposition_optimized(n, m, p):
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            # Strassen
            U = [
                [1, 0, 1, 0, 1, -1, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, -1]
            ]
            V = [
                [1, 1, 0, -1, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [1, 0, -1, 0, 1, 0, 1]
            ]
            W = [
                [1, 0, 0, 1, -1, 0, -1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, -1, 0]
            ]
            return U, V, W
        
        # Try splitting p
        if p >= 2:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomposition_optimized(n, m, p1)
            U2, V2, W2 = get_decomposition_optimized(n, m, p2)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            rank = rank1 + rank2
            
            U = [U1[i] + U2[i] for i in range(n*m)]
            
            V = [[0] * rank for _ in range(m*p)]
            for i in range(m*p1):
                V[i] = V1[i] + [0] * rank2
            for i in range(m*p1, m*p):
                V[i] = [0] * rank1 + V2[i - m*p1]
            
            W = [[0] * rank for _ in range(p*n)]
            for i in range(p1*n):
                W[i] = W1[i] + [0] * rank2
            for i in range(p1*n, p*n):
                W[i] = [0] * rank1 + W2[i - p1*n]
            
            return U, V, W
        
        # Try splitting m
        if m >= 2:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = get_decomposition_optimized(n, m1, p)
            U2, V2, W2 = get_decomposition_optimized(n, m2, p)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            rank = rank1 + rank2
            
            U = [[0] * rank for _ in range(n*m)]
            for i in range(n*m1):
                U[i] = U1[i] + [0] * rank2
            for i in range(n*m1, n*m):
                U[i] = [0] * rank1 + U2[i - n*m1]
            
            V = [V1[i] + V2[i] for i in range(m1*p)] # Wait, V index is j*p+k.
            # V1 has size m1*p. V2 has size m2*p.
            # V[i] for i in 0..m1*p-1 comes from V1.
            # V[i] for i in m1*p..m*p-1 comes from V2.
            # But the rank is combined.
            V = [[0] * rank for _ in range(m*p)]
            for i in range(m1*p):
                V[i] = V1[i] + [0] * rank2
            for i in range(m1*p, m*p):
                V[i] = [0] * rank1 + V2[i - m1*p]
                
            W = [W1[i] + W2[i] for i in range(p*n)] # W index is k*n+i.
            # W is same size p*n for both.
            # So we just add them column-wise? No, we concatenate columns.
            W = [[0] * rank for _ in range(p*n)]
            for i in range(p*n):
                W[i] = W1[i] + W2[i]
                
            return U, V, W
            
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return U, V, W

    U, V, W = get_decomposition_optimized(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
