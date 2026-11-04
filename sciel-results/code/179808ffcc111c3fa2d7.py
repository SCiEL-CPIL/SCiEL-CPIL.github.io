import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Try to build a rank-reduced decomposition
    # Strategy: Use the structure of the problem
    
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # We need to find rank-r decomposition:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Wait, the convention is W[k*n+i, r] so W is indexed by (k,i)
    
    # Actually the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook: rank = n*m*p = 90
    # Each term c corresponds to a unique (i,j,k)
    
    # Let me try to find a better decomposition using known results
    
    # For now, let's implement Strassen-based approach for sub-blocks
    
    # Actually, let me think about this differently.
    # The bilinear complexity of 3x5x6 matrix multiplication.
    
    # One approach: use the fact that we can reduce the problem
    # by combining rows/columns.
    
    # Let me try a direct sum decomposition:
    # Split n=3 into n1=2, n2=1
    # Split m=5 into m1=2, m2=3  
    # Split p=6 into p1=3, p2=3
    
    # Then C = A*B has blocks:
    # C11 = A11*B11 + A12*B21 (2x3)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (1x3)
    # C22 = A21*B12 + A22*B22 (1x3)
    
    # For each 2x3x3 sub-problem, what's the rank?
    # 2x3x3: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1 and k=0,1,2
    # This is 2*3*3 = 18 schoolbook
    # Can we do better? 
    
    # For 2xm x p, one can use the "lattice" method or other approaches.
    # Actually, for 2x3x3, I believe the rank is 15 or so.
    
    # This is getting complex. Let me just implement a verified decomposition.
    
    # I'll use a hybrid: Strassen for 2x2x2 blocks where possible,
    # schoolbook for the rest.
    
    # Actually, let me try the simplest possible improvement:
    # Use the known rank-7 decomposition for 2x2x2.
    
    # Split: n=3 = 2+1, m=5 = 2+2+1, p=6 = 2+2+2
    # This gives many sub-blocks, too complex.
    
    # Let me just return schoolbook for now, but with a cleaner implementation
    # and then try to optimize.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
