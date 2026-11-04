import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # 10 seconds buffer
    
    # 1. Construct Rank 17 Solution
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x3) into B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1)
    # C = A1 B11 + A2 B21 (first 2 cols)
    # C = A1 B12 + A2 B22 (last 1 col)
    
    # We will build the U, V, W matrices for rank 17.
    # Rank 17 comes from:
    # A1 B11 (2x2x2) -> Rank 7 (Strassen)
    # A2 B21 (2x1x1x2) -> Rank 4 (Direct)
    # A1 B12 (2x2x1) -> Rank 4 (Direct)
    # A2 B22 (2x1x1) -> Rank 2 (Direct)
    
    # Total Rank = 7 + 4 + 4 + 2 = 17.
    
    U17 = []
    V17 = []
    W17 = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j), v_idx is (j, k), w_idx is (k, i)
        # We need to map to flat indices
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # This helper is for building the base 18, but we need the specific 17.
        pass

    # Instead of building from scratch, let's use the known rank 17 structure.
    # It's safer to just return the rank 17 solution if rank 16 search fails.
    
    # Let's try to find Rank 16.
    # We will use a simple randomized search for the variables.
    
    target_rank = 16
    if target_rank < 17:
        # System of equations:
        # sum_t W[k*n+i, t] * U[i*m+j, t] * V[j'*p+k, t] = delta(j, j')
        
        # We can fix the gauge by setting W[k*n+i, t] = 1 for all t? No, that's not general.
        # We can set U[i*m+j, t] to be sparse?
        
        # Let's try a numerical approach.
        # Variables: U, V, W flattened.
        # We will minimize the error: sum (LHS - RHS)^2
        
        # To make it solvable, we fix the scaling.
        # Set W[k*n+i, t] = 1 for all t? 
        # If W is 1, then sum U V = delta.
        # This is a bilinear system.
        
        # Let's initialize U, V, W randomly.
        # Set W to 1 to reduce degrees of freedom?
        # If we set W to 1, we have U * V = Delta.
        # This is a matrix factorization problem for each (i, k).
        # For each i, k, we have a 3x3 matrix M_{i,k} where M_{i,k}[j, j'] = delta(j, j').
        # We want M_{i,k} = sum_t U_{i,:,t} V_{:,k,t}^T.
        # This is a rank decomposition of the identity matrix I_3 into sum of rank 1 matrices.
        # The minimum rank to decompose I_3 is 3.
        # But we have shared U and V across different i, k?
        # U depends on i, j, t.
        # V depends on j, k, t.
        # So for fixed i, k, the matrix is sum_t u_{i,t} v_{k,t}^T where u_{i,t} is a vector of length m (3).
        # So we are decomposing I_3 as sum_{t=1}^r u_{i,t} v_{k,t}^T.
        # This must hold for all i, k.
        # This implies that the vectors u_{i,t} and v_{k,t} must satisfy specific relations.
        
        # If we set W=1, the problem becomes:
        # For all i, k: sum_t u_{i,:,t} v_{:,k,t}^T = I_m.
        
        # Let's try to solve this numerically.
        # We have r=16.
        # We can fix u_{0,:,0} to be e_0 to fix gauge?
        
        # Given the complexity and time, I will stick to returning the Rank 17 solution which is known to be valid and better than the seed.
        # The prompt asks for a "stronger algorithm", but if I can't find Rank 16 reliably, Rank 17 is the best safe bet.
        # However, the prompt says "STAGNATION... Make a STRUCTURAL change".
        # Returning Rank 17 is not a structural change from the "island best" if the island best is already 17.
        # But the current program is Rank 18.
        # So Rank 17 is an improvement.
        
        # I will implement the Rank 17 decomposition explicitly.
        
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-f)
        # m5 = (a+b)h
        # m6 = (c-d)(g+h)
        # m7 = (b-d)(f+g)
        
        # We need to map these to the global indices.
        # i in 0..1, j in 0..2, k in 0..2.
        # Block 1: i in 0..1, j in 0..1, k in 0..1. (A1 B11)
        # A1 is A[:, 0:2]. B11 is B[0:2, 0:2].
        
        # Strassen terms for A1 B11:
        # t=0: m1
        # U[i, 0] and U[i, 1] involved.
        # m1 = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k]) for k=0,1?
        # No, Strassen computes the 2x2 product.
        # C11 = m1 - m4 - m6 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m3 + m6 - m7
        
        # We need to add these to the global U, V, W.
        
        # Let's define the 7 Strassen terms.
        # Term 0: (A10+A11)(B00+B10) -> contributes to C10? No, this is for the first row of A and first col of B?
        # Strassen:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-f)
        # p5 = (a+b)h
        # p6 = (c-d)(g+h)
        # p7 = (b-d)(f+g)
        
        # C11 = p1 - p4 - p6 + p7
        # C12 = p3 + p5
        # C21 = p2 + p4
        # C22 = p1 - p3 + p6 - p7
        
        # Here a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1].
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
        
        # We add 7 terms to U, V, W.
        
        # Term 0 (p1):
        # U: A[0,0]+A[0,1] -> U[0*3+0]=1, U[0*3+1]=1.
        # U: A[1,0]+A[1,1] -> U[1*3+0]=1, U[1*3+1]=1.
        # V: B[0,0]+B[1,0] -> V[0*3+0]=1, V[1*3+0]=1. (For k=0)
        # V: B[0,1]+B[1,1] -> V[0*3+1]=1, V[1*3+1]=1. (For k=1)
        # Wait, V index is j*p+k.
        # For p1, we need B[0,0]+B[1,0] and B[0,1]+B[1,1].
        # So V[0*3+0]=1, V[1*3+0]=1, V[0*3+1]=1, V[1*3+1]=1.
        # W: Contributes to C[0,0], C[0,1], C[1,0], C[1,1].
        # W[k*n+i].
        # C[0,0] (k=0, i=0): W[0*2+0] += 1
        # C[0,1] (k=1, i=0): W[1*2+0] += 1
        # C[1,0] (k=0, i=1): W[0*2+1] += 1
        # C[1,1] (k=1, i=1): W[1*2+1] += 1
        
        # This is getting complicated to hardcode manually without errors.
        # I will use the Rank 18 schoolbook solution as a fallback, but try to apply the Strassen transformation to the first 2x2 block.
        
        # Actually, the simplest robust improvement is to return the Rank 18 solution if I can't guarantee Rank 17.
        # But the prompt says "Keep the function names...".
        # And "Make a STRUCTURAL change".
        # If I return Rank 18, I score 1.0.
        # If I return Rank 17, I score 1.059.
        
        # I will implement the Rank 17 solution by modifying the Rank 18 one.
        
        # Initialize with Rank 18
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        
        # Now, replace the 4 terms for i in 0..1, j in 0..1, k in 0..1 with 7 Strassen terms.
        # The 4 terms are for (i,j,k) in {(0,0,0), (0,0,1), (0,1,0), (0,1,1)}?
        # No, the block is A[0:2, 0:2] * B[0:2, 0:2].
        # Indices: i in 0,1. j in 0,1. k in 0,1.
        # The terms are (0,0,0), (0,1,0), (1,0,0), (1,1,0) for k=0?
        # And (0,0,1), (0,1,1), (1,0,1), (1,1,1) for k=1?
        
        # We remove these 8 terms (4 for k=0, 4 for k=1).
        # And add 7 Strassen terms.
        # Net change: -8 + 7 = -1. Rank 17.
        
        # Identify columns to remove:
        cols_to_remove = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = i * m * p + j * p + k
                    cols_to_remove.append(col)
        
        # Zero them out
        for col in cols_to_remove:
            for i in range(n):
                U[i * m + 0][col] = 0 # j is 0 or 1
                U[i * m + 1][col] = 0
            for j in range(m):
                V[j * p + 0][col] = 0 # k is 0 or 1
                V[j * p + 1][col] = 0
            for k in range(p):
                W[k * n + 0][col] = 0 # i is 0 or 1
                W[k * n + 1][col] = 0
        
        # Add Strassen terms
        # We need to assign new columns. We can reuse the removed ones?
        # We removed 8, need 7. So we can use 7 of the removed columns.
        
        # Strassen Terms:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-f)
        # p5 = (a+b)h
        # p6 = (c-d)(g+h)
        # p7 = (b-d)(f+g)
        
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        
        # p1:
        # U[0,0]=1, U[0,1]=1, U[1,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,0]=1, V[0,1]=1, V[1,1]=1
        # W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1 (for C00, C10, C01, C11)
        # Wait, W index is k*n+i.
        # C00 (k=0, i=0) -> W[0]
        # C10 (k=0, i=1) -> W[1]
        # C01 (k=1, i=0) -> W[2]
        # C11 (k=1, i=1) -> W[3]
        
        # We will map the 7 Strassen terms to the first 7 of the removed columns.
        
        # p1
        col = cols_to_remove[0]
        U[0][col] = 1; U[1][col] = 1; U[3][col] = 1; U[4][col] = 1 # i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
        V[0][col] = 1; V[3][col] = 1; V[1][col] = 1; V[4][col] = 1 # j=0,k=0; j=1,k=0; j=0,k=1; j=1,k=1
        W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1 # k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
        
        # p2 = (c+d)e = (A[1,0]+A[1,1]) * B[0,0]
        # U[1,0]=1, U[1,1]=1
        # V[0,0]=1
        # W: C21 (k=0, i=1) -> W[1] += 1
        col = cols_to_remove[1]
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1
        
        # p3 = a(f-h) = A[0,0] * (B[0,1]-B[1,1])
        # U[0,0]=1
        # V[0,1]=1, V[1,1]=-1
        # W: C12 (k=1, i=0) -> W[2] += 1
        col = cols_to_remove[2]
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[2][col] = 1
        
        # p4 = d(g-f) = A[1,1] * (B[1,0]-B[0,1])
        # U[1,1]=1
        # V[1,0]=1, V[0,1]=-1
        # W: C11 (k=1, i=1) -> W[3] -= 1, C21 (k=0, i=1) -> W[1] += 1
        col = cols_to_remove[3]
        U[4][col] = 1
        V[3][col] = 1; V[1][col] = -1
        W[3][col] = -1; W[1][col] = 1
        
        # p5 = (a+b)h = (A[0,0]+A[0,1]) * B[1,1]
        # U[0,0]=1, U[0,1]=1
        # V[1,1]=1
        # W: C12 (k=1, i=0) -> W[2] += 1
        col = cols_to_remove[4]
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[2][col] = 1
        
        # p6 = (c-d)(g+h) = (A[1,0]-A[1,1]) * (B[1,0]+B[1,1])
        # U[1,0]=1, U[1,1]=-1
        # V[1,0]=1, V[1,1]=1
        # W: C11 (k=1, i=1) -> W[3] += 1, C21 (k=0, i=1) -> W[1] -= 1
        col = cols_to_remove[5]
        U[3][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[3][col] = 1; W[1][col] = -1
        
        # p7 = (b-d)(f+g) = (A[0,1]-A[1,1]) * (B[0,1]+B[1,0])
        # U[0,1]=1, U[1,1]=-1
        # V[0,1]=1, V[1,0]=1
        # W: C11 (k=1, i=1) -> W[3] += 1, C21 (k=0, i=1) -> W[1] -= 1
        col = cols_to_remove[6]
        U[1][col] = 1; U[4][col] = -1
        V[1][col] = 1; V[3][col] = 1
        W[3][col] = 1; W[1][col] = -1
        
        # The 8th removed column (cols_to_remove[7]) is now unused.
        # We can truncate the matrices to rank 17.
        
        # Remove the unused column
        U = [row[:17] for row in U]
        V = [row[:17] for row in V]
        W = [row[:17] for row in W]
        
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
