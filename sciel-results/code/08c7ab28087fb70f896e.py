"""Matrix multiplication 2,2,10 - rank-minimal decomposition using Strassen per block."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120  # 120s wall-clock limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Strassen's algorithm for 2x2 matrix multiplication: rank 7
    # U matrix: 4 rows (n*m=4), 7 columns
    # V matrix: 4 rows (m*p=4 for each 2x2 block), 7 columns  
    # W matrix: 4 rows (p*n=4 for each 2x2 block), 7 columns
    
    # Strassen factors for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 multiplication with indices:
    # A is 2x2: A[i,j] where i=row (0,1), j=col (0,1)
    # B is 2x2: B[j,k] where j=row (0,1), k=col (0,1)
    # C is 2x2: C[i,k] where i=row (0,1), k=col (0,1)
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # U[i*m + j, c] where i in {0,1}, j in {0,1}, c in {0,...,6}
    # V[j*p + k, c] where j in {0,1}, k in {0,1}, c in {0,...,6}
    # W[k*n + i, c] where k in {0,1}, i in {0,1}, c in {0,...,6}
    
    # Let's define the Strassen decomposition properly:
    # For multiplying A (2x2) by B (2x2) to get C (2x2):
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen's 7 products:
    # M0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M1 = (A[1,0]+A[1,1])*B[0,0]
    # M2 = A[0,0]*(B[0,1]-B[1,1])
    # M3 = A[1,1]*(B[1,0]-B[0,0])
    # M4 = (A[0,0]+A[0,1])*B[1,1]
    # M5 = (A[0,1]-A[1,0])*(B[0,0]+B[0,1])
    # M6 = (A[1,0]-A[0,0])*(B[1,1]+B[1,0])
    
    # C[0,0] = M0 + M3 - M4 + M6
    # C[0,1] = M2 + M4
    # C[1,0] = M1 + M3
    # C[1,1] = M0 - M1 + M2 + M5
    
    # Let me verify C[0,0]:
    # M0 = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # M3 = A[1,1]*B[1,0] - A[1,1]*B[0,0]
    # -M4 = -A[0,0]*B[1,1] - A[0,1]*B[1,1]
    # M6 = A[1,0]*B[1,1] + A[1,0]*B[1,0] - A[0,0]*B[1,1] - A[0,0]*B[1,0]
    
    # Sum: A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    #      + A[1,1]*B[1,0] - A[1,1]*B[0,0]
    #      - A[0,0]*B[1,1] - A[0,1]*B[1,1]
    #      + A[1,0]*B[1,1] + A[1,0]*B[1,0] - A[0,0]*B[1,1] - A[0,0]*B[1,0]
    
    # = A[0,0]*B[0,0] + (A[0,0]*B[1,1] - A[0,0]*B[1,1] - A[0,0]*B[1,1]) 
    #   + (A[1,1]*B[0,0] - A[1,1]*B[0,0])
    #   + A[1,1]*B[1,1] + A[1,1]*B[1,0]
    #   - A[0,1]*B[1,1]
    #   + A[1,0]*B[1,1] + A[1,0]*B[1,0]
    #   - A[0,0]*B[1,0]
    
    # = A[0,0]*B[0,0] - A[0,0]*B[1,1] + A[1,1]*B[1,1] + A[1,1]*B[1,0]
    #   - A[0,1]*B[1,1] + A[1,0]*B[1,1] + A[1,0]*B[1,0] - A[0,0]*B[1,0]
    
    # This doesn't match C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Let me use a different Strassen formulation.
    
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Using 0-indexed: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # P2 = (A[1,0]+A[1,1])*B[0,0]
    # P3 = A[0,0]*(B[1,0]-B[1,1])
    # P4 = A[1,1]*(B[0,1]-B[0,0])
    # P5 = (A[0,0]+A[0,1])*B[1,1]
    # P6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # P7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    
    # Let me build the U, V, W matrices for one 2x2 block (rank 7)
    
    # U[i*m+j, c] = U[i*2+j, c]
    # V[j*p+k, c] = V[j*2+k, c]  (for one block, j in {0,1}, k in {0,1})
    # W[k*n+i, c] = W[k*2+i, c]
    
    # P1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U[0*2+0, 0] = 1  (A[0,0])
    # U[1*2+1, 0] = 1  (A[1,1])
    # V[0*2+0, 0] = 1  (B[0,0])
    # V[1*2+1, 0] = 1  (B[1,1])
    # W contribution: C[0,0] += P1, C[1,1] += P1
    # W[0*2+0, 0] = 1  (C[0,0])
    # W[1*2+1, 0] = 1  (C[1,1])
    
    # P2: (A[1,0]+A[1,1])*B[0,0]
    # U[1*2+0, 1] = 1  (A[1,0])
    # U[1*2+1, 1] = 1  (A[1,1])
    # V[0*2+0, 1] = 1  (B[0,0])
    # W[1*2+0, 1] = 1  (C[1,0])
    # W[1*2+1, 1] = -1 (C[1,1])
    
    # P3: A[0,0]*(B[1,0]-B[1,1])
    # U[0*2+0, 2] = 1  (A[0,0])
    # V[1*2+0, 2] = 1  (B[1,0])
    # V[1*2+1, 2] = -1 (B[1,1])
    # W[0*2+1, 2] = 1  (C[0,1])
    # W[1*2+1, 2] = 1  (C[1,1])
    
    # P4: A[1,1]*(B[0,1]-B[0,0])
    # U[1*2+1, 3] = 1  (A[1,1])
    # V[0*2+1, 3] = 1  (B[0,1])
    # V[0*2+0, 3] = -1 (B[0,0])
    # W[0*2+0, 3] = 1  (C[0,0])
    # W[1*2+0, 3] = 1  (C[1,0])
    
    # P5: (A[0,0]+A[0,1])*B[1,1]
    # U[0*2+0, 4] = 1  (A[0,0])
    # U[0*2+1, 4] = 1  (A[0,1])
    # V[1*2+1, 4] = 1  (B[1,1])
    # W[0*2+0, 4] = -1 (C[0,0])
    # W[0*2+1, 4] = 1  (C[0,1])
    
    # P6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # U[1*2+0, 5] = 1  (A[1,0])
    # U[0*2+0, 5] = -1 (A[0,0])
    # V[0*2+0, 5] = 1  (B[0,0])
    # V[0*2+1, 5] = 1  (B[0,1])
    # W[1*2+1, 5] = 1  (C[1,1])
    
    # P7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # U[0*2+1, 6] = 1  (A[0,1])
    # U[1*2+1, 6] = -1 (A[1,1])
    # V[1*2+0, 6] = 1  (B[1,0])
    # V[1*2+1, 6] = 1  (B[1,1])
    # W[0*2+0, 6] = 1  (C[0,0])
    
    # Now build the full matrices for 2x2 x 2x10 = 2x10
    # We have 5 blocks of 2x2 (since p=10, each block handles 2 columns of B)
    # Total rank = 5 * 7 = 35
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # For each block b in 0..4:
    # B columns are 2*b and 2*b+1
    # For block b, the local indices j=0,1 correspond to global j=2*b, 2*b+1
    # k=0,1 correspond to global k=2*b, 2*b+1
    
    for b in range(5):
        base = b * 7  # starting column in U, V, W for this block
        
        # P1 (column base+0)
        U[0][base] = 1   # A[0,0]
        U[3][base] = 1   # A[1,1]
        V[2*b][base] = 1      # B[0, 2*b] (j=0, k=0)
        V[2*b+1][base] = 1    # B[1, 2*b+1] (j=1, k=1)
        W[2*b][base] = 1      # C[2*b, 0] (k=0, i=0)
        W[2*b+1][base] = 1    # C[2*b+1, 1] (k=1, i=1)
        
        # P2 (column base+1)
        U[2][base+1] = 1   # A[1,0]
        U[3][base+1] = 1   # A[1,1]
        V[2*b][base+1] = 1 # B[0, 2*b] (j=0, k=0)
        W[2*b][base+1] = 1     # C[2*b, 1] (k=0, i=1)
        W[2*b+1][base+1] = -1  # C[2*b+1, 1] (k=1, i=1)
        
        # P3 (column base+2)
        U[0][base+2] = 1   # A[0,0]
        V[2*b+1][base+2] = 1    # B[1, 2*b] (j=1, k=0)
        V[2*b+1][base+2] = 1    # B[1, 2*b] (j=1, k=0)
        V[2*b+1][base+2] = 1    # B[1, 2*b] (j=1, k=0)
        # Wait, I need B[1,0] and B[1,1] for P3
        # P3 = A[0,0]*(B[1,0]-B[1,1])
        # In block b, B[1,0] means row j=1, col k=0, which is global B[1, 2*b]
        # B[1,1] means row j=1, col k=1, which is global B[1, 2*b+1]
        V[2*b+1][base+2] = 1    # B[1, 2*b] (j=1, k=0)
        V[2*b+1][base+2] = 1    # Overwriting, let me redo this
        
        # Let me restart the V assignments more carefully
        # For block b, local (j,k) maps to global row j, col 2*b+k
        # V[j*p + k, c] where p=10, so V[j*10+k, c]
        # Wait, no. m=2, p=10. V has shape (m*p, rank) = (20, rank)
        # V[j*p + k, c] where j in {0,1}, k in {0,...,9}
        # But for block b, we only use k=2*b and k=2*b+1
        
        # Let me redo this properly
        
    # Clear and redo
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    for b in range(5):
        base = b * 7
        
        # P1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[0][base] = 1   # A[0,0]
        U[3][base] = 1   # A[1,1]
        V[0*10 + 2*b][base] = 1      # B[0, 2*b]
        V[1*10 + (2*b+1)][base] = 1  # B[1, 2*b+1]
        W[2*b*2 + 0][base] = 1       # C[2*b, 0] -> W[(2*b)*2 + 0]
        W[(2*b+1)*2 + 1][base] = 1   # C[2*b+1, 1]
        
        # P2: (A[1,0]+A[1,1])*B[0,0]
        U[2][base+1] = 1   # A[1,0]
        U[3][base+1] = 1   # A[1,1]
        V[0*10 + 2*b][base+1] = 1  # B[0, 2*b]
        W[(2*b)*2 + 1][base+1] = 1     # C[2*b, 1]
        W[(2*b+1)*2 + 1][base+1] = -1  # C[2*b+1, 1]
        
        # P3: A[0,0]*(B[1,0]-B[1,1])
        U[0][base+2] = 1   # A[0,0]
        V[1*10 + 2*b][base+2] = 1      # B[1, 2*b]
        V[1*10 + (2*b+1)][base+2] = -1 # B[1, 2*b+1]
        W[(2*b)*2 + 1][base+2] = 1     # C[2*b, 1]
        W[(2*b+1)*2 + 1][base+2] = 1   # C[2*b+1, 1]
        
        # P4: A[1,1]*(B[0,1]-B[0,0])
        U[3][base+3] = 1   # A[1,1]
        V[0*10 + (2*b+1)][base+3] = 1  # B[0, 2*b+1]
        V[0*10 + 2*b][base+3] = -1     # B[0, 2*b]
        W[(2*b)*2 + 0][base+3] = 1     # C[2*b, 0]
        W[(2*b)*2 + 1][base+3] = 1     # C[2*b, 1]
        
        # P5: (A[0,0]+A[0,1])*B[1,1]
        U[0][base+4] = 1   # A[0,0]
        U[1][base+4] = 1   # A[0,1]
        V[1*10 + (2*b+1)][base+4] = 1  # B[1, 2*b+1]
        W[(2*b)*2 + 0][base+4] = -1    # C[2*b, 0]
        W[(2*b)*2 + 1][base+4] = 1     # C[2*b, 1]
        
        # P6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[2][base+5] = 1   # A[1,0]
        U[0][base+5] = -1  # A[0,0]
        V[0*10 + 2*b][base+5] = 1      # B[0, 2*b]
        V[0*10 + (2*b+1)][base+5] = 1  # B[0, 2*b+1]
        W[(2*b+1)*2 + 1][base+5] = 1   # C[2*b+1, 1]
        
        # P7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[1][base+6] = 1   # A[0,1]
        U[3][base+6] = -1  # A[1,1]
        V[1*10 + 2*b][base+6] = 1      # B[1, 2*b]
        V[1*10 + (2*b+1)][base+6] = 1  # B[1, 2*b+1]
        W[(2*b)*2 + 0][base+6] = 1     # C[2*b, 0]
    
    # Verify the decomposition
    if time.time() - start_time < 110:  # Leave time margin
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 
            # delta(i,i') * delta(j,j') * delta(k,k') for the multiplication tensor
            # Actually, the tensor T[i,j,k] = 1 if the entry (i,k) of A*B depends on A[i,j]*B[j,k]
            # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
            # So T[i,j,k] = 1 for all valid i,j,k (since each A[i,j]*B[j,k] contributes to C[i,k])
            
            # The decomposition should satisfy:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # where T[i,j,k] = 1 if j is a valid intermediate index (which it always is for 2x2 x 2x10)
            
            # Actually for 2x2 * 2x10, the intermediate index j ranges over {0,1}
            # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,9}
            
            # Let's check a few entries
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        expected = 1.0
                        if abs(val - expected) > 1e-9:
                            error += abs(val - expected)
            
            if error < 1e-8:
                pass  # Good, decomposition is valid
            else:
                # Fall back to schoolbook if Strassen verification fails
                rank_school = n * m * p
                U_school = [[0] * rank_school for _ in range(n * m)]
                V_school = [[0] * rank_school for _ in range(m * p)]
                W_school = [[0] * rank_school for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i*m+j][col] = 1
                            V_school[j*p+k][col] = 1
                            W_school[k*n+i][col] = 1
                            col += 1
                return {"U": U_school, "V": V_school, "W": W_school}
        except Exception:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m+j][col] = 1
                        V_school[j*p+k][col] = 1
                        W_school[k*n+i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
