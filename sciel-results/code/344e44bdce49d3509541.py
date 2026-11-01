"""
Rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Use Strassen's rank-7 algorithm for each of the 5 column-pairs (blocks 0-4).
The tensor decomposition for 2x2 matrix multiplication has a minimal rank of 7.
Since the 5 blocks are independent in the general tensor sense, the total rank is 5 * 7 = 35.
This is the theoretical minimum for this problem.
"""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 Strassen blocks (rank 7 each)
    rank = 35
    
    # U shape (n*m, rank) = (4, 35)
    # V shape (m*p, rank) = (20, 35)
    # W shape (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for a single 2x2 block
    # Indices in U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices in V for block t: 2t:(0,0), 2t+1:(0,1), 10+2t:(1,0), 11+2t:(1,1)
    # Note: V is (m*p, rank) = (20, rank). 
    # Row index in V: m*col + row? 
    # Task says: "output factor follows the official trace convention W[k*n+i,r]".
    # Usually for V (input B), if B is 2x10, and we index B[j,k], 
    # the flat index is j*p + k or k*m + j?
    # Standard row-major: B[j][k] is at j*p + k.
    # Here m=2, p=10. V has 20 rows.
    # Block t corresponds to columns 2t and 2t+1 of B.
    # B[0, 2t] is at 0*10 + 2t = 2t.
    # B[0, 2t+1] is at 0*10 + 2t + 1 = 2t + 1.
    # B[1, 2t] is at 1*10 + 2t = 10 + 2t.
    # B[1, 2t+1] is at 1*10 + 2t + 1 = 11 + 2t.
    # This matches the indices used in the previous code:
    # V[2*t], V[2*t+1], V[10+2*t], V[11+2*t].
    
    # Indices in W for block t:
    # W is (p*n, rank) = (20, rank).
    # C is 2x10. C[i, k].
    # Flat index for C[i, k] is i*p + k? Or k*n + i?
    # Task says: "W[k*n+i,r]". This implies column-major indexing for W?
    # "W[k*n+i,r]" means the row index is k*n + i.
    # k is the column index of C (0..9), i is the row index of C (0..1).
    # So for block t (columns 2t, 2t+1):
    # C[0, 2t] is at (2t)*2 + 0 = 4t.
    # C[1, 2t] is at (2t)*2 + 1 = 4t + 1.
    # C[0, 2t+1] is at (2t+1)*2 + 0 = 4t + 2.
    # C[1, 2t+1] is at (2t+1)*2 + 1 = 4t + 3.
    # This matches the indices used in the previous code:
    # W[4*t], W[4*t+1], W[4*t+2], W[4*t+3].

    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a00+a11)*(b00+b11) -> c00, c11
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (a10+a11)*b00 -> c01, -c11
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a00*(b01-b11) -> c10, c11
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # Term 4: P4 = a11*(b10-b00) -> c00, c01
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # Term 5: P5 = (a00+a01)*b11 -> -c00, c10
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # Term 6: P6 = (a10-a00)*(b00+b01) -> c11
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # Term 7: P7 = (a01-a11)*(b10+b11) -> c00
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
