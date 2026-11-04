"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Block decomposition: 4x5x5 = (2+2)x(3+2)x(3+2)
    # C = [A1; A2] * [B1 B2] = [A1*B1, A1*B2; A2*B1, A2*B2]
    # where A1,A2 are 2x5, B1 is 5x3, B2 is 5x2
    
    # We need decompositions for:
    # 1. 2x3x3 (rank 16)
    # 2. 2x2x3 (rank 10) 
    # 3. 2x2x2 (rank 7)
    
    # Helper: create schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Helper: 2x2x2 Strassen decomposition (rank 7)
    def strassen_2x2x2():
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B11-B22)
        # M4 = A22(B22-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U: A indices (i,j) where i in {0,1}, j in {0,1}
        # V: B indices (j,k) where j in {0,1}, k in {0,1}
        # W: C indices (k,i) where k in {0,1}, i in {0,1}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11-A22)(B21+B22) -> indices: A[0,0],A[1,1] and B[1,0],B[1,1]
        # A11 is (0,0), A22 is (1,1)
        U[0, 0] = 1.0; U[3, 0] = -1.0
        V[2, 0] = 1.0; V[3, 0] = 1.0
        # C11 += M1, C22 -= M1
        W[0, 0] = 1.0; W[3, 0] = -1.0
        
        # M2 = (A21+A22)B11 -> A[1,0],A[1,1] and B[0,0]
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        # C21 += M2, C22 -= M2
        W[1, 1] = 1.0; W[3, 1] = -1.0
        
        # M3 = A11(B11-B22) -> A[0,0] and B[0,0],B[1,1]
        U[0, 2] = 1.0
        V[0, 2] = 1.0; V[3, 2] = -1.0
        # C12 += M3, C22 += M3
        W[0, 2] = 1.0; W[3, 2] = 1.0
        
        # M4 = A22(B22-B11) -> A[1,1] and B[1,1],B[0,0]
        U[3, 3] = 1.0
        V[3, 3] = 1.0; V[0, 3] = -1.0
        # C11 += M4, C21 += M4
        W[0, 3] = 1.0; W[1, 3] = 1.0
        
        # M5 = (A11+A12)B22 -> A[0,0],A[0,1] and B[1,1]
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        # C11 -= M5, C12 += M5
        W[0, 4] = -1.0; W[0, 4] += 1.0  # Wait, C12 is (0,1) which is index 1
        # Let me redo: C indices: C11=(0,0)->idx0, C12=(0,1)->idx1, C21=(1,0)->idx2, C22=(1,1)->idx3
        # Actually W[k*n+i] where k is col of C, i is row of C
        # C[i,k] so W[k*2+i]
        # C11: i=0,k=0 -> W[0]
        # C12: i=0,k=1 -> W[1]
        # C21: i=1,k=0 -> W[2]
        # C22: i=1,k=1 -> W[3]
        
        # Let me restart W
        W = np.zeros((4, 7))
        # M1: C11 += M1, C22 -= M1
        W[0, 0] = 1.0; W[3, 0] = -1.0
        # M2: C21 += M2, C22 -= M2
        W[2, 1] = 1.0; W[3, 1] = -1.0
        # M3: C12 += M3, C22 += M3
        W[1, 2] = 1.0; W[3, 2] = 1.0
        # M4: C11 += M4, C21 += M4
        W[0, 3] = 1.0; W[2, 3] = 1.0
        # M5: C11 -= M5, C12 += M5
        W[0, 4] = -1.0; W[1, 4] = 1.0
        # M6: (A21-A11)(B11+B12) -> A[1,0],A[0,0] and B[0,0],B[0,1]
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        # C22 += M6
        W[3, 5] = 1.0
        # M7: (A12-A22)(B21+B22) -> A[0,1],A[1,1] and B[1,0],B[1,1]
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        # C11 += M7
        W[0, 6] = 1.0
        
        return U, V, W, 7
    
    # Helper: 2x2x3 decomposition (rank 10)
    # This can be seen as 2x2x2 + 2x2x1, but let me construct it
    # Actually, for 2x2x3, we can use: split B into 2x1 and 2x2
    # Or use a direct construction
    def rank_2x2x3():
        # 2x2x3: A is 2x2, B is 2x3, C is 3x2
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # We can use Strassen-like ideas
        # For now, use schoolbook rank 12 and try to reduce
        # Actually, known rank for 2x2x3 is 10
        # Let me construct it explicitly
        
        # One approach: embed into 2x3x3 and use rank 16, but that's worse
        # Better: use the fact that 2x2x3 can be done in 10
        
        # I'll use a known construction
        # M1 = A11 * B11
        # M2 = A11 * B12  
        # M3 = A11 * B13
        # M4 = A22 * B21
        # M5 = A22 * B22
        # M6 = A22 * B23
        # M7 = (A11+A22) * (B11+B21)
        # M8 = (A11+A22) * (B12+B22)
        # M9 = (A11+A22) * (B13+B23)
        # M10 = (A12+A21) * (B11+B21)  # This doesn't seem right
        
        # Let me just use schoolbook for now and optimize later
        # Actually, let me try a different approach
        # 2x2x3: think of it as multiplying 2x2 by 2x3
        # We can write B = [B_left, B_right] where B_left is 2x1, B_right is 2x2
        # Then C = [A*B_left, A*B_right]
        # A*B_left: 2x2x1 (rank 4)
        # A*B_right: 2x2x2 (rank 7)
        # Total: 11, but we can do better
        
        # Known result: rank(2x2x3) = 10
        # I'll construct it using the following method:
        # Treat the 2x2x3 as a 2x3x3 with one dimension fixed
        
        # For simplicity, let me use a numerical approach to find rank 10
        # But that might be slow. Let me just use schoolbook rank 12 for now
        # and see if we can improve
        
        # Actually, let me try to construct rank 10 explicitly
        # Reference: the rank of 2x2x3 is indeed 10
        
        # I'll use the following construction (adapted from literature):
        # The 10 multiplications are:
        # m1 = a11*b11
        # m2 = a11*b12
        # m3 = a11*b13
        # m4 = a22*b21
        # m5 = a22*b22
        # m6 = a22*b23
        # m7 = (a11+a22)*(b11+b21)
        # m8 = (a11+a22)*(b12+b22)
        # m9 = (a11+a22)*(b13+b23)
        # m10 = (a12+a21)*(b12+b21)  # This needs verification
        
        # Let me just use schoolbook for now
        U, V, W, rank = schoolbook(2, 2, 3)
        return U, V, W, rank
    
    # For 2x3x3, known rank is 16
    # Let me construct it or use schoolbook (rank 18)
    def rank_2x3x3():
        # 2x3x3: A is 2x3, B is 3x3, C is 3x2
        # Schoolbook rank: 18
        # Known rank: 16
        # I'll use schoolbook for now
        U, V, W, rank = schoolbook(2, 3, 3)
        return U, V, W, rank
    
    # Now combine the blocks
    # C = [A1; A2] * [B1 B2]
    # A1 is 2x5, A2 is 2x5
    # B1 is 5x3, B2 is 5x2
    
    # Block 1: A1*B1 (2x5 * 5x3 = 2x3) -> 2x3x3 tensor, rank 18 (schoolbook)
    U1, V1, W1, r1 = rank_2x3x3()
    
    # Block 2: A1*B2 (2x5 * 5x2 = 2x2) -> 2x3x2 tensor, rank 10 (schoolbook)
    U2, V2, W2, r2 = schoolbook(2, 3, 2)
    
    # Block 3: A2*B1 (2x5 * 5x3 = 2x3) -> 2x3x3 tensor, rank 18 (schoolbook)
    U3, V3, W3, r3 = rank_2x3x3()
    
    # Block 4: A2*B2 (2x5 * 5x2 = 2x2) -> 2x3x2 tensor, rank 10 (schoolbook)
    U4, V4, W4, r4 = schoolbook(2, 3, 2)
    
    total_rank = r1 + r2 + r3 + r4
    
    # Now we need to assemble the full U, V, W matrices
    # U has shape (n*m, rank) = (4*5, total_rank) = (20, total_rank)
    # V has shape (m*p, rank) = (5*5, total_rank) = (25, total_rank)
    # W has shape (p*n, rank) = (5*4, total_rank) = (20, total_rank)
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Block 1: A1*B1
    # A1 is rows 0-1 of A, so U1 rows correspond to A indices (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    # In full U, these are at positions 0-5 (i*m+j where i in {0,1}, j in {0,1,2})
    # But wait, A is 4x5, so A[i,j] for i in 0..3, j in 0..4
    # A1 is A[0:2, 0:5], A2 is A[2:4, 0:5]
    # B1 is B[0:5, 0:3], B2 is B[0:5, 3:5]
    
    # U1 has shape (2*3, r1) = (6, r1) for A1 which is 2x3? No, A1 is 2x5
    # I made an error. A1 is 2x5, not 2x3.
    
    # Let me reconsider. The multiplication is:
    # C = A * B where A is 4x5, B is 5x5
    # Split A into A_top (2x5) and A_bottom (2x5)
    # Split B into B_left (5x3) and B_right (5x2)
    # Then C = [A_top*B_left, A_top*B_right; A_bottom*B_left, A_bottom*B_right]
    
    # A_top*B_left: 2x5 * 5x3 = 2x3, tensor is 2x5x3, rank 30 schoolbook
    # That's worse! I need to be more careful.
    
    # The issue is that when I split, the intermediate dimension (5) is still full.
    # So A_top*B_left is a 2x5x3 tensor, not 2x3x3.
    
    # I need a different approach. Let me use the fact that we can apply
    # low-rank methods to the full tensor directly.
    
    # Alternative: Use the schoolbook decomposition and then try to reduce rank
    # by finding linear dependencies.
    
    # For now, let me just return the schoolbook decomposition
    U, V, W, rank = schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
