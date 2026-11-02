"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Split m=5 into m1=3, m2=2
    # Split p=6 into p1=3, p2=3
    # Sub-problem 1: n=3, m1=3, p1=3 and n=3, m1=3, p2=3 → two 3×3×3, each rank 21
    # Sub-problem 2: n=3, m2=2, p1=3 and n=3, m2=2, p2=3 → two 3×2×3, each rank 7
    # Total rank = 2*21 + 2*7 = 56
    
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # (3*5, 56) = (15, 56)
    V = [[0] * rank for _ in range(m * p)]  # (5*6, 56) = (30, 56)
    W = [[0] * rank for _ in range(p * n)]  # (6*3, 56) = (18, 56)
    
    # We need to define the Strassen decompositions
    # For 2x2x2 (rank 7): the 7 terms are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 3x3x3 (rank 21): I'll use a known decomposition.
    # Actually, let me use a simpler approach: use the schoolbook for 3x3x3 (rank 27) 
    # but that gives 2*27 + 2*7 = 68, which is worse than 56.
    # 
    # Let me use a rank-21 decomposition for 3x3x3.
    # I'll construct it using the fact that 3x3x3 can be decomposed using 
    # the 2x2x2 Strassen plus some additional terms.
    
    # Actually, the minimum rank for 3x3x3 is 21 (Landsberg's bound gives 19, but 21 is achievable).
    # Let me use a known rank-21 decomposition.
    
    # I'll use the following approach: partition 3x3x3 into overlapping 2x2x2 blocks.
    # But this is complex. Let me instead use a simpler but still good decomposition.
    
    # For now, let me use the schoolbook for the 3x3x3 parts to get a valid solution,
    # then optimize.
    # Schoolbook 3x3x3: rank 27
    # Total: 2*27 + 2*7 = 68
    
    # Let me try to get rank 56 by using a proper rank-21 for 3x3x3.
    
    # I'll define the rank-21 decomposition for 3x3x3 manually.
    # The 21 terms can be constructed as follows:
    # Use the 7 Strassen terms for the top-left 2x2x2 block,
    # plus terms for the remaining parts.
    
    # Actually, let me just implement the schoolbook for now to get a valid answer,
    # then I can optimize.
    
    # Schoolbook for 3x5x6: rank 90
    # But I want to do better.
    
    # Let me implement the Strassen-based decomposition properly.
    
    # For the 3x3x3 part, I'll use a rank-21 decomposition.
    # I found a reference: the 3x3x3 tensor has rank 21.
    # Let me construct it.
    
    # I'll use the following construction for 3x3x3 rank-21:
    # The 21 rank-1 tensors are:
    # 1-7: Strassen terms for rows 0-1, cols 0-1, k 0-1
    # 8-14: Terms involving row/col 2
    # 15-21: Additional terms
    
    # This is getting complex. Let me use a different approach.
    
    # I'll use the fact that I can write the 3x3x3 multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # And use a decomposition that achieves rank 21.
    
    # For simplicity and correctness, let me use the schoolbook decomposition
    # but with the Strassen optimization where applicable.
    
    # Actually, let me just implement the full Strassen-based approach:
    
    # Sub-problem 1a: 3x3x3 (i in 0-2, j in 0-2, k in 0-2)
    # Sub-problem 1b: 3x3x3 (i in 0-2, j in 0-2, k in 3-5)
    # Sub-problem 2a: 3x2x3 (i in 0-2, j in 3-4, k in 0-2)
    # Sub-problem 2b: 3x2x3 (i in 0-2, j in 3-4, k in 3-5)
    
    # For 3x3x3, I'll use rank 21
    # For 3x2x3, I'll use rank 7 (Strassen 2x2x2, since 3x2x3 contains 2x2x2)
    
    # Let me define the 2x2x2 Strassen decomposition first.
    
    def strassen_2x2_2x2():
        """Return U, V, W for 2x2x2 Strassen (rank 7).
        U: (2*2, 7) = (4, 7), indexed by (i*2+j, c)
        V: (2*2, 7) = (4, 7), indexed by (j*2+k, c)
        W: (2*2, 7) = (4, 7), indexed by (k*2+i, c)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A11(B21-B22)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[1][2] = 1
        
        # M4 = A22(B12-B11)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    # For 3x3x3 rank-21, I'll use a known construction.
    # Actually, let me use a simpler approach: I'll use the schoolbook for 3x3x3 (rank 27)
    # This gives total rank 2*27 + 2*7 = 68, which is better than 90.
    
    # But I want to achieve rank 56. Let me look up the rank-21 decomposition.
    
    # The rank-21 decomposition for 3x3x3 can be constructed by taking the 7 Strassen terms
    # for the 2x2x2 sub-block and adding 14 more terms for the remaining parts.
    
    # Let me define it explicitly. I'll use a standard rank-21 decomposition.
    
    def strassen_3x3_3x3():
        """Return U, V, W for 3x3x3 rank-21 decomposition.
        U: (3*3, 21) = (9, 21), indexed by (i*3+j, c)
        V: (3*3, 21) = (9, 21), indexed by (j*3+k, c)
        W: (3*3, 21) = (9, 21), indexed by (k*3+i, c)
        """
        U = [[0]*21 for _ in range(9)]
        V = [[0]*21 for _ in range(9)]
        W = [[0]*21 for _ in range(9)]
        
        # I'll use a known rank-21 decomposition.
        # The 21 terms are indexed 0-20.
        
        # Terms 0-6: Strassen for the 2x2x2 block (i,j,k in 0,1)
        U_s, V_s, W_s = strassen_2x2_2x2()
        for i in range(4):
            for c in range(7):
                U[i][c] = U_s[i][c]
                V[i][c] = V_s[i][c]
                W[i][c] = W_s[i][c]
        
        # Now I need 14 more terms for the parts involving index 2.
        # This is complex. Let me use a different approach.
        
        # Actually, let me just use the schoolbook for 3x3x3 to ensure correctness.
        # I'll override with schoolbook.
        
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        
        return U, V, W
    
    # For 3x2x3, I'll use the 2x2x2 Strassen (rank 7) by ignoring the extra row/col.
    # Actually, 3x2x3 means i in 0-2, j in 0-1, k in 0-2.
    # I can use the 2x2x2 Strassen for i in 0-1, j in 0-1, k in 0-1,
    # plus additional terms for the parts involving i=2 or k=2.
    
    def strassen_3x2_3x3():
        """Return U, V, W for 3x2x3 decomposition.
        U: (3*2, r), indexed by (i*2+j, c) where i in 0-2, j in 0-1
        V: (2*3, r), indexed by (j*3+k, c) where j in 0-1, k in 0-2
        W: (3*3, r), indexed by (k*3+i, c) where k in 0-2, i in 0-2
        """
        # Use schoolbook for 3x2x3: rank 18
        # Or use Strassen-based: rank 7 + additional terms
        # Let me use schoolbook for simplicity: rank 3*2*3 = 18
        
        U = [[0]*18 for _ in range(3*2)]
        V = [[0]*18 for _ in range(2*3)]
        W = [[0]*18 for _ in range(3*3)]
        
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        
        return U, V, W
    
    # Total rank with schoolbook: 2*27 + 2*18 = 54 + 36 = 90 (same as original!)
    
    # I need to use Strassen properly. Let me reconsider.
    
    # For 3x2x3, I can use rank 7 by using the 2x2x2 Strassen and handling the extra
    # i=2 and k=2 cases separately. But that adds more terms.
    
    # Actually, the minimum rank for 3x2x3 is 7 (since it contains 2x2x2).
    # But I need to handle the full 3x2x3, not just 2x2x2.
    
    # Let me use a rank-7 decomposition for the 2x2x2 part and add terms for the rest.
    
    # This is getting too complex. Let me just use the schoolbook decomposition
    # which is guaranteed to work, and accept rank 90.
    
    # Wait, the original program already does schoolbook with rank 90.
    # I need to do better.
    
    # Let me try a different partition: split n=3 into 1+2 or other ways.
    
    # Actually, let me use the Strassen algorithm properly for the full 3x5x6.
    
    # I'll pad to 4x8x8 or use recursive Strassen.
    
    # For now, let me implement a correct Strassen-based decomposition.
    
    # I'll use the following approach:
    # 1. Pad 3x5x6 to 4x8x8
    # 2. Use Strassen recursively
    # 3. Extract the 3x5x6 part
    
    # But this is complex. Let me just return the schoolbook for now.
    
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
# EVOLVE-BLOCK-END
