"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = 3, 3, 5
    
    # We'll construct a rank-40 decomposition
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 9 x 40
    V = [[0] * rank for _ in range(m * p)]  # 15 x 40
    W = [[0] * rank for _ in range(p * n)]  # 15 x 40
    
    # Helper function to add a term
    def add_term(col, u_entries, v_entries, w_entries):
        for (idx, val) in u_entries:
            U[idx][col] += val
        for (idx, val) in v_entries:
            V[idx][col] += val
        for (idx, val) in w_entries:
            W[idx][col] += val
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # i in 0..2, j in 0..2, k in 0..4
    
    # Part 1: Strassen for top-left 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # Term 0: M1 = (A00+A11)(B00+B11)
    add_term(0, 
             [(0*3+0, 1), (1*3+1, 1)],
             [(0*5+0, 1), (1*5+1, 1)],
             [(0*3+0, 1), (1*3+1, 1)])
    
    # Term 1: M2 = (A10+A11)B00
    add_term(1,
             [(1*3+0, 1), (1*3+1, 1)],
             [(0*5+0, 1)],
             [(1*3+0, 1), (1*3+1, -1)])
    
    # Term 2: M3 = A00(B01-B11)
    add_term(2,
             [(0*3+0, 1)],
             [(0*5+1, 1), (1*5+1, -1)],
             [(0*3+1, 1), (1*3+1, 1)])
    
    # Term 3: M4 = A11(B10-B00)
    add_term(3,
             [(1*3+1, 1)],
             [(1*5+0, 1), (0*5+0, -1)],
             [(0*3+0, 1), (1*3+1, -1)])
    
    # Term 4: M5 = (A00+A10)B11
    add_term(4,
             [(0*3+0, 1), (1*3+0, 1)],
             [(1*5+1, 1)],
             [(0*3+0, -1), (1*3+1, 1)])
    
    # Term 5: M6 = (A11-A00)(B00+B10)
    add_term(5,
             [(1*3+1, 1), (0*3+0, -1)],
             [(0*5+0, 1), (1*5+0, 1)],
             [(1*3+1, 1)])
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    add_term(6,
             [(0*3+1, 1), (1*3+1, -1)],
             [(1*5+0, 1), (1*5+1, 1)],
             [(0*3+1, -1), (1*3+1, -1)])
    
    # Now we need to handle the remaining parts:
    # - Row 2 of A (i=2) with all columns of B
    # - Column 2 of A (j=2) with rows 0,1 of A
    # - Columns 2,3,4 of B with rows 0,1 of A
    
    # Let's add terms for the cross terms systematically
    
    # Term 7-13: Handle j=2 (third column of A) with k=0,1
    # C[i,0] += A[i,2]*B[2,0], C[i,1] += A[i,2]*B[2,1]
    # We can use 6 terms: 2 for each of i=0,1,2 with k=0,1, but we can merge
    
    # Actually, let's use a more efficient construction.
    # For the 3x3x3 part (k=0,1,2), we need 23 terms total.
    # We have 7 so far, need 16 more.
    
    # Let me use a known 23-term construction for 3x3x3.
    # I'll add terms 7-22 for the remaining 3x3x3 parts.
    
    # Terms for i=2 (third row of A), k=0,1,2
    # Term 7: A20 * B00 -> C20
    add_term(7, [(2*3+0, 1)], [(0*5+0, 1)], [(0*3+2, 1)])
    
    # Term 8: A21 * B10 -> C20
    add_term(8, [(2*3+1, 1)], [(1*5+0, 1)], [(0*3+2, 1)])
    
    # Term 9: A22 * B20 -> C20
    add_term(9, [(2*3+2, 1)], [(2*5+0, 1)], [(0*3+2, 1)])
    
    # Term 10: A20 * B01 -> C21
    add_term(10, [(2*3+0, 1)], [(0*5+1, 1)], [(1*3+2, 1)])
    
    # Term 11: A21 * B11 -> C21
    add_term(11, [(2*3+1, 1)], [(1*5+1, 1)], [(1*3+2, 1)])
    
    # Term 12: A22 * B21 -> C21
    add_term(12, [(2*3+2, 1)], [(2*5+1, 1)], [(1*3+2, 1)])
    
    # Term 13: A20 * B02 -> C22
    add_term(13, [(2*3+0, 1)], [(0*5+2, 1)], [(2*3+2, 1)])
    
    # Term 14: A21 * B12 -> C22
    add_term(14, [(2*3+1, 1)], [(1*5+2, 1)], [(2*3+2, 1)])
    
    # Term 15: A22 * B22 -> C22
    add_term(15, [(2*3+2, 1)], [(2*5+2, 1)], [(2*3+2, 1)])
    
    # Now handle j=2 (third column of A) with i=0,1 and k=0,1,2
    # These are cross terms not covered by Strassen
    # Term 16: A02 * B20 -> C00
    add_term(16, [(0*3+2, 1)], [(2*5+0, 1)], [(0*3+0, 1)])
    
    # Term 17: A12 * B20 -> C10
    add_term(17, [(1*3+2, 1)], [(2*5+0, 1)], [(0*3+1, 1)])
    
    # Term 18: A02 * B21 -> C01
    add_term(18, [(0*3+2, 1)], [(2*5+1, 1)], [(1*3+0, 1)])
    
    # Term 19: A12 * B21 -> C11
    add_term(19, [(1*3+2, 1)], [(2*5+1, 1)], [(1*3+1, 1)])
    
    # Term 20: A02 * B22 -> C02
    add_term(20, [(0*3+2, 1)], [(2*5+2, 1)], [(2*3+0, 1)])
    
    # Term 21: A12 * B22 -> C12
    add_term(21, [(1*3+2, 1)], [(2*5+2, 1)], [(2*3+1, 1)])
    
    # That's 22 terms so far. We need one more for 3x3x3.
    # Actually, we might have double-counted or missed something.
    # Let me verify the count: 7 (Strassen) + 9 (i=2) + 6 (j=2, i=0,1) = 22
    # We need 23 for 3x3x3. Let me add one more term.
    
    # Actually, I think my construction is incomplete. Let me use a different approach.
    # I'll just use the schoolbook for the remaining parts and accept rank 45.
    
    # But wait, the task says rank-minimal, and we know rank(3,3,5) <= 40.
    # Let me try a different construction.
    
    # For now, let me complete with schoolbook for k=3,4
    # Terms 22-36: Schoolbook for k=3,4 with all i,j
    col = 22
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                if col < rank:
                    U[i*3+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
    
    # We have 22 + 12 = 34 terms used, but rank is 40.
    # Let me just return this and verify it's valid.
    
    # Actually, I realize my construction might not be correct.
    # Let me use a simpler, verified approach.
    
    # Reset and use a clean construction
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use schoolbook for all 45 terms, but that's rank 45.
    # I need to find a way to reduce to 40.
    
    # Let me use the known result that rank(3,3,5) = 40 and implement it.
    # I'll use a constructive approach based on the literature.
    
    # For simplicity and correctness, I'll return rank 45 for now.
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
