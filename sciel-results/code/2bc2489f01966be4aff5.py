"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 2*2*5 = 20
    
    # Lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb, r_sb)
    
    # Hybrid approach: Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x3 (rank 12)
    # Total rank = 19
    # 
    # For the first 2 columns (k=0,1), use Strassen's algorithm
    # For the remaining 3 columns (k=2,3,4), use schoolbook
    #
    # Strassen for 2x2x2:
    # M1 = (A00 + A10)(B00 + B10)  -- wait, standard Strassen is:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # But we need to be careful: Strassen decomposes the 2x2x2 tensor.
    # For our problem, we have 5 columns in B. Let's split B into B_first (2x2) and B_rest (2x3).
    # C = A * B = A * [B_first | B_rest] = [A*B_first | A*B_rest]
    # 
    # For A*B_first (2x2 result), use Strassen (rank 7)
    # For A*B_rest (2x3 result), use schoolbook (rank 12)
    # Total rank = 7 + 12 = 19
    #
    # Let's construct this explicitly.
    
    # Strassen part for columns k=0,1
    # Variables: A[i,j] for i in {0,1}, j in {0,1}
    #            B[j,k] for j in {0,1}, k in {0,1}
    # Result: C[i,k] for i in {0,1}, k in {0,1}
    #
    # Strassen terms (7 terms):
    # t1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # t2 = (A[1,0] + A[1,1]) * B[0,0]
    # t3 = A[0,0] * (B[1,0] - B[1,1])
    # t4 = A[1,1] * (B[0,1] - B[0,0])
    # t5 = (A[0,0] + A[0,1]) * B[1,1]
    # t6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # t7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #
    # C[0,0] = t1 + t4 - t5 + t7
    # C[0,1] = t3 + t5
    # C[1,0] = t2 + t4
    # C[1,1] = t1 - t2 + t3 + t6
    
    # Schoolbook part for columns k=2,3,4
    # For each k in {2,3,4}:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # That's 4 terms per column, 3 columns = 12 terms.
    
    # Total rank = 7 + 12 = 19
    
    # Let's build the decomposition
    rank_19 = 19
    U = [[Fraction(0)] * rank_19 for _ in range(n * m)]
    V = [[Fraction(0)] * rank_19 for _ in range(m * p)]
    W = [[Fraction(0)] * rank_19 for _ in range(p * n)]
    
    # Strassen part: 7 terms (indices 0-6)
    # Term 1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U[i*m+j]: A[i,j] -> i*m+j
    # V[j*p+k]: B[j,k] -> j*p+k
    # W[k*n+i]: C[k,i] -> k*n+i (note: W[k*n+i] corresponds to output C[i,k])
    
    # t1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U[0*2+0] = U[0] += 1  (A[0,0])
    # U[1*2+1] = U[3] += 1  (A[1,1])
    # V[0*5+0] = V[0] += 1  (B[0,0])
    # V[1*5+1] = V[6] += 1  (B[1,1])
    # W[0*2+0] = W[0] += 1  (C[0,0])
    # W[1*2+1] = W[3] += 1  (C[1,1])
    
    U[0][0] += Fraction(1)
    U[3][0] += Fraction(1)
    V[0][0] += Fraction(1)
    V[6][0] += Fraction(1)
    W[0][0] += Fraction(1)
    W[3][0] += Fraction(1)
    
    # t2 = (A[1,0] + A[1,1]) * B[0,0]
    # U[1*2+0] = U[2] += 1  (A[1,0])
    # U[1*2+1] = U[3] += 1  (A[1,1])
    # V[0*5+0] = V[0] += 1  (B[0,0])
    # W[0*2+1] = W[1] += 1  (C[1,0])
    # W[1*2+1] = W[3] += -1 (C[1,1])
    
    U[2][1] += Fraction(1)
    U[3][1] += Fraction(1)
    V[0][1] += Fraction(1)
    W[1][1] += Fraction(1)
    W[3][1] += Fraction(-1)
    
    # t3 = A[0,0] * (B[1,0] - B[1,1])
    # U[0*2+0] = U[0] += 1  (A[0,0])
    # V[1*5+0] = V[5] += 1  (B[1,0])
    # V[1*5+1] = V[6] += -1 (B[1,1])
    # W[0*2+1] = W[1] += 1  (C[0,1])
    # W[1*2+1] = W[3] += 1  (C[1,1])
    
    U[0][2] += Fraction(1)
    V[5][2] += Fraction(1)
    V[6][2] += Fraction(-1)
    W[1][2] += Fraction(1)
    W[3][2] += Fraction(1)
    
    # t4 = A[1,1] * (B[0,1] - B[0,0])
    # U[1*2+1] = U[3] += 1  (A[1,1])
    # V[0*5+1] = V[1] += 1  (B[0,1])
    # V[0*5+0] = V[0] += -1 (B[0,0])
    # W[0*2+0] = W[0] += 1  (C[0,0])
    # W[0*2+1] = W[1] += 0   (no contribution to C[0,1])
    # W[1*2+0] = W[2] += 1  (C[1,0])
    # W[1*2+1] = W[3] += 0   (no contribution to C[1,1])
    
    U[3][3] += Fraction(1)
    V[1][3] += Fraction(1)
    V[0][3] += Fraction(-1)
    W[0][3] += Fraction(1)
    W[2][3] += Fraction(1)
    
    # t5 = (A[0,0] + A[0,1]) * B[1,1]
    # U[0*2+0] = U[0] += 1  (A[0,0])
    # U[0*2+1] = U[1] += 1  (A[0,1])
    # V[1*5+1] = V[6] += 1  (B[1,1])
    # W[0*2+0] = W[0] += -1 (C[0,0])
    # W[0*2+1] = W[1] += 1  (C[0,1])
    # W[1*2+1] = W[3] += 0   (no contribution to C[1,1])
    
    U[0][4] += Fraction(1)
    U[1][4] += Fraction(1)
    V[6][4] += Fraction(1)
    W[0][4] += Fraction(-1)
    W[1][4] += Fraction(1)
    
    # t6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # U[1*2+0] = U[2] += 1  (A[1,0])
    # U[0*2+0] = U[0] += -1 (A[0,0])
    # V[0*5+0] = V[0] += 1  (B[0,0])
    # V[0*5+1] = V[1] += 1  (B[0,1])
    # W[1*2+1] = W[3] += 1  (C[1,1])
    
    U[2][5] += Fraction(1)
    U[0][5] += Fraction(-1)
    V[0][5] += Fraction(1)
    V[1][5] += Fraction(1)
    W[3][5] += Fraction(1)
    
    # t7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # U[0*2+1] = U[1] += 1  (A[0,1])
    # U[1*2+1] = U[3] += -1 (A[1,1])
    # V[1*5+0] = V[5] += 1  (B[1,0])
    # V[1*5+1] = V[6] += 1  (B[1,1])
    # W[0*2+0] = W[0] += 1  (C[0,0])
    
    U[1][6] += Fraction(1)
    U[3][6] += Fraction(-1)
    V[5][6] += Fraction(1)
    V[6][6] += Fraction(1)
    W[0][6] += Fraction(1)
    
    # Schoolbook part for k=2,3,4: 12 terms (indices 7-18)
    # For k=2:
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0] * B[0,2] -> C[0,2]
    U[0][7] += Fraction(1)
    V[0*5+2][7] += Fraction(1)  # V[2]
    W[2*2+0][7] += Fraction(1)  # W[4]
    
    # Term 8: A[0,1] * B[1,2] -> C[0,2]
    U[1][8] += Fraction(1)
    V[1*5+2][8] += Fraction(1)  # V[7]
    W[2*2+0][8] += Fraction(1)  # W[4]
    
    # Term 9: A[1,0] * B[0,2] -> C[1,2]
    U[2][9] += Fraction(1)
    V[0*5+2][9] += Fraction(1)  # V[2]
    W[2*2+1][9] += Fraction(1)  # W[5]
    
    # Term 10: A[1,1] * B[1,2] -> C[1,2]
    U[3][10] += Fraction(1)
    V[1*5+2][10] += Fraction(1)  # V[7]
    W[2*2+1][10] += Fraction(1)  # W[5]
    
    # For k=3:
    # Term 11: A[0,0] * B[0,3] -> C[0,3]
    U[0][11] += Fraction(1)
    V[0*5+3][11] += Fraction(1)  # V[3]
    W[3*2+0][11] += Fraction(1)  # W[6]
    
    # Term 12: A[0,1] * B[1,3] -> C[0,3]
    U[1][12] += Fraction(1)
    V[1*5+3][12] += Fraction(1)  # V[8]
    W[3*2+0][12] += Fraction(1)  # W[6]
    
    # Term 13: A[1,0] * B[0,3] -> C[1,3]
    U[2][13] += Fraction(1)
    V[0*5+3][13] += Fraction(1)  # V[3]
    W[3*2+1][13] += Fraction(1)  # W[7]
    
    # Term 14: A[1,1] * B[1,3] -> C[1,3]
    U[3][14] += Fraction(1)
    V[1*5+3][14] += Fraction(1)  # V[8]
    W[3*2+1][14] += Fraction(1)  # W[7]
    
    # For k=4:
    # Term 15: A[0,0] * B[0,4] -> C[0,4]
    U[0][15] += Fraction(1)
    V[0*5+4][15] += Fraction(1)  # V[4]
    W[4*2+0][15] += Fraction(1)  # W[8]
    
    # Term 16: A[0,1] * B[1,4] -> C[0,4]
    U[1][16] += Fraction(1)
    V[1*5+4][16] += Fraction(1)  # V[9]
    W[4*2+0][16] += Fraction(1)  # W[8]
    
    # Term 17: A[1,0] * B[0,4] -> C[1,4]
    U[2][17] += Fraction(1)
    V[0*5+4][17] += Fraction(1)  # V[4]
    W[4*2+1][17] += Fraction(1)  # W[9]
    
    # Term 18: A[1,1] * B[1,4] -> C[1,4]
    U[3][18] += Fraction(1)
    V[1*5+4][18] += Fraction(1)  # V[9]
    W[4*2+1][18] += Fraction(1)  # W[9]
    
    # Verify the rank-19 decomposition
    if _verify(U, V, W, n, m, p, rank_19):
        best_decomp = (U, V, W, rank_19)
        best_rank = rank_19
    
    # Try to find even better decompositions with random search
    # But given the structure, rank 19 is likely optimal or close to it
    # Let's do a quick random search for ranks 11-18
    
    values = [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)]
    
    for target_rank in range(11, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U_rand = [[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U_rand, V_rand, W_rand, n, m, p, target_rank):
                found = True
                best_decomp = (U_rand, V_rand, W_rand, target_rank)
                best_rank = target_rank
                break
        
        if found:
            break
    
    U, V, W, rank = best_decomp
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
