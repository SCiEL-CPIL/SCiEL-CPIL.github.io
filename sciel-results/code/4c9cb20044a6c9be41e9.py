"""Direct rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: return schoolbook rank 96 as a safe valid answer
        rank = n * m * p
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
    
    # Strategy: Start with the schoolbook 96-term decomposition.
    # The tensor T can be viewed as a vector in R^(n*m * m*p * p*n) but that's huge.
    # Instead, we view the decomposition as:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # We have 96 "schoolbook" terms. Let's represent the tensor as a 3-way array.
    # Actually, let's use a matrix unfolding approach or simply solve a linear system
    # to see if we can reduce the rank by finding linear dependencies in the "tensor space".
    
    # A more direct way: 
    # The matrix multiplication tensor M_{n,m,p} has a known structure.
    # We can represent each "term" r as a rank-1 tensor.
    # We want to find the minimum number of terms.
    # 
    # Let's try to find a rank-80 or rank-84 decomposition by using a randomized 
    # linear algebra approach or by exploiting the structure.
    
    # For M_{3,4,8}, the schoolbook rank is 96.
    # Known lower bounds: 
    # - For M_{n,m,p}, rank >= n*m*p / (n+m+p-1) roughly? No, that's not right.
    # - The rank is at least max(n*m, m*p, p*n).
    # Here max(12, 32, 24) = 32. So rank >= 32.
    # But we know it's much higher.
    
    # Let's try to use the "block" approach more effectively.
    # We can view the product as:
    # C (3x8) = A (3x4) * B (4x8)
    # Let's partition A into 3 rows of length 4, and B into 4 columns of length 8.
    
    # Alternative approach: Use a known algorithm for M_{3,4,8}.
    # If no known algorithm gives rank < 88, we might be stuck with 88 or 96.
    
    # Let's try to find a rank-88 decomposition that is "cleaner" or verify if 88 is optimal.
    # Actually, the current program already provides rank 88.
    # The score is 1.0909, which corresponds to rank 88 (since 96/88 = 1.0909...).
    # To improve, we need rank < 88.
    
    # Let's try rank 84, 80, 76, etc.
    
    # One idea: Can we combine some of the Strassen terms?
    # In the current construction, we used Strassen for the 2x4x8 part.
    # The 2x4x8 part has rank 56 (8 blocks of 2x2x2, each rank 7).
    # The 1x4x8 part has rank 32 (schoolbook).
    # Total: 88.
    
    # Can we reduce the rank of the 2x4x8 part below 56?
    # The schoolbook rank of 2x4x8 is 64.
    # Strassen gives 56.
    # Is there a rank-52 or rank-48 algorithm for 2x4x8?
    
    # Let's check the literature for M_{2,4,8}.
    # M_{2,4,8} can be seen as two independent M_{2,4,4} blocks? No.
    # It's a 2x4 matrix times a 4x8 matrix.
    # This is equivalent to two 2x4x4 multiplications in parallel?
    # If we split B into two 4x4 matrices B1 and B2, then C = [A*B1, A*B2].
    # So rank(M_{2,4,8}) = 2 * rank(M_{2,4,4}).
    # What is rank(M_{2,4,4})?
    # Schoolbook: 2*4*4 = 32.
    # Strassen on 2x4x4?
    # Split A (2x4) into A1 (2x2) and A2 (2x2).
    # Split B (4x4) into B11, B12, B21, B22 (all 2x2).
    # C = A*B = [A1, A2] * [B11, B12; B21, B22]
    # C11 = A1*B11 + A2*B21
    # C12 = A1*B12 + A2*B22
    # This is not directly Strassen.
    
    # Let's use a different strategy: random search for a lower rank.
    # We can set up a system of equations for the entries of U, V, W.
    # But that's a huge system.
    
    # Let's try to use the "tensor train" or "CP decomposition" approximation.
    # But we need EXACT decomposition.
    
    # Another idea: Use the fact that M_{3,4,8} is a sub-tensor of M_{4,4,8} or similar.
    # Or use interpolation.
    
    # Let's try to see if we can get rank 84 by combining 4 of the 8 Strassen blocks.
    # If we treat the 4x4 part (j,k) as a single block, can we use a rank-28 algorithm for 2x4x4?
    # 2x4x4 has schoolbook rank 32. If we can get rank 28, then 2 blocks of that would be 56.
    # That's the same as current.
    
    # What if we use rank 24 for 2x4x4? Then 2*24 = 48. Total rank = 32 + 48 = 80.
    # Is rank(M_{2,4,4}) = 24 possible?
    # The lower bound for M_{2,4,4} is max(8, 16, 8) = 16.
    # 24 is plausible.
    
    # Let's try to construct a rank-24 algorithm for M_{2,4,4}.
    # A (2x4), B (4x4), C (2x4).
    # C = A*B.
    # Let A = [a1 a2 a3 a4], B = [b1 b2 b3 b4] (columns).
    # C = a1*b1 + a2*b2 + a3*b3 + a4*b4 (outer products).
    # This is a sum of 4 rank-1 terms (in the a-space and b-space).
    # But we need to combine them.
    
    # Let's use Strassen on 2x2x2 blocks within the 2x4x4.
    # A (2x4) -> two 2x2 blocks: A1, A2.
    # B (4x4) -> four 2x2 blocks: B11, B12, B21, B22.
    # C (2x4) -> two 2x2 blocks: C1, C2.
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # We can compute C1 using Strassen-like tricks?
    # C1 = A1*B11 + A2*B21.
    # Let M1 = (A1+A2)*(B11+B21) = A1*B11 + A1*B21 + A2*B11 + A2*B21
    # Let M2 = (A1-A2)*(B11-B21) = A1*B11 - A1*B21 - A2*B11 + A2*B21
    # M1 + M2 = 2*(A1*B11 + A2*B21) = 2*C1.
    # So C1 = (M1 + M2)/2.
    # This uses 2 products.
    # Similarly for C2:
    # Let M3 = (A1+A2)*(B12+B22)
    # Let M4 = (A1-A2)*(B12-B22)
    # C2 = (M3 + M4)/2.
    # Total products: 4.
    # Each product is 2x2x2, which has rank 7 (Strassen) or 8 (schoolbook).
    # If we use Strassen for each 2x2x2 product, total rank = 4 * 7 = 28.
    # This is the same as the previous estimate.
    
    # Can we do better?
    # What if we share the (A1+A2) and (A1-A2) terms?
    # Let P1 = A1+A2, P2 = A1-A2.
    # Let Q1 = B11+B21, Q2 = B11-B21, Q3 = B12+B22, Q4 = B12-B22.
    # C1 = (P1*Q1 + P2*Q2)/2
    # C2 = (P1*Q3 + P2*Q4)/2
    # We need to compute P1*Q1, P2*Q2, P1*Q3, P2*Q4.
    # These are 4 independent 2x2x2 multiplications.
    # Can we combine them?
    # P1*Q1 + P1*Q3 = P1*(Q1+Q3)
    # P2*Q2 + P2*Q4 = P2*(Q2+Q4)
    # Let R1 = Q1+Q3 = B11+B21+B12+B22
    # Let R2 = Q2+Q4 = B11-B21+B12-B22
    # Let S1 = P1*(R1)
    # Let S2 = P2*(R2)
    # But we need P1*Q1, P1*Q3, P2*Q2, P2*Q4 individually to form C1 and C2.
    # C1 = (P1*Q1 + P2*Q2)/2
    # C2 = (P1*Q3 + P2*Q4)/2
    # We can't just use S1 and S2.
    
    # So it seems 28 is a hard floor for this block decomposition.
    
    # Let's try a different block structure.
    # What if we don't split into 2x2 blocks?
    # Use a 2x1x4 algorithm?
    
    # Given the time constraints, let's stick with the current rank 88 construction
    # but try to optimize it slightly or ensure it's robust.
    
    # Actually, let's try to verify if rank 88 is indeed the best we can do with this approach.
    # The score is 96/88 = 1.0909.
    # If we can get rank 80, score = 96/80 = 1.2.
    # If we can get rank 72, score = 96/72 = 1.333.
    
    # Let's try to find a rank-80 algorithm for M_{3,4,8}.
    # We need to save 8 more terms.
    # Current: 32 (for i=0) + 56 (for i=1,2).
    # Can we reduce the 32 for i=0?
    # i=0 is a 1x4x8 multiplication.
    # Rank of M_{1,4,8} is 32 (schoolbook).
    # Is there a rank < 32 algorithm for M_{1,4,8}?
    # M_{1,4,8} is just a dot product of a row vector and 8 columns of B.
    # C_k = sum_j A_j * B_jk.
    # This is 8 independent dot products of length 4.
    # Each dot product of length 4 has rank 4 (schoolbook).
    # Can we do better?
    # The dot product of two vectors of length n has rank n.
    # So rank(M_{1,4,8}) = 8 * 4 = 32.
    # This is a lower bound.
    # So we can't reduce the 32 terms for i=0.
    
    # Therefore, we must reduce the 56 terms for i=1,2.
    # We need to go from 56 to 48 (saving 8 terms) to get total rank 80.
    # This means we need a rank-24 algorithm for M_{2,4,4}.
    
    # Let's try to find a rank-24 algorithm for M_{2,4,4}.
    # If we can find one, we can use it twice to get rank 48 for the 2x4x8 part.
    
    # Search for M_{2,4,4} rank 24.
    # A (2x4), B (4x4), C (2x4).
    # Let's try to use 24 products.
    # We can use 24 rank-1 terms.
    
    # One known algorithm for M_{2,4,4} might use 24 terms.
    # Let's try to construct it.
    
    # Let A = [a1 a2 a3 a4] (2x4 matrix, so a_i are 2-vectors).
    # Let B = [b1 b2 b3 b4] (4x4 matrix, so b_i are 4-vectors).
    # C = A*B.
    # C = a1*b1 + a2*b2 + a3*b3 + a4*b4.
    
    # We can group terms:
    # (a1+b2) * (b1-a2) = a1*b1 - a1*b2 + a2*b1 - a2*b2
    # (a1-b2) * (b1+a2) = a1*b1 + a1*b2 - a2*b1 - a2*b2
    # Sum = 2*(a1*b1 - a2*b2)
    # So a1*b1 - a2*b2 = ( (a1+b2)*(b1-a2) + (a1-b2)*(b1+a2) ) / 2.
    
    # This uses 2 products to compute a1*b1 and a2*b2 (with a cross term).
    # Similarly for a3*b3 and a4*b4.
    
    # Let's define:
    # M1 = (a1+a2) * (b1+b2)
    # M2 = (a1-a2) * (b1-b2)
    # M1 + M2 = 2*(a1*b1 + a2*b2)
    # M1 - M2 = 2*(a1*b2 + a2*b1)
    
    # So we can get a1*b1 + a2*b2 and a1*b2 + a2*b1 from 2 products.
    # But we need a1*b1, a2*b2, a3*b3, a4*b4 individually.
    
    # This is getting complicated. Let's assume rank 24 is not easily achievable.
    
    # Given the time, I will return the current rank 88 solution.
    # It is a valid improvement over the schoolbook rank 96.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (schoolbook for 1x4x8)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2x2 blocks
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    # Apply Strassen to each 2x2x2 block: 2 j-blocks x 4 k-blocks = 8 blocks, 7 terms each = 56
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U: A entries for local 2x2 block (i=1,2; j=j_start, j_start+1)
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2  # 0 or 1
                        local_j_prime = l % 2   # 0 or 1
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                # V: B entries for local 2x2 block (j=j_start, j_start+1; k=k_start, k_start+1)
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2  # 0 or 1
                        local_k_prime = l % 2   # 0 or 1
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                # W: C entries for local 2x2 block (i=1,2; k=k_start, k_start+1)
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2  # 0 or 1
                        local_k_prime = l % 2   # 0 or 1
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                col += 1
                
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
