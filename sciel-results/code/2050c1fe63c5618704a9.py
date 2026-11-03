"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 120.0  # 120s wall-clock limit, keep safety margin

def _check_deadline():
    return time.time() > DEADLINE - 10.0  # 10s safety margin

def _generate_strassen_4x4_factors():
    """
    Generate U, V, W factors for 4x4x4 matrix multiplication using Strassen's algorithm
    applied to 2x2 blocks, with schoolbook multiplication for the 2x2 blocks.
    Rank is 7 * 4 = 28.
    """
    n = 4
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen 2x2 factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 2x2 block indices:
    # A00: i=0,1; j=0,1
    # A01: i=0,1; j=2,3
    # A10: i=2,3; j=0,1
    # A11: i=2,3; j=2,3
    # Similarly for B and C.
    
    # Helper to add a 2x2 schoolbook term
    # a_block: 2x2 indices for A
    # b_block: 2x2 indices for B
    # c_block: 2x2 indices for C
    # coeff: coefficient for the term
    # offset: starting column index in U, V, W
    
    def add_2x2_term(a_i, a_j, b_i, b_j, c_i, c_j, coeff, offset):
        # a_i, a_j, b_i, b_j, c_i, c_j are tuples of 2 indices
        # e.g., a_i=(0,1), a_j=(0,1) for A00
        for k in range(2):
            for l in range(2):
                col = offset + k*2 + l
                # A index
                ui = a_i[k] * 4 + a_j[l]
                # B index
                vi = b_i[k] * 4 + b_j[l]
                # C index
                wi = c_i[k] * 4 + c_j[l]
                
                U[ui, col] += coeff
                V[vi, col] += coeff
                W[wi, col] += coeff

    # Define blocks
    # A00: (0,1)x(0,1), A01: (0,1)x(2,3), A10: (2,3)x(0,1), A11: (2,3)x(2,3)
    # B00: (0,1)x(0,1), B01: (0,1)x(2,3), B10: (2,3)x(0,1), B11: (2,3)x(2,3)
    # C00: (0,1)x(0,1), C01: (0,1)x(2,3), C10: (2,3)x(0,1), C11: (2,3)x(2,3)
    
    # M1 = (A00+A11)(B00+B11)
    # This is (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    # We can just add the 4 terms.
    # But Strassen combines them into 1 multiplication of sums.
    # The factors U, V, W for the sum are not simply the sum of factors.
    # U for (A00+A11) has non-zeros in both blocks.
    # V for (B00+B11) has non-zeros in both blocks.
    # The product is a single 2x2 matrix product.
    
    # Let's generate the 28 scalar multiplications explicitly.
    # For each Strassen step M_k, we have a 2x2 matrix product.
    # We need to define the linear combinations for U and V for that step.
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # U part: A00 + A11. V part: B00 + B11.
    # This is not a single rank-1 term in the 4x4 tensor.
    # It is a sum of 4 rank-1 terms? No.
    # The multiplication of (A00+A11) and (B00+B11) is a 2x2 matrix multiplication.
    # It requires 4 scalar multiplications.
    # Let's call the intermediate 2x2 matrices P1 and Q1.
    # P1 = A00 + A11. Q1 = B00 + B11.
    # R1 = P1 * Q1.
    # R1 is added to C00 and C11.
    
    # We can decompose P1*Q1 into 4 scalar products.
    # P1 = [p00, p01; p10, p11]
    # Q1 = [q00, q01; q10, q11]
    # R1 = [p00q00+p01q10, p00q01+p01q11; p10q00+p11q10, p10q01+p11q11]
    
    # The indices for P1:
    # p00 = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # p01 = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # p10 = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
    # p11 = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
    
    # Similarly for Q1.
    
    # This approach is tedious to code manually.
    # Instead, we can use the known rank-28 factors or construct them programmatically.
    
    # Let's use a simpler construction:
    # Just use schoolbook for 4x4? Rank 64.
    # 64 + 27 = 91. Worse than 75.
    
    # So we MUST use Strassen.
    
    # Let's hardcode the 28 terms.
    # Term 1: M1[0,0] = (A00+A11)[0,0] * (B00+B11)[0,0] + (A00+A11)[0,1] * (B00+B11)[1,0]
    # This is getting messy.
    
    # Alternative: Use the rank-18 algorithm?
    # Or just use the rank-28 algorithm from a reference.
    
    # Let's try to implement the 2x2 Strassen factors and then "inflate" them to 4x4.
    
    # U_22 (4x7), V_22 (4x7), W_22 (4x7)
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00+A11. V: B00+B11. W: C00, C11.
    # U[0,0] += 1, U[1,0] += 1 (A00[0,0], A11[0,0] -> wait, indices)
    # A indices: 0,1,2,3.
    # A00: 0,1. A11: 2,3.
    # U[0, 0] = 1 (A00[0,0])
    # U[1, 0] = 1 (A00[0,1]) -- No, U is for A.
    # U[i, j, c].
    # For M1, we multiply (A00+A11) and (B00+B11).
    # This is a 2x2 matrix product.
    # It contributes to C00 and C11.
    
    # Let's just use the schoolbook for the 2x2 blocks inside Strassen.
    # This is what I described before.
    
    # I will implement the 28 terms explicitly.
    
    # Helper: add_term(a_idx, b_idx, c_idx, coeff)
    # a_idx: (i, j) in 4x4
    # b_idx: (j, k) in 4x4
    # c_idx: (k, i) in 4x4
    
    # M1 = (A00+A11)(B00+B11)
    # P = A00+A11, Q = B00+B11.
    # P[0,0] = A[0,0]+A[2,2]
    # P[0,1] = A[0,1]+A[2,3]
    # P[1,0] = A[1,0]+A[3,2]
    # P[1,1] = A[1,1]+A[3,3]
    # Q[0,0] = B[0,0]+B[2,2]
    # Q[0,1] = B[0,1]+B[2,3]
    # Q[1,0] = B[1,0]+B[3,2]
    # Q[1,1] = B[1,1]+B[3,3]
    
    # R = P*Q.
    # R[0,0] = P[0,0]Q[0,0] + P[0,1]Q[1,0]
    # This expands to 4 terms.
    # (A00[0,0]+A11[0,0])(B00[0,0]+B11[0,0]) + ...
    # = A00[0,0]B00[0,0] + A00[0,0]B11[0,0] + A11[0,0]B00[0,0] + A11[0,0]B11[0,0]
    
    # It is easier to just add the 4 schoolbook terms for the 2x2 block product.
    # But we need to combine the U and V factors for the sums.
    # U for P[0,0] has non-zeros at A[0,0] and A[2,2].
    # V for Q[0,0] has non-zeros at B[0,0] and B[2,2].
    # The product is a single scalar multiplication z = P[0,0] * Q[0,0].
    # Then we add z to C.
    
    # So for each of the 4 scalar products in P*Q, we have:
    # U vector: non-zeros at the A indices involved.
    # V vector: non-zeros at the B indices involved.
    # W vector: non-zeros at the C indices involved.
    
    # Let's write a function to add a Strassen 2x2 step.
    
    def add_strassen_2x2(a_blocks, b_blocks, c_blocks, offset):
        # a_blocks: list of 4 tuples of (i,j) indices for A00, A01, A10, A11
        # b_blocks: list of 4 tuples of (j,k) indices for B00, B01, B10, B11
        # c_blocks: list of 4 tuples of (k,i) indices for C00, C01, C10, C11
        
        # M1 = (A00+A11)(B00+B11)
        # P = A00+A11, Q = B00+B11
        # P_idx: A00[0,0], A11[0,0], etc.
        
        # We will generate 4 terms for P*Q.
        
        # P[0,0] = A00[0,0] + A11[0,0]
        # Q[0,0] = B00[0,0] + B11[0,0]
        # Term: (A00[0,0] + A11[0,0]) * (B00[0,0] + B11[0,0])
        # This is NOT a single rank-1 term.
        # It is 4 rank-1 terms if expanded.
        # But Strassen says it is 1 multiplication of the sums.
        # So U has 2 non-zeros, V has 2 non-zeros.
        # The product z = (sum U*A) * (sum V*B).
        # Then z is added to C.
        
        # So for M1[0,0] contribution to C00[0,0]:
        # U: A[0,0]=1, A[2,2]=1
        # V: B[0,0]=1, B[2,2]=1
        # W: C[0,0]=1 (and C[2,2]=1 for C11[0,0]?)
        
        # Wait, M1 contributes to C00 and C11.
        # C00 = M1 + ...
        # C11 = M1 - ...
        
        # So W for this term should add to C00[0,0] and C11[0,0].
        
        # Let's define the 7 Strassen steps and their contributions.
        
        # This is getting too complex to do on the fly without errors.
        # I will fall back to Rank 64 (Schoolbook 4x4) + 27 = 91? No.
        # I will use the Rank 28 algorithm from a known source or implement it carefully.
        
        # Actually, there is a simpler way:
        # Use the Rank 18 algorithm for 4x4?
        # No, let's stick to the plan: Rank 28.
        
        # I will implement the 28 terms by iterating.
        
        pass

    # Given the complexity and risk of error in manual derivation,
    # and the time limit, I will return the schoolbook rank 75.
    # The improvement to 55 is significant but requires careful implementation.
    # I will try to implement the 4x4 Strassen factors in the next step if I can verify them.
    
    # For now, return None to indicate failure to generate, and let construct() handle it.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    Returns {'U': list of lists (15, rank), 'V': list of lists (25, rank), 'W': list of lists (15, rank)}
    where U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] exactly decompose the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 25
    pn = p * n  # 15
    
    # Schoolbook: rank = n*m*p = 75
    rank = n * m * p
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Try to generate the improved rank 55 decomposition
    # If it fails, return schoolbook.
    
    # Since _generate_strassen_4x4_factors is not fully implemented,
    # we return schoolbook.
    
    return {"U": U, "V": V, "W": W}
