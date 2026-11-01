"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a tensor decomposition for matrix multiplication 3x3 * 3x8.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 72).
    2. Use the Laderman decomposition for the 3x3 sub-problems to reduce rank.
    
    The 3x3 * 3x8 multiplication can be viewed as multiplying a 3x3 matrix by
    three different 3x3 sub-matrices of the 3x8 matrix (columns 0-2, 3-5, 6-7+pad).
    Actually, it's better to view it as:
    A (3x3) * B (3x8) = C (3x8).
    
    We can split B into B1 (3x4) and B2 (3x4).
    C = [A*B1 | A*B2].
    
    For each A*B1 (3x3 * 3x4), we can further split B1 into columns.
    Actually, let's just use the schoolbook for the parts that don't fit Laderman
    and see if we can reduce.
    
    However, a more direct approach:
    The tensor rank of T_{n,m,p} is at most n * m * p.
    For 3x3 * 3x3, Laderman gives rank 20.
    For 3x3 * 3x8, we can consider it as 8 independent 3x3 * 3x1 multiplications?
    No, that's not how it works.
    
    Let's try a hybrid approach:
    Use Laderman for the first 3 columns of B (3x3 * 3x3 -> 3x3) with rank 20.
    Use schoolbook for the remaining 5 columns of B (3x3 * 3x5 -> 3x5) with rank 15.
    Total rank = 20 + 15 = 35.
    
    Wait, is this valid?
    C1 = A * B[:, 0:3] (3x3)
    C2 = A * B[:, 3:8] (3x5)
    
    Yes, because matrix multiplication is linear in the columns of B.
    So C = [C1 | C2].
    
    We can decompose the tensor for C1 using Laderman (rank 20).
    We can decompose the tensor for C2 using schoolbook (rank 15).
    Since the inputs A are the same, we can just sum the contributions.
    The output tensors are disjoint (C1 vs C2), so the total rank is 20 + 15 = 35.
    
    Let's verify this.
    T_total = T_C1 + T_C2 (with appropriate zero padding).
    Rank(T_total) <= Rank(T_C1) + Rank(T_C2) = 20 + 15 = 35.
    
    This is much better than 72.
    
    Now, can we do better for C2 (3x3 * 3x5)?
    We can split C2 into C2a (3x3 * 3x3) and C2b (3x3 * 3x2).
    C2a: use Laderman (rank 20).
    C2b: use schoolbook (rank 6).
    Total for C2: 20 + 6 = 26.
    Total for C: 20 (for C1) + 26 (for C2) = 46.
    
    Wait, that's worse than 35. Why?
    Because for C1, I used Laderman (20) instead of schoolbook (9).
    Schoolbook for 3x3 is 9. Laderman is 20.
    So using Laderman for a 3x3 block INCREASES the rank compared to schoolbook!
    
    My mistake. Laderman is for 3x3*3x3 with rank 20, but schoolbook is 9.
    So schoolbook is better for small matrices?
    No, Strassen for 2x2 is 7 vs schoolbook 8.
    Laderman for 3x3 is 20 vs schoolbook 9? That can't be right.
    Laderman's algorithm has rank 20, but schoolbook is 9.
    Is Laderman actually rank 20?
    Yes, Laderman's algorithm for 3x3 matrix multiplication has rank 20.
    But 20 > 9. So why is it famous?
    Because it's a recursive algorithm? No, it's a direct algorithm.
    Wait, 3x3 schoolbook is 9 multiplications.
    Laderman is 20 multiplications.
    So Laderman is WORSE than schoolbook for 3x3?
    
    Let me check.
    Strassen 2x2: 7 mults. Schoolbook 8 mults. Strassen is better.
    Laderman 3x3: 20 mults. Schoolbook 9 mults.
    This implies Laderman is not an improvement over schoolbook for 3x3?
    
    Actually, I think I am confusing "rank" with "multiplications" in a specific context.
    Tensor rank is the number of terms in the decomposition.
    For 3x3*3x3, the tensor rank is at most 9 (schoolbook).
    Is it less than 9?
    It is known that the rank of 3x3 matrix multiplication is at most 20 (Laderman).
    But if schoolbook gives 9, then the rank is at most 9.
    Is it possible that the rank is less than 9?
    No, the rank of 3x3*3x3 is exactly 9?
    Actually, it is an open problem whether the rank of 3x3*3x3 is less than 9.
    But we know it is at most 9.
    
    So my assumption that Laderman gives a rank of 20 is correct in the sense that it's a valid decomposition of rank 20.
    But since schoolbook gives 9, we should use schoolbook for 3x3 blocks.
    
    So, going back:
    For 3x3 * 3x8:
    Split into 3x3 * 3x3 and 3x3 * 3x5.
    Use schoolbook for both.
    Rank = 9 + 15 = 24.
    
    Can we do better?
    The lower bound for 3x3 * 3x8 is at least max(9, 24, 24) = 24.
    So 24 is the lower bound!
    
    If we can achieve 24, that's optimal.
    Schoolbook for 3x3*3x3 (rank 9) + schoolbook for 3x3*3x5 (rank 15) = 24.
    
    So the minimal rank is 24.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # We split the 8 columns of B into 3 and 5.
    # C = [A*B1 | A*B2] where B1 is 3x3, B2 is 3x5.
    # C1 = A*B1 (3x3), C2 = A*B2 (3x5).
    
    # Decomposition for C1 (3x3 * 3x3 -> 3x3):
    # Schoolbook: 9 terms.
    # U1, V1, W1 of shape (9, 9), (9, 9), (9, 9)
    
    # Decomposition for C2 (3x3 * 3x5 -> 3x5):
    # Schoolbook: 15 terms.
    # U2, V2, W2 of shape (9, 15), (15, 15), (15, 9)
    
    # Total rank = 9 + 15 = 24.
    
    # Let's construct the matrices.
    # The total U, V, W will have 24 columns.
    # Columns 0-8 are for C1.
    # Columns 9-23 are for C2.
    
    rank = 24
    U = np.zeros((n * m, rank))  # (9, 24)
    V = np.zeros((m * p, rank))  # (24, 24)
    W = np.zeros((p * n, rank))  # (24, 24)
    
    # Part 1: C1 = A * B1 (columns 0-2 of B)
    # B1 corresponds to columns 0, 1, 2 of B (3x3 matrix).
    # C1 corresponds to columns 0, 1, 2 of C (3x3 matrix).
    
    # Schoolbook for 3x3 * 3x3:
    # For i in 0..2, j in 0..2, k in 0..2:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1  (Note: B1 uses columns 0-2, so j*p+k is correct)
    # W[k*n + i, col] = 1  (Note: C1 uses columns 0-2, so k*n+i is correct)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # Only first 3 columns of B
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: C2 = A * B2 (columns 3-7 of B)
    # B2 corresponds to columns 3, 4, 5, 6, 7 of B (3x5 matrix).
    # C2 corresponds to columns 3, 4, 5, 6, 7 of C (3x5 matrix).
    
    # Schoolbook for 3x3 * 3x5:
    # For i in 0..2, j in 0..2, k in 3..7:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(3, p):  # Columns 3 to 7 of B
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

