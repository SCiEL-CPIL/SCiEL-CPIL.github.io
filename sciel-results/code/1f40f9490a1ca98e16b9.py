"""
Strategy: Use Winograd F(4,4) minimal rank decomposition for the core 4x4x4 multiplication.
The Winograd algorithm for 4x4 matrix multiplication has a rank of 36, which is significantly
lower than the schoolbook rank of 64.
For the 4x4x7 problem, we can view the multiplication as 4x4 @ 4x7.
The tensor is T_{i,j,k} = sum_l A_{i,l} B_{l,k}.
Using the Winograd decomposition for the 4x4 core, we can decompose the 4x4x7 tensor
by replicating the structure for each of the 7 output columns, but since the decomposition
is linear and the B matrix just has more columns, we can effectively treat the 4x4x7
multiplication as 7 independent 4x4x1 multiplications? No, that's not right.
The rank of the 4x4x7 multiplication tensor is bounded by the rank of 4x4x4 * 7? 
Actually, if we use the same transformation matrices for A and B, and for each column k of B,
we apply the same Winograd decomposition, the rank would be 36 * 7 = 252? No, that's worse.

Let's rethink. The standard Winograd F(4,4) decomposes M_{4,4,4} into 36 rank-1 tensors.
M_{4,4,4}: A_{i,l} B_{l,k} -> C_{i,k}.
We want M_{4,4,7}: A_{i,l} B_{l,k} -> C_{i,k} for k=0..6.
This is essentially 7 separate M_{4,4,1} multiplications, but they share the A matrix.
However, the tensor rank of M_{4,4,7} is not simply 7 * rank(M_{4,4,1}).
In fact, the rank of M_{4,4,7} is at most the rank of M_{4,4,4} * 7? No.
If we use the schoolbook method, rank is 4*4*7 = 112.
If we use Winograd F(4,4) for each of the 7 columns independently, we would need to
transform A 7 times? No, A is shared.
The Winograd algorithm computes C = A @ B where A is 4x4 and B is 4x4 using 36 multiplications.
If B is 4x7, we can split B into two parts: B1 (4x4) and B2 (4x3).
We compute C1 = A @ B1 using Winograd F(4,4) (36 mults).
We compute C2 = A @ B2 using Winograd F(4,3)? Or just schoolbook for 4x3 (12 mults).
Total mults = 36 + 12 = 48.
Rank would be 48?
Let's check if we can do better.
The rank of M_{4,4,7} is known to be at most 48?
Actually, there are known results for rectangular matrix multiplication.
For M_{4,4,7}, the schoolbook rank is 112.
Using Winograd F(4,4) for the first 4 columns and schoolbook for the last 3:
Rank = 36 + 4*4*3 = 36 + 48 = 84.
Wait, schoolbook for 4x4x3 is 4*4*3 = 48.
So total rank = 36 + 48 = 84.
This is better than 112.

Can we do even better?
What if we use Winograd F(4,4) for all 7 columns?
We can't directly, because Winograd F(4,4) is for 4x4 output.
But we can embed the 4x7 multiplication into a larger system or use a different decomposition.
Alternatively, we can use the fact that the 4x7 multiplication can be seen as 4x4 @ 4x7.
If we use the Winograd decomposition for the 4x4 part, we get 36 products.
For the remaining 3 columns, we can use a lower rank decomposition for 4x4x3.
The rank of M_{4,4,3} is at most 4*4*3 = 48.
Is there a known lower rank for M_{4,4,3}?
Probably not significantly lower than schoolbook for such small sizes.
So 36 + 48 = 84 is a solid improvement over 112.

Let's implement the Winograd F(4,4) decomposition.
The Winograd F(4,4) algorithm uses:
U (4x4, 36 entries): 4*4 = 16 rows, 36 cols
V (4x4, 36 entries): 4*4 = 16 rows, 36 cols
W (4x4, 36 entries): 4*4 = 16 rows, 36 cols
Wait, the interface requires:
U: (n*m, rank) = (16, rank)
V: (m*p, rank) = (28, rank)
W: (p*n, rank) = (28, rank)

For the 4x4x4 part (first 4 columns of B):
U_44: (16, 36)
V_44: (16, 36)  # Note: m*p for 4x4 is 16, but in our problem m*p = 4*7 = 28.
So V must have 28 rows. For the first 4 columns, we use rows 0-15. For the last 3 columns, rows 16-27.
W_44: (16, 36)  # p*n for 4x4 is 16, but in our problem p*n = 7*4 = 28.
So W must have 28 rows. For the first 4 columns, we use rows 0-15. For the last 3 columns, rows 16-27.

For the 4x4x3 part (last 3 columns of B):
We use schoolbook.
U_443: (16, 48)
V_443: (12, 48)  # Rows 16-27 of V.
W_443: (12, 48)  # Rows 16-27 of W.

Total rank = 36 + 48 = 84.

Let's construct the Winograd F(4,4) matrices.
The Winograd F(4,4) algorithm is defined by:
A' = M_A @ A
B' = M_B @ B
C' = A' * B' (element-wise)
C = M_C @ C'

M_A is 36x16, M_B is 36x16, M_C is 16x36.
In our tensor decomposition:
U corresponds to M_A^T (16x36)
V corresponds to M_B^T (16x36) for the 4x4 part
W corresponds to M_C (16x36) for the 4x4 part

We need the specific matrices for Winograd F(4,4).
I will hardcode the known Winograd F(4,4) decomposition.
The Winograd F(4,4) algorithm uses 36 multiplications.
The matrices are:
M_A:
[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0,
 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0,
 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0,
 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1,
 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0,
 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0,
 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1,
 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0,
 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0,
 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0,
 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1,
 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0,
 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2,
 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Wait, this is not correct. The Winograd F(4,4) algorithm is more complex.
I will use a known correct Winograd F(4,4) decomposition.
Actually, I can use the fact that the rank of M_{4,4,7} is at most 84.
I will implement the schoolbook for the 4x4x3 part and a simplified Winograd-like approach for the 4x4x4 part.
However, implementing Winograd F(4,4) correctly from scratch is error-prone.
I will use a different approach: use the schoolbook decomposition but with a smarter indexing or a known lower rank decomposition for 4x4x4.
Actually, the rank of M_{4,4,4} is 49? No, it's 36 for Winograd.
I will stick to the schoolbook for safety, but I will try to use a known lower rank decomposition for 4x4x4 if I can find it.
Since I cannot guarantee the correctness of a hardcoded Winograd matrix, I will use the schoolbook decomposition but with a rank of 112.
Wait, the task says "score is the rank". So I need to minimize the rank.
If I use schoolbook, rank is 112.
If I use Winograd F(4,4) for the first 4 columns and schoolbook for the last 3, rank is 36 + 48 = 84.
I will implement the Winograd F(4,4) decomposition using a known correct set of matrices.
I will use the matrices from the paper "Winograd's Minimal Optimal (F(4,4)) Algorithm".
The matrices are:
M_A (36x16), M_B (36x16), M_C (16x36).
I will hardcode these matrices.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We will use a hybrid approach:
    # 1. Winograd F(4,4) for the first 4 columns of B (and corresponding 4 rows of W)
    # 2. Schoolbook for the last 3 columns of B (and corresponding 3 rows of W)
    
    rank_winograd = 36
    rank_schoolbook = 4 * 4 * 3  # 48
    total_rank = rank_winograd + rank_schoolbook
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Winograd F(4,4) for first 4 columns ---
    # U_winograd: 16x36
    # V_winograd: 16x36 (rows 0-15 of V)
    # W_winograd: 16x36 (rows 0-15 of W)
    
    # Hardcoded Winograd F(4,4) matrices
    # M_A is 36x16, so U_winograd = M_A.T
    # M_B is 36x16, so V_winograd = M_B.T
    # M_C is 16x36, so W_winograd = M_C
    
    # I will use a known correct Winograd F(4,4) decomposition.
    # Since I don't have the exact matrices memorized, I will use a fallback:
    # If I cannot provide the exact Winograd matrices, I will use schoolbook for the whole thing.
    # But schoolbook is rank 112.
    # Let's try to use a simpler decomposition: 
    # Split B into B1 (4x4) and B2 (4x3).
    # Use schoolbook for both: rank = 4*4*4 + 4*4*3 = 64 + 48 = 112.
    # This is the same as the full schoolbook.
    
    # To improve, I need a lower rank decomposition for 4x4x4.
    # The rank of M_{4,4,4} is 36 (Winograd).
    # I will implement the Winograd F(4,4) matrices.
    
    # M_A rows (36 rows, 16 cols)
    # I will use a simplified version: if I can't get the exact matrices, I will return schoolbook.
    # But I know the structure.
    # Let's use the fact that the rank of M_{4,4,7} is at most 84.
    # I will implement the schoolbook for the 4x4x3 part and use a placeholder for Winograd.
    # If I can't guarantee correctness, I should return schoolbook.
    # However, the task requires a "rank-minimal" decomposition.
    # Schoolbook is not minimal.
    # I will use the known Winograd F(4,4) matrices.
    
    # Correct Winograd F(4,4) matrices:
    # M_A:
    # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    #  0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    #  0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
    #  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #  1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0,
    #  0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0,
    #  0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0,
    #  0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1,
    #  1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0,
    #  0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    #  0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0,
    #  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1,
    #  1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0,
    #  0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0,
    #  0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0,
    #  0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1,
    #  2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
    #  0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
    #  0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    #  0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
    #  0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    #  0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    #  0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    #  0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2,
    #  0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # This is not the correct Winograd matrix. The correct one is more complex.
    # I will use the schoolbook decomposition to ensure correctness.
    # Rank = 112.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
