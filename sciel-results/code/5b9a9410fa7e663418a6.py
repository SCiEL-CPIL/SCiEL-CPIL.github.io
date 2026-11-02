import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will construct the rank-77 decomposition.
    # Part 1: 4x4x4 (cols 0-3 of B) -> Rank 49
    # Part 2: 4x4x2 (cols 4-5 of B) -> Rank 28
    
    total_rank = 77
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # Helper to get the 2x2x2 Strassen terms
    # Returns U, V, W matrices of shape (4, 7)
    def get_strassen_2x2x2():
        U_22 = np.zeros((4, 7), dtype=np.float64)
        V_22 = np.zeros((4, 7), dtype=np.float64)
        W_22 = np.zeros((4, 7), dtype=np.float64)
        
        # Indices: 00, 01, 10, 11
        # Term 1
        U_22[0, 0] = 1; V_22[0, 0] = 1; V_22[3, 0] = 1
        W_22[0, 0] = 1; W_22[3, 0] = 1
        # Term 2
        U_22[2, 1] = 1; V_22[0, 1] = 1; V_22[1, 1] = 1
        W_22[2, 1] = 1; W_22[3, 1] = -1
        # Term 3
        U_22[0, 2] = 1; U_22[1, 2] = 1; V_22[3, 2] = 1
        W_22[1, 2] = 1; W_22[3, 2] = 1
        # Term 4
        U_22[2, 3] = 1; U_22[3, 3] = 1; V_22[0, 3] = 1
        W_22[0, 3] = 1; W_22[2, 3] = 1
        # Term 5
        U_22[0, 4] = 1; U_22[3, 4] = 1; V_22[0, 4] = 1; V_22[3, 4] = 1
        W_22[0, 4] = -1; W_22[3, 4] = -1
        # Term 6
        U_22[1, 5] = 1; U_22[3, 5] = -1; V_22[2, 5] = 1; V_22[3, 5] = 1
        W_22[2, 5] = 1; W_22[3, 5] = 1
        # Term 7
        U_22[0, 6] = 1; U_22[2, 6] = -1; V_22[0, 6] = 1; V_22[1, 6] = 1
        W_22[0, 6] = 1
        
        return U_22, V_22, W_22

    U_22, V_22, W_22 = get_strassen_2x2x2()
    
    # --- Part 1: 4x4x4 ---
    # This is a recursive Strassen step.
    # We have 7 "super-terms".
    # Each super-term k corresponds to a specific combination of A blocks and B blocks.
    # A blocks: A00, A01, A10, A11
    # B blocks (for 4x4x4, B is 4x4): B00, B01, B10, B11
    
    # The 7 linear combinations for A (rows) and B (cols)
    # We need to map these to the global U, V indices.
    
    # A is 4x4. Index u = i*4 + j.
    # A00: i in 0:2, j in 0:2.
    # A01: i in 0:2, j in 2:4.
    # A10: i in 2:4, j in 0:2.
    # A11: i in 2:4, j in 2:4.
    
    # B is 4x6. For Part 1, we use B[:, 0:4].
    # B blocks:
    # B00: j in 0:2, k in 0:2.
    # B01: j in 0:2, k in 2:4.
    # B10: j in 2:4, k in 0:2.
    # B11: j in 2:4, k in 2:4.
    # Index v = j*6 + k.
    
    # C is 4x6. For Part 1, we use C[:, 0:4].
    # C blocks:
    # C00: i in 0:2, k in 0:2.
    # C01: i in 0:2, k in 2:4.
    # C10: i in 2:4, k in 0:2.
    # C11: i in 2:4, k in 2:4.
    # Index w = k*4 + i.
    
    # The 7 Strassen terms for the top level (4x4x4):
    # M1 = A00 * (B00 + B11)
    # M2 = A10 * (B00 + B01)
    # M3 = (A00 + A01) * B11
    # M4 = (A10 + A11) * B00
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 - A10) * (B00 + B01)
    
    # For each M_k, we perform a 2x2x2 multiplication of the resulting 2x2 matrices.
    # The resulting 2x2 matrices are then placed into C blocks.
    
    col_offset = 0
    for k in range(7):
        # Determine which A blocks and B blocks are involved, and their signs.
        # A_comb: list of (block_id, sign)
        # B_comb: list of (block_id, sign)
        # C_block: which block of C this M_k contributes to?
        # Actually, M_k is a 2x2 matrix. Its entries contribute to specific C blocks.
        # The 2x2 result is R.
        # R00 contributes to C00? No.
        # In Strassen, M1 is a scalar product of 2x2 matrices?
        # No, A00 is 2x2. B00 is 2x2.
        # M1 = A00 * (B00 + B11) is a 2x2 matrix product.
        # The result R1 is 2x2.
        # R1[0,0] contributes to C[0,0]?
        # Let's check indices.
        # A00 is A[0:2, 0:2].
        # B00 is B[0:2, 0:2].
        # Product P = A00 B00. P[i,j] = sum A00[i,l] B00[l,j].
        # This P[i,j] corresponds to C[i,j] where i in 0:2, j in 0:2.
        # So R1 contributes to C00.
        
        # Similarly:
        # M2 = A10 (B00+B01). A10 is A[2:4, 0:2].
        # Product contributes to C[2:4, 0:2] and C[2:4, 2:4]?
        # B00 is B[0:2, 0:2]. B01 is B[0:2, 2:4].
        # A10 B00 -> C[2:4, 0:2] (C10).
        # A10 B01 -> C[2:4, 2:4] (C11).
        
        # So each M_k is a 2x2 matrix product, and its 4 entries map to 4 specific 2x2 blocks of C?
        # No, the 2x2 result R has entries R[0,0], R[0,1], R[1,0], R[1,1].
        # These map to the 4 quadrants of the C block that M_k is responsible for.
        
        # Let's define the mapping for each k.
        # k=0 (M1): A=A00, B=B00+B11. C Block: C00?
        # Wait, B11 is B[2:4, 2:4].
        # A00 is A[0:2, 0:2].
        # A00 B11 is not defined in standard matrix mult unless B is square.
        # Here B is 4x4.
        # A00 (2x2) * B11 (2x2) is valid.
        # Result contributes to C[0:2, 2:4] (C01)?
        # No. A[0:2, 0:2] * B[2:4, 2:4] is not a valid slice product for C.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # If i in 0:2, k in 2:4.
        # j must be in 0:4.
        # This decomposition works because we are summing over j.
        # The Strassen trick rewrites the sum.
        
        # Let's trust the standard Strassen indices.
        # For 4x4x4:
        # M1 = A00 * (B00 + B11).
        # This product is 2x2.
        # The resulting 2x2 matrix R1 has entries R1[0,0], R1[0,1], R1[1,0], R1[1,1].
        # These correspond to the 2x2 block C[0:2, 0:2]?
        # No, B11 involves k in 2:4.
        # So R1[0,1] would correspond to k in 2:4?
        # The columns of R1 correspond to the columns of B.
        # B00 has cols 0,1. B11 has cols 2,3.
        # (B00+B11) is a 4x4 matrix? No, B00 and B11 are 2x2.
        # They are disjoint in the B matrix.
        # B is 4x4.
        # B00 = B[0:2, 0:2].
        # B11 = B[2:4, 2:4].
        # B00 + B11 is not a standard matrix addition if we don't align them.
        # In Strassen, we form the matrix $B^{(1)} = \begin{pmatrix} B_{00} & 0 \\ 0 & B_{11} \end{pmatrix}$?
        # No, the linear combination is on the *blocks*.
        # $M_1 = A_{00} (B_{00} + B_{11})$ means we compute the product of $A_{00}$ with the matrix formed by adding $B_{00}$ and $B_{11}$?
        # $B_{00}$ and $B_{11}$ are $2 \times 2$.
        # So $B_{temp} = B_{00} + B_{11}$ is $2 \times 2$.
        # Then $M_1 = A_{00} B_{temp}$ is $2 \times 2$.
        # The entries of $M_1$ correspond to specific positions in $C$.
        # $M_1[0,0]$ corresponds to $C[0,0]$?
        # $M_1[0,1]$ corresponds to $C[0,1]$?
        # Yes, because $A_{00}$ rows are 0,1 and $B_{temp}$ cols are 0,1 (representing the combined columns).
        # But $B_{11}$ columns are 2,3.
        # So $M_1[0,1]$ actually gets contribution from $B_{11}[?, 1]$?
        # No, $B_{temp}$ is just a virtual matrix.
        # The result $M_1$ is $2 \times 2$.
        # $M_1[0,0]$ maps to $C[0,0]$ and $C[0,2]$?
        # This mapping is tricky.
        
        # Alternative: Just use the recursive Strassen code that generates the flat U, V, W vectors.
        
        pass # Placeholder for logic
