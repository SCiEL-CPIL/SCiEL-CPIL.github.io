import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 23 decomposition for 3x3x3 part (B cols 0-2, C cols 0-2)
    # Plus rank 9 schoolbook for 4th column (B col 3, C col 3)
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    # Helper to add a term
    def add_term(i, j, k, col, u_val=1, v_val=1, w_val=1):
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val

    # --- Part 1: 3x3x3 Decomposition (Rank 23) ---
    # This computes C[0:2, 0:2], C[0:2, 2], C[2, 0:2], C[2, 2]
    # using A[0:2, 0:2], A[0:2, 2], A[2, 0:2], A[2, 2]
    # and B[0:2, 0:2], B[0:2, 2], B[2, 0:2], B[2, 2]
    
    col = 0
    
    # 1. Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B00)
    # M4 = A11 (B01 - B11)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition.
    # Each Strassen "M" is a product of sums.
    # This requires expanding into linear terms.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This contributes to C00 and C11.
    # C00 += M1, C11 += M1.
    # So we add terms for A00B00->C00, A00B00->C11? No.
    # The decomposition is linear.
    # We just add the 4 products as separate terms, but we must ensure they contribute to the correct C entries.
    # Actually, Strassen reduces the number of *multiplications*, but in tensor decomposition rank,
    # each product $A_{ij} B_{jk}$ is a rank 1 term.
    # Wait, Strassen works by *sharing* intermediate sums.
    # In tensor decomposition, we can't "share" sums in the same way unless we combine terms.
    # However, the rank of the tensor is the minimum number of rank-1 terms.
    # Strassen shows that 7 multiplications are enough.
    # This implies the tensor rank is 7.
    # So we can just list the 7 rank-1 terms that make up Strassen's algorithm?
    # No, Strassen's algorithm uses 7 *general* multiplications.
    # The terms are not just $A_{ij} B_{jk}$. They are linear combinations.
    # Example: $M_1 = (A_{00} + A_{11})(B_{00} + B_{11})$.
    # This is NOT a rank 1 term in the original tensor $T_{ijk}$.
    # It is a rank 1 term in a *different* tensor (the one for the linear combinations).
    # To express the original tensor as a sum of rank 1 terms, we must expand.
    # If we expand M1, we get 4 terms.
    # Then total terms would be high.
    
    # CORRECTION:
    # The rank of the $2 \times 2 \times 2$ tensor is 7.
    # This means there exist 7 triples of vectors $u_t, v_t, w_t$ such that
    # $T_{ijk} = \sum_{t=1}^7 u_{t, i, j} v_{t, j, k} w_{t, k, i}$.
    # These $u_t, v_t, w_t$ are the matrices in the decomposition.
    # I need the explicit 7 matrices for Strassen's rank 7 decomposition.
    
    # The Strassen decomposition matrices are:
    # U1 = [[1,0],[1,0]], V1 = [[1,0],[0,0]], W1 = [[1,0],[0,0]]? No.
    # Let's use the known 7 terms.
    
    # I will use the explicit 7 terms for 2x2x2.
    # Term 1: A00 B00 -> C00, C11? No.
    # The 7 terms are:
    # 1. (A00+A11)(B00+B11) -> C00, C11
    # 2. (A10+A11)B00 -> C10
    # 3. A00(B10-B00) -> C01
    # 4. A11(B01-B11) -> C00, C11? No.
    
    # Actually, I can just use the schoolbook for 2x2x2 (rank 4) if I don't have the Strassen matrices.
    # But 4 > 2.
    # If I use schoolbook for 2x2x2, the total rank for 3x3x3 becomes:
    # 4 (2x2x2) + 1 (abT) + 4 (A2x2b) + 2 (ab11) + 4 (aTB2x2) + 2 (a11bT) + 2 (aTb) + 1 (a11b11) = 20.
    # Wait, schoolbook for 2x2x2 is 4.
    # My previous sum was 23 using Strassen (7).
    # If I use schoolbook (4), sum is 20?
    # 4 + 1 + 4 + 2 + 4 + 2 + 2 + 1 = 20.
    # Is rank 20 possible for 3x3x3?
    # Lower bound is 12.
    # Schoolbook is 27.
    # If I can construct rank 20, that's great.
    
    # Let's verify the schoolbook 2x2x2 part.
    # C00 = A00B00 + A01B10. (2 terms)
    # C01 = A00B01 + A01B11. (2 terms)
    # C10 = A10B00 + A11B10. (2 terms)
    # C11 = A10B01 + A11B11. (2 terms)
    # Total 8 terms? No, 2x2x2 schoolbook is 4 terms?
    # No, 2x2x2 schoolbook is 4 multiplications?
    # C00 = A00B00 + A01B10.
    # C01 = A00B01 + A01B11.
    # C10 = A10B00 + A11B10.
    # C11 = A10B01 + A11B11.
    # Total 8 terms.
    # Strassen reduces 8 to 7.
    
    # So if I use schoolbook for 2x2x2, I use 8 terms.
    # Total: 8 + 1 + 4 + 2 + 4 + 2 + 2 + 1 = 24.
    # Rank 24.
    # Plus 9 for the 4th column = 33.
    
    # This is a safe, constructive decomposition with rank 33.
    # It is better than 36.

    # I will implement the rank 33 decomposition.
    
    # Reset U, V, W to 0
    U = [[0] * 33 for _ in range(9)]
    V = [[0] * 33 for _ in range(12)]
    W = [[0] * 33 for _ in range(12)]
    
    col = 0
    # 1. Schoolbook 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # C00
    add_term(0, 0, 0, col); col+=1
    add_term(0, 1, 0, col); col+=1
    # C01
    add_term(0, 0, 1, col); col+=1
    add_term(0, 1, 1, col); col+=1
    # C10
    add_term(1, 0, 0, col); col+=1
    add_term(1, 1, 0, col); col+=1
    # C11
    add_term(1, 0, 1, col); col+=1
    add_term(1, 1, 1, col); col+=1
    
    # 2. a b^T: A[i, 2] B[2, j] -> C[i, j] for i,j in 0,1
    # i=0, j=0
    add_term(0, 2, 0, col); col+=1
    # i=0, j=1
    add_term(0, 2, 1, col); col+=1
    # i=1, j=0
    add_term(1, 2, 0, col); col+=1
    # i=1, j=1
    add_term(1, 2, 1, col); col+=1
    # Wait, a b^T is rank 1?
    # C_ij = A_i2 B_2j.
    # This is 4 terms in schoolbook.
    # But can it be 1 term?
    # U_i2 = 1, V_2j = 1, W_ji = 1.
    # Yes, it is 1 term in the tensor decomposition sense?
    # No, the tensor entry is T_{i, 2, j} = 1.
    # The decomposition is sum_r U_{i,2,r} V_{2,j,r} W_{j,i,r}.
    # If we set r=0, U_{0,2,0}=1, U_{1,2,0}=1, V_{2,0,0}=1, V_{2,1,0}=1, W_{0,0,0}=1, W_{1,0,0}=1, W_{0,1,0}=1, W_{1,1,0}=1.
    # Then for i=0, j=0: U_02 V_20 W_00 = 1. Correct.
    # For i=0, j=1: U_02 V_21 W_01 = 1. Correct.
    # So yes, it is 1 term.
    # I added 4 terms above. I should use 1 term.
    # Let's fix this.
    
    # Re-doing Part 2 with 1 term.
    # But I already incremented col. I'll just overwrite or adjust.
    # It's easier to restart the logic.

    U = [[0] * 33 for _ in range(9)]
    V = [[0] * 33 for _ in range(12)]
    W = [[0] * 33 for _ in range(12)]
    col = 0
    
    # 1. Schoolbook 2x2x2 (8 terms)
    add_term(0, 0, 0, col); col+=1
    add_term(0, 1, 0, col); col+=1
    add_term(0, 0, 1, col); col+=1
    add_term(0, 1, 1, col); col+=1
    add_term(1, 0, 0, col); col+=1
    add_term(1, 1, 0, col); col+=1
    add_term(1, 0, 1, col); col+=1
    add_term(1, 1, 1, col); col+=1
    
    # 2. a b^T (1 term)
    # U[i, 2, col] = 1 for i=0,1
    # V[2, j, col] = 1 for j=0,1
    # W[j, i, col] = 1 for j=0,1, i=0,1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = 1
    V[2 * 4 + 0][col] = 1
    V[2 * 4 + 1][col] = 1
    W[0 * 3 + 0][col] = 1
    W[0 * 3 + 1][col] = 1
    W[1 * 3 + 0][col] = 1
    W[1 * 3 + 1][col] = 1
    col += 1
    
    # 3. A2x2 b (4 terms)
    # C[i, 2] += A[i, k] B[k, 2]
    add_term(0, 0, 2, col); col+=1
    add_term(0, 1, 2, col); col+=1
    add_term(1, 0, 2, col); col+=1
    add_term(1, 1, 2, col); col+=1
    
    # 4. a b11 (2 terms)
    # C[i, 2] += A[i, 2] B[2, 2]
    add_term(0, 2, 2, col); col+=1
    add_term(1, 2, 2, col); col+=1
    
    # 5. aT B2x2 (4 terms)
    # C[2, j] += A[2, k] B[k, j]
    add_term(2, 0, 0, col); col+=1
    add_term(2, 0, 1, col); col+=1
    add_term(2, 1, 0, col); col+=1
    add_term(2, 1, 1, col); col+=1
    
    # 6. a11 bT (2 terms)
    # C[2, j] += A[2, 2] B[2, j]
    add_term(2, 2, 0, col); col+=1
    add_term(2, 2, 1, col); col+=1
    
    # 7. aT b (2 terms)
    # C[2, 2] += A[2, k] B[k, 2]
    add_term(2, 0, 2, col); col+=1
    add_term(2, 1, 2, col); col+=1
    
    # 8. a11 b11 (1 term)
    add_term(2, 2, 2, col); col+=1
    
    # Current col = 8 + 1 + 4 + 2 + 4 + 2 + 2 + 1 = 24.
    
    # 9. Schoolbook for 4th column (9 terms)
    # C[i, 3] += A[i, j] B[j, 3]
    for i in range(3):
        for j in range(3):
            add_term(i, j, 3, col); col+=1
            
    # Total col = 24 + 9 = 33.
    
    return {"U": U, "V": V, "W": W}
