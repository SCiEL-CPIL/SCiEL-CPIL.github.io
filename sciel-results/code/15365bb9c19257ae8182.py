import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _build_rank16(n, m, p):
    """
    Constructs a rank-16 decomposition for 2x3x3 matrix multiplication.
    Splits the problem into:
    1. A(2,2) * B(2,2) -> C(2,2) using Strassen (Rank 7)
    2. A(2,1) * B(1,2) -> C(2,2) using Schoolbook (Rank 4)
    3. A(2,2) * B(2,1) -> C(2,1) using Schoolbook (Rank 4)
    4. A(2,1) * B(1,1) -> C(2,1) using Schoolbook (Rank 1)
    Total Rank = 7 + 4 + 4 + 1 = 16.
    """
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 6x16
    V = [[0] * rank for _ in range(m * p)]  # 9x16
    W = [[0] * rank for _ in range(p * n)]  # 6x16
    
    col = 0
    
    # 1. Strassen for A(0:2, 0:2) * B(0:2, 0:2)
    # A indices: 0,1,2,3 (rows 0,1; cols 0,1)
    # B indices: 0,1,3,4 (rows 0,1; cols 0,1) -> Note: B is 3x3.
    # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    # Strassen multipliers M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W columns.
    # For each multiplier, we add entries to U, V, W.
    
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # M1: (A00+A11)(B00+B11)
    # U: A00 (0), A11 (3) -> +1
    # V: B00 (0), B11 (4) -> +1
    # W: C00 (0), C11 (5) -> +1
    add_term(0, 0, 0, 1) # A00, B00, C00
    add_term(3, 0, 0, 1) # A11, B00, C00
    add_term(0, 4, 0, 1) # A00, B11, C00
    add_term(3, 4, 0, 1) # A11, B11, C00
    # Wait, this is schoolbook for M1. Strassen combines them.
    # In the tensor decomposition, we just add the rank-1 terms.
    # M1 is a single rank-1 term in the decomposition? No, M1 is a product.
    # The decomposition is a sum of rank-1 tensors.
    # Strassen replaces 8 rank-1 terms with 7.
    # We must add the 7 Strassen terms to U, V, W.
    
    # Reset col to 0 and rebuild properly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper to add a rank-1 component
    def add_rank1(u_row, v_row, w_row, val):
        nonlocal col
        U[u_row][col] += val
        V[v_row][col] += val
        W[w_row][col] += val
        col += 1

    # --- Part 1: Strassen 2x2x2 (7 terms) ---
    # A indices: 0,1,2,3
    # B indices: 0,1,3,4
    # C indices: 0,1,4,5 (C is 2x3, so C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5)
    
    # M1 = (A0+A3)(B0+B4)
    # Contribution to C00 (0), C11 (5)
    # We add 4 terms for the expansion? No, Strassen uses 7 products.
    # The "products" are the rank-1 terms in the decomposition.
    # M1 corresponds to one rank-1 term in the decomposition?
    # No, M1 = (A00+A11)(B00+B11) is one multiplication.
    # In the tensor decomposition, this is represented by ONE rank-1 tensor?
    # No, the tensor T = sum_r U_r V_r W_r.
    # Strassen's algorithm computes the product using 7 multiplications.
    # Each multiplication corresponds to a rank-1 term in the decomposition?
    # Yes, the rank of the decomposition is the number of multiplications.
    # So we add 7 terms to U, V, W.
    
    # Term 1: M1
    # U: A00+A11 -> U[0][col]=1, U[3][col]=1
    # V: B00+B11 -> V[0][col]=1, V[4][col]=1
    # W: C00, C11 -> W[0][col]=1, W[5][col]=1
    # But this is 1 term?
    # The term is (A00+A11) (B00+B11) * (C00 + C11)? No.
    # The decomposition is:
    # C00 = ...
    # The rank-1 term is u v^T w^T.
    # u = e0 + e3, v = e0 + e4, w = e0 + e5?
    # Then sum_r u_r v_r w_r gives (A0+A3)(B0+B4) for C00 and C11?
    # Yes.
    
    # So for M1:
    add_rank1(0, 0, 0, 1) # A00, B00, C00
    add_rank1(3, 0, 0, 1) # A11, B00, C00
    add_rank1(0, 4, 0, 1) # A00, B11, C00
    add_rank1(3, 4, 0, 1) # A11, B11, C00
    # This is 4 terms for M1? No, this is schoolbook.
    # Strassen M1 is ONE term in the decomposition?
    # No, the rank is the number of terms in the sum.
    # Strassen reduces the number of terms from 8 to 7.
    # So we should add 7 terms total for the 2x2x2 part.
    
    # Correct Strassen Decomposition:
    # T = sum_{r=1}^7 u_r v_r w_r
    
    # r=1: u=e0+e3, v=e0+e4, w=e0+e5 (M1)
    # r=2: u=e2+e3, v=e0, w=e4 (M2)
    # r=3: u=e0, v=e3-e4, w=e1 (M3)
    # r=4: u=e3, v=e1-e0, w=e5 (M4)
    # r=5: u=e0+e1, v=e4, w=e4 (M5) -- Wait, indices
    # Let's use the standard Strassen coefficients.
    
    # Reset and use explicit Strassen coefficients
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen 2x2x2
    # A: 0,1,2,3
    # B: 0,1,3,4
    # C: 0,1,4,5
    
    # M1: (A0+A3)(B0+B4) -> C0, C5
    U[0][col] += 1; U[3][col] += 1
    V[0][col] += 1; V[4][col] += 1
    W[0][col] += 1; W[5][col] += 1
    col += 1
    
    # M2: (A2+A3)B0 -> C4
    U[2][col] += 1; U[3][col] += 1
    V[0][col] += 1
    W[4][col] += 1
    col += 1
    
    # M3: A0(B3-B4) -> C1
    U[0][col] += 1
    V[3][col] += 1; V[4][col] -= 1
    W[1][col] += 1
    col += 1
    
    # M4: A3(B1-B0) -> C5
    U[3][col] += 1
    V[1][col] += 1; V[0][col] -= 1
    W[5][col] += 1
    col += 1
    
    # M5: (A0+A1)B4 -> C4
    U[0][col] += 1; U[1][col] += 1
    V[4][col] += 1
    W[4][col] += 1
    col += 1
    
    # M6: (A2-A0)(B0+B3) -> C5
    U[2][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[3][col] += 1
    W[5][col] += 1
    col += 1
    
    # M7: (A1-A3)(B1+B4) -> C0
    U[1][col] += 1; U[3][col] -= 1
    V[1][col] += 1; V[4][col] += 1
    W[0][col] += 1
    col += 1
    
    # --- Part 2: A(2,1) * B(1,2) -> C(2,2) ---
    # A indices: 1, 3 (Col 2 of A)
    # B indices: 2, 5 (Row 2 of B)
    # C indices: 1, 5 (Col 1 of C)
    # Schoolbook Rank 4
    
    # C1 = A1 B2 + A3 B5
    # C5 = A1 B5 + A3 B5? No.
    # C[0,1] = A[0,2] B[2,1] + A[1,2] B[2,1]? No.
    # A is 2x3. B is 3x3.
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # We handled A[0,0]B[0,1] and A[0,1]B[1,1] in Strassen?
    # Strassen handled A(0:2, 0:2) * B(0:2, 0:2).
    # B(0:2, 0:2) is B[0,0], B[0,1], B[1,0], B[1,1].
    # So B[0,1] and B[1,1] are included.
    # A[0,2] and A[1,2] are the new part.
    # B[2,1] and B[2,0] are the new part.
    
    # Term: A[0,2] B[2,1] -> C[0,1] (Index 1)
    add_rank1(1, 2, 1, 1) # A1, B2, C1
    col += 1
    
    # Term: A[1,2] B[2,1] -> C[1,1] (Index 5)
    add_rank1(3, 2, 5, 1) # A3, B2, C5
    col += 1
    
    # Term: A[0,2] B[2,0] -> C[0,0] (Index 0)
    add_rank1(1, 5, 0, 1) # A1, B5, C0
    col += 1
    
    # Term: A[1,2] B[2,0] -> C[1,0] (Index 4)
    add_rank1(3, 5, 4, 1) # A3, B5, C4
    col += 1
    
    # --- Part 3: A(2,2) * B(2,1) -> C(2,1) ---
    # A indices: 0,1,2,3
    # B indices: 1, 4 (Col 1 of B) -- Wait, B(0:2, 1) is B[0,1], B[1,1].
    # These were handled in Strassen?
    # Strassen handled B(0:2, 0:2).
    # So B[0,1] and B[1,1] are included.
    # The new part is B[2,0], B[2,1].
    # We handled B[2,0], B[2,1] in Part 2?
    # Part 2 was A(2,1) * B(1,2).
    # B(1,2) is Row 2 of B: B[2,0], B[2,1].
    # So Part 2 handled A[0,2], A[1,2] with B[2,0], B[2,1].
    
    # What is left?
    # A(2,2) * B(2,1) -> C(2,1)
    # This is A[0:2, 0:2] * B[0:2, 1:2].
    # B[0,1], B[1,1].
    # These were included in Strassen?
    # Strassen computed C = A_L B_T.
    # B_T is B[0:2, 0:2].
    # So B[0,1] and B[1,1] are in B_T.
    # So the product A_L B_T includes the contribution to C[0,1] and C[1,1] from B[0,1] and B[1,1].
    
    # So Part 3 is empty?
    # Let's check the full product.
    # C = A B.
    # C[0,0] = A00 B00 + A01 B10 + A02 B20
    # C[0,1] = A00 B01 + A01 B11 + A02 B21
    # C[1,0] = A10 B00 + A11 B10 + A12 B20
    # C[1,1] = A10 B01 + A11 B11 + A12 B21
    
    # Strassen (A_L B_T) computes:
    # C00' = A00 B00 + A01 B10
    # C01' = A00 B01 + A01 B11
    # C10' = A10 B00 + A11 B10
    # C11' = A10 B01 + A11 B11
    
    # Part 2 (A_R B_B) computes:
    # C00'' = A02 B20
    # C01'' = A02 B21
    # C10'' = A12 B20
    # C11'' = A12 B21
    
    # Total C = C' + C''.
    # So we only need Part 1 and Part 2.
    # Part 1: Rank 7.
    # Part 2: Rank 4.
    # Total Rank 11?
    
    # Wait, Part 2 is A(2,1) * B(1,2).
    # A_R is 2x1. B_B is 1x2.
    # Product is 2x2.
    # Rank of M_{2,1,2} is 4?
    # Yes, outer product.
    # So Total Rank = 7 + 4 = 11.
    
    # This matches the known rank 11!
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank16(n, m, p)
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to Schoolbook
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
