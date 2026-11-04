import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    n, m, p = 4, 4, 5
    
    # Total rank
    R = 44
    
    U = np.zeros((n * m, R))  # (16, 44)
    V = np.zeros((m * p, R))  # (20, 44)
    W = np.zeros((p * n, R))  # (20, 44)
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, coeff):
        """
        Adds a rank-1 term U[i,j] * V[j,k] * W[k,i] * coeff
        i: row of A (0..3)
        j: col of A / row of B (0..3)
        k: col of B (0..4)
        """
        nonlocal col
        U[i * m + j, col] = coeff
        V[j * p + k, col] = coeff
        W[k * n + i, col] = coeff
        col += 1

    # --- Part 1: 4x4 x 4x4 using Strassen ---
    # A blocks:
    # A00: i in [0,1], j in [0,1]
    # A01: i in [0,1], j in [2,3]
    # A10: i in [2,3], j in [0,1]
    # A11: i in [2,3], j in [2,3]
    
    # B1 blocks (cols 0..3 of B):
    # B00: j in [0,1], k in [0,1]
    # B01: j in [0,1], k in [2,3]
    # B10: j in [2,3], k in [0,1]
    # B11: j in [2,3], k in [2,3]
    
    # C1 blocks (cols 0..3 of C):
    # C00: i in [0,1], k in [0,1]
    # C01: i in [0,1], k in [2,3]
    # C10: i in [2,3], k in [0,1]
    # C11: i in [2,3], k in [2,3]
    
    # Strassen 7 products.
    # Each product is 2x2. We expand each into 4 schoolbook terms.
    
    # M1 = (A00 + A11) * (B00 + B11)
    # Contributions to C00 and C11?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to trace the coefficients.
    # Let's define the linear combinations for A and B for each M.
    # M1:
    # A_part = A00 + A11
    # B_part = B00 + B11
    # The product is a 2x2 matrix.
    # (A00+A11)_{pq} (B00+B11)_{qr} = sum_q ...
    # But we are decomposing the tensor.
    # The term for C_{i,k} is sum_{j} A_{i,j} B_{j,k}.
    # In Strassen, C_{i,k} is a linear combination of the M's.
    # M_l = X_l Y_l.
    # X_l is a 2x2 matrix, Y_l is a 2x2 matrix.
    # X_l Y_l is a 2x2 matrix.
    # The (a,b) entry of M_l is (X_l Y_l)_{a,b} = sum_{q=0}^1 X_{a,q} Y_{q,b}.
    # Each X_{a,q} is a linear combo of A entries.
    # Each Y_{q,b} is a linear combo of B entries.
    # So (X_l Y_l)_{a,b} is a sum of terms A_{i,j} B_{j,k}.
    # We need to add these terms to U, V, W with the correct coefficients for C_{i,k}.
    
    # This is getting complex to do symbolically in code quickly.
    # Let's use a known explicit expansion or a simpler reduction.
    # Actually, we can just hardcode the 28 terms for the 4x4x4 Strassen part
    # and 16 terms for the 4x4x1 part.
    
    # Let's verify if a simpler rank is possible.
    # Rank 44 is safe.
    
    # To ensure correctness, I will implement the Strassen expansion carefully.
    
    # Indices for 2x2 blocks within the 4x4:
    # Local indices (a, b) for 2x2 block.
    # Global i = 2*block_i + a
    # Global j = 2*block_j + b
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00: global (0,0), (0,1), (1,0), (1,1)
    # A11: global (2,2), (2,3), (3,2), (3,3)
    # B00: global (0,0), (0,1), (1,0), (1,1)
    # B11: global (2,2), (2,3), (3,2), (3,3)
    
    # Let's define a helper to add a "block multiplication" contribution.
    # This is still tricky because of the sums.
    
    # Alternative: Use the fact that the seed's logic for "schoolbook" is correct.
    # I will construct the 44-rank decomposition by explicitly listing the 28 Strassen terms
    # expanded into scalar multiplications.
    
    # M1 terms:
    # (A00+A11)_{pq} (B00+B11)_{qr}
    # p,q,r in 0,1.
    # (A00+A11)_{pq} = A_{2*0+p, 2*0+q} + A_{2*1+p, 2*1+q} = A_{p,q} + A_{p+2,q+2}
    # (B00+B11)_{qr} = B_{2*0+q, 2*0+r} + B_{2*1+q, 2*1+r} = B_{q,r} + B_{q+2,r+2}
    
    # Product term for local (p,r) of M1:
    # Sum_{q=0}^1 (A_{p,q} + A_{p+2,q+2}) (B_{q,r} + B_{q+2,r+2})
    # = (A_{p,q} B_{q,r} + A_{p,q} B_{q+2,r+2} + A_{p+2,q+2} B_{q,r} + A_{p+2,q+2} B_{q+2,r+2})
    # for q=0,1.
    
    # This M1 contributes to C00 (top-left 2x2) and C11 (bottom-right 2x2)?
    # No, C00 = M1 + M4 - M5 + M7.
    # C11 = M1 - M2 + M3 + M6.
    # So M1 contributes +1 to C00 and +1 to C11.
    # The local index (p,r) in M1 corresponds to global C index:
    # For C00: i=p, k=r.
    # For C11: i=p+2, k=r+2.
    
    # So for each q in 0,1 and p,r in 0,1:
    # Term 1: A_{p,q} B_{q,r} -> C_{p,r} (coeff 1) and C_{p+2,r+2} (coeff 1)
    # Term 2: A_{p,q} B_{q+2,r+2} -> C_{p,r} (coeff 1) and C_{p+2,r+2} (coeff 1)
    # Term 3: A_{p+2,q+2} B_{q,r} -> C_{p,r} (coeff 1) and C_{p+2,r+2} (coeff 1)
    # Term 4: A_{p+2,q+2} B_{q+2,r+2} -> C_{p,r} (coeff 1) and C_{p+2,r+2} (coeff 1)
    
    # Wait, the tensor decomposition is linear.
    # We are decomposing the identity tensor.
    # T[i,j,k] = 1.
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # The Strassen algorithm computes C = AB.
    # The "rank" is the number of scalar multiplications.
    # Each scalar multiplication in Strassen corresponds to a rank-1 term in the decomposition?
    # Yes, if we view the Strassen algorithm as a basis change.
    
    # So, for M1, we have 4 (from p,r) * 2 (from q) * 4 (from expansion) = 32 scalar terms?
    # No.
    # M1 is a 2x2 matrix. It has 4 entries.
    # Each entry is a sum of 4 scalar products (from the expansion of the sums).
    # So M1 involves 4 * 4 = 16 scalar products?
    # But Strassen says M1 is ONE multiplication of two 2x2 matrices.
    # The rank of a 2x2 multiplication is 4.
    # So M1 contributes 4 to the rank.
    # This means we should NOT expand the sums into 16 terms.
    # We should treat the 2x2 multiplication as a primitive of rank 4.
    # But the primitive 2x2 multiplication itself is decomposed into 4 rank-1 terms (schoolbook).
    # So M1 contributes 4 rank-1 terms to the global decomposition.
    
    # How do we map the 4 rank-1 terms of M1 to the global U,V,W?
    # The 4 rank-1 terms of a 2x2 multiplication X Y = Z are:
    # 1. X_{00} Y_{00} -> Z_{00}
    # 2. X_{01} Y_{01} -> Z_{01}
    # 3. X_{10} Y_{10} -> Z_{10}
    # 4. X_{11} Y_{11} -> Z_{11}
    
    # For M1:
    # X = A00 + A11. Y = B00 + B11.
    # Term 1: X_{00} Y_{00} = (A_{00} + A_{22}) (B_{00} + B_{22})
    # This is a product of two linear forms.
    # In the tensor decomposition, we can have U entries that are sums.
    # U[i,j,r] is the coefficient of A_{i,j} in the r-th term.
    # So for Term 1 of M1:
    # It contributes to Z_{00} of M1.
    # Z_{00} of M1 contributes to C_{00} (global) and C_{22} (global)?
    # No, M1 is the top-left block of the result of the Strassen step?
    # No, M1 is a 2x2 matrix.
    # C00 (2x2 block) = M1 + ...
    # So Z_{00} of M1 contributes to C_{0,0} global.
    # Z_{01} of M1 contributes to C_{0,1} global.
    # Z_{10} of M1 contributes to C_{1,0} global.
    # Z_{11} of M1 contributes to C_{1,1} global.
    
    # And M1 also contributes to C11 (2x2 block) = M1 + ...
    # So Z_{00} of M1 ALSO contributes to C_{2,2} global?
    # Yes, if the formula is C11 = M1 + ...
    
    # So, for Term 1 of M1:
    # U part: A_{0,0} and A_{2,2} (coeff 1 each).
    # V part: B_{0,0} and B_{2,2} (coeff 1 each).
    # W part: C_{0,0} (coeff 1) and C_{2,2} (coeff 1).
    
    # This single rank-1 term (in the Strassen sense) actually requires multiple non-zero entries in U, V, W.
    # But it counts as ONE rank-1 term in the global decomposition?
    # No!
    # The rank of the tensor is the minimum number of rank-1 tensors needed.
    # A rank-1 tensor is U_r \otimes V_r \otimes W_r.
    # If U_r has two non-zero entries, it is still a vector.
    # So yes, this counts as 1 rank.
    
    # So, for M1, we have 4 terms (schoolbook for 2x2).
    # Term 1: X_{00} Y_{00} -> Z_{00}
    # U: A_{0,0}=1, A_{2,2}=1
    # V: B_{0,0}=1, B_{2,2}=1
    # W: C_{0,0}=1, C_{2,2}=1
    
    # Term 2: X_{01} Y_{01} -> Z_{01}
    # X_{01} = A_{0,1} + A_{2,3}
    # Y_{01} = B_{0,1} + B_{2,3}
    # U: A_{0,1}=1, A_{2,3}=1
    # V: B_{0,1}=1, B_{2,3}=1
    # W: C_{0,1}=1, C_{2,3}=1
    
    # Term 3: X_{10} Y_{10} -> Z_{10}
    # X_{10} = A_{1,0} + A_{3,2}
    # Y_{10} = B_{1,0} + B_{3,2}
    # U: A_{1,0}=1, A_{3,2}=1
    # V: B_{1,0}=1, B_{3,2}=1
    # W: C_{1,0}=1, C_{3,2}=1
    
    # Term 4: X_{11} Y_{11} -> Z_{11}
    # X_{11} = A_{1,1} + A_{3,3}
    # Y_{11} = B_{1,1} + B_{3,3}
    # U: A_{1,1}=1, A_{3,3}=1
    # V: B_{1,1}=1, B_{3,3}=1
    # W: C_{1,1}=1, C_{3,3}=1
    
    # This accounts for 4 ranks for M1.
    
    # We need to do this for all 7 M's.
    # And handle the signs and which C blocks they contribute to.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 contributes to C00 (coeff 1) and C11 (coeff 1).
    # M2 contributes to C10 (coeff 1) and C11 (coeff -1).
    # M3 contributes to C01 (coeff 1) and C11 (coeff 1).
    # M4 contributes to C00 (coeff 1) and C10 (coeff 1).
    # M5 contributes to C00 (coeff -1) and C01 (coeff 1).
    # M6 contributes to C11 (coeff 1).
    # M7 contributes to C00 (coeff 1).
    
    # We will iterate through M1..M7, and for each, generate 4 terms.
    
    def add_strassen_term(m_idx, x_a, x_b, y_a, y_b, c_targets):
        """
        m_idx: not used for logic, just for tracking
        x_a, x_b: indices (i,j) for the two A entries in X
        y_a, y_b: indices (j,k) for the two B entries in Y
        c_targets: list of (i,k,coeff) for C
        """
        nonlocal col
        # U
        U[x_a[0]*m + x_a[1], col] = 1
        U[x_b[0]*m + x_b[1], col] = 1
        # V
        V[y_a[0]*p + y_a[1], col] = 1
        V[y_b[0]*p + y_b[1], col] = 1
        # W
        for i, k, coeff in c_targets:
            W[k*n + i, col] = coeff
        col += 1

    # Define A indices for blocks
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # M1: X = A00+A11, Y = B00+B11
    # X00 = A(0,0)+A(2,2), Y00 = B(0,0)+B(2,2) -> C00(0,0), C11(2,2)
    add_strassen_term(1, (0,0), (2,2), (0,0), (2,2), [(0,0,1), (2,2,1)])
    # X01 = A(0,1)+A(2,3), Y01 = B(0,1)+B(2,3) -> C00(0,1), C11(2,3)
    add_strassen_term(1, (0,1), (2,3), (0,1), (2,3), [(0,1,1), (2,3,1)])
    # X10 = A(1,0)+A(3,2), Y10 = B(1,0)+B(3,2) -> C00(1,0), C11(3,2)
    add_strassen_term(1, (1,0), (3,2), (1,0), (3,2), [(1,0,1), (3,2,1)])
    # X11 = A(1,1)+A(3,3), Y11 = B(1,1)+B(3,3) -> C00(1,1), C11(3,3)
    add_strassen_term(1, (1,1), (3,3), (1,1), (3,3), [(1,1,1), (3,3,1)])
    
    # M2: X = A10+A11, Y = B00
    # X00 = A(2,0)+A(2,2), Y00 = B(0,0) -> C10(2,0), C11(2,0) with coeff -1
    add_strassen_term(2, (2,0), (2,2), (0,0), (0,0), [(2,0,1), (2,0,-1)])
    # X01 = A(2,1)+A(2,3), Y01 = B(0,1) -> C10(2,1), C11(2,1) with coeff -1
    add_strassen_term(2, (2,1), (2,3), (0,1), (0,1), [(2,1,1), (2,1,-1)])
    # X10 = A(3,0)+A(3,2), Y10 = B(1,0) -> C10(3,0), C11(3,0) with coeff -1
    add_strassen_term(2, (3,0), (3,2), (1,0), (1,0), [(3,0,1), (3,0,-1)])
    # X11 = A(3,1)+A(3,3), Y11 = B(1,1) -> C10(3,1), C11(3,1) with coeff -1
    add_strassen_term(2, (3,1), (3,3), (1,1), (1,1), [(3,1,1), (3,1,-1)])
    
    # M3: X = A00, Y = B01-B11
    # X00 = A(0,0), Y00 = B(0,2)-B(2,2) -> C01(0,2), C11(2,2)
    add_strassen_term(3, (0,0), (0,0), (0,2), (2,2), [(0,2,1), (2,2,1)])
    # Wait, Y00 = B(0,2) - B(2,2). So coeff for B(2,2) is -1.
    # My helper assumes coeff 1 for all. I need to handle signs.
    
    # Let's rewrite helper to accept coefficients.
    
    def add_strassen_term_signed(m_idx, x_entries, y_entries, c_targets):
        """
        x_entries: list of (i, j, coeff)
        y_entries: list of (j, k, coeff)
        c_targets: list of (i, k, coeff)
        """
        nonlocal col
        for i, j, c in x_entries:
            U[i*m + j, col] += c
        for j, k, c in y_entries:
            V[j*p + k, col] += c
        for i, k, c in c_targets:
            W[k*n + i, col] += c
        col += 1

    # Reset col and matrices to use the new helper
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    
    # M1
    add_strassen_term_signed(1, [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)])
    add_strassen_term_signed(1, [(0,1,1), (2,3,1)], [(0,1,1), (2,3,1)], [(0,1,1), (2,3,1)])
    add_strassen_term_signed(1, [(1,0,1), (3,2,1)], [(1,0,1), (3,2,1)], [(1,0,1), (3,2,1)])
    add_strassen_term_signed(1, [(1,1,1), (3,3,1)], [(1,1,1), (3,3,1)], [(1,1,1), (3,3,1)])
    
    # M2
    add_strassen_term_signed(2, [(2,0,1), (2,2,1)], [(0,0,1), (0,0,1)], [(2,0,1), (2,0,-1)])
    add_strassen_term_signed(2, [(2,1,1), (2,3,1)], [(0,1,1), (0,1,1)], [(2,1,1), (2,1,-1)])
    add_strassen_term_signed(2, [(3,0,1), (3,2,1)], [(1,0,1), (1,0,1)], [(3,0,1), (3,0,-1)])
    add_strassen_term_signed(2, [(3,1,1), (3,3,1)], [(1,1,1), (1,1,1)], [(3,1,1), (3,1,-1)])
    
    # M3: X=A00, Y=B01-B11
    add_strassen_term_signed(3, [(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)])
    add_strassen_term_signed(3, [(0,1,1)], [(0,3,1), (2,3,-1)], [(0,3,1), (2,3,1)])
    add_strassen_term_signed(3, [(1,0,1)], [(1,2,1), (3,2,-1)], [(1,2,1), (3,2,1)])
    add_strassen_term_signed(3, [(1,1,1)], [(1,3,1), (3,3,-1)], [(1,3,1), (3,3,1)])
    
    # M4: X=A11, Y=B10-B00
    add_strassen_term_signed(4, [(2,2,1)], [(2,0,1), (0,0,-1)], [(2,0,1), (0,0,1)])
    add_strassen_term_signed(4, [(2,3,1)], [(2,1,1), (0,1,-1)], [(2,1,1), (0,1,1)])
    add_strassen_term_signed(4, [(3,2,1)], [(3,0,1), (1,0,-1)], [(3,0,1), (1,0,1)])
    add_strassen_term_signed(4, [(3,3,1)], [(3,1,1), (1,1,-1)], [(3,1,1), (1,1,1)])
    
    # M5: X=A00+A01, Y=B11
    add_strassen_term_signed(5, [(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (2,2,1)])
    add_strassen_term_signed(5, [(0,1,1), (0,3,1)], [(2,3,1)], [(0,1,-1), (2,3,1)])
    add_strassen_term_signed(5, [(1,0,1), (1,2,1)], [(3,2,1)], [(1,0,-1), (3,2,1)])
    add_strassen_term_signed(5, [(1,1,1), (1,3,1)], [(3,3,1)], [(1,1,-1), (3,3,1)])
    
    # M6: X=A10-A00, Y=B00+B01
    add_strassen_term_signed(6, [(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,0,1)])
    add_strassen_term_signed(6, [(2,1,1), (0,1,-1)], [(0,1,1), (0,3,1)], [(2,1,1)])
    add_strassen_term_signed(6, [(3,0,1), (1,0,-1)], [(1,0,1), (1,2,1)], [(3,0,1)])
    add_strassen_term_signed(6, [(3,1,1), (1,1,-1)], [(1,1,1), (1,3,1)], [(3,1,1)])
    
    # M7: X=A01-A11, Y=B10+B11
    add_strassen_term_signed(7, [(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1)])
    add_strassen_term_signed(7, [(0,3,1), (2,3,-1)], [(2,1,1), (2,3,1)], [(0,1,1)])
    add_strassen_term_signed(7, [(1,2,1), (3,2,-1)], [(3,0,1), (3,2,1)], [(1,0,1)])
    add_strassen_term_signed(7, [(1,3,1), (3,3,-1)], [(3,1,1), (3,3,1)], [(1,1,1)])
    
    # --- Part 2: 4x4 x 4x1 Schoolbook ---
    # B2 is cols 4 of B (index 4).
    # C2 is col 4 of C.
    for i in range(4):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
            
    # Check if we used all 44 ranks
    if col != R:
        print(f"Warning: used {col} ranks, expected {R}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
