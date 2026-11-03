import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 135s limit to be safe
    
    # 1. Construct a high-quality initial decomposition using Block-Strassen + Schoolbook
    # This is better than pure schoolbook (56) because it exploits structure.
    # M (2x4) * N (4x7)
    # Split M into M1 (2x2) and M2 (2x2)
    # Split N into N1 (4x4) and N2 (4x3)
    
    # P1 = M1*N1_top + M2*N1_bottom  (2x4 result)
    # P2 = M1*N2_top + M2*N2_bottom  (2x3 result)
    
    # Let's use a constructive approach for R(2,4,7).
    # Known bounds: R(2,4,7) <= 40? 
    # Actually, let's construct a specific valid decomposition with rank < 56.
    # We can use the "Border Rank" or explicit formulas.
    
    # Strategy: 
    # 1. Start with Schoolbook (Rank 56).
    # 2. Use a "Greedy Sparse Solver" to reduce rank.
    #    We assume the solution lies in the span of the schoolbook basis.
    #    We look for a subset of columns (or linear combinations) that sums to the tensor.
    
    # A more powerful structural approach:
    # Use the decomposition of (2x2)*(2x2) with rank 7 (Strassen).
    # M = [A, B] (2x2 blocks)
    # N = [C, D; E, F] (2x2 blocks for the 4x4 part) + [G, H, I; J, K, L] (2x3 part? No, N is 4x7)
    
    # Let's split the 7 columns of N into 4 and 3.
    # N_left (4x4), N_right (4x3).
    
    # Part 1: M * N_left (2x4 * 4x4 -> 2x4)
    # This is equivalent to two (2x2 * 2x2) multiplications? No.
    # M = [A, B]. N_left = [[C, D], [E, F]].
    # P_left = [A*C + B*E, A*D + B*F].
    # This requires computing A*C, B*E, A*D, B*F.
    # If we compute these 4 products using Strassen (rank 7 each), total rank 28.
    # But we can do better by sharing the Strassen intermediates?
    # Strassen for A*C and A*D shares A.
    # Strassen for B*E and B*F shares B.
    
    # Let's define the Strassen factors for 2x2 * 2x2.
    # Inputs: A (2x2), B (2x2).
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to construct U, V, W for the full tensor.
    # This is complex to code from scratch. 
    # Alternative: Use the "Kronecker Product" structure.
    # The tensor T_{i,j,k} is 1 if i,j,k correspond to the multiplication.
    
    # Let's try a different structural insight:
    # R(2,4,7) <= R(2,4,4) + R(2,4,3).
    # R(2,4,4) is the rank of (2x4)*(4x4).
    # This is isomorphic to 4 copies of (2x1)*(1x4) + ...? No.
    # (2x4)*(4x4) can be seen as multiplying a 2x4 matrix by 4 different 4x1 columns.
    # Each column multiplication is (2x4)*(4x1) -> 2x1.
    # R(2,4,1) = 8 (Schoolbook).
    # So R(2,4,4) <= 4 * 8 = 32? 
    # But we can share the left matrix M.
    # M * N_col1, M * N_col2, M * N_col3, M * N_col4.
    # This is equivalent to M * [N1 N2 N3 N4].
    # This is just the original problem with p=4.
    
    # Let's look at R(2,4,1).
    # M (2x4) * v (4x1) -> u (2x1).
    # u1 = M11 v1 + M12 v2 + M13 v3 + M14 v4
    # u2 = M21 v1 + M22 v2 + M23 v3 + M24 v4
    # This is a bilinear form. Rank is 8?
    # Actually, R(2,4,1) = 8.
    
    # If we just use schoolbook for the whole thing, rank is 56.
    # Can we do better?
    # R(2,4,7) <= 7 * R(2,4,1)? No, that's 56.
    
    # Let's try to find a rank < 56 decomposition.
    # Alon-Boppana bound: R(m,n,p) >= mnp / (m+n+p-2)? No.
    # For 2,4,7: 56 / (2+4+7-2) = 56/11 = 5.09. This is a lower bound on rank? No, that's for something else.
    # The lower bound for R(2,4,7) is at least max(2*4, 4*7, 2*7) = 28.
    # So rank must be >= 28.
    
    # Let's try to construct a rank 40 decomposition.
    # We can split the 7 columns into 4 and 3.
    # Part 1: 2x4 * 4x4.
    # Part 2: 2x4 * 4x3.
    
    # For 2x4 * 4x4:
    # We can use the fact that 4x4 = 2x2 blocks.
    # M = [A, B]. N = [[C, D], [E, F]].
    # P = [AC+BE, AD+BF].
    # We need to compute AC, BE, AD, BF.
    # Instead of 4 separate Strassen (rank 7 each = 28), can we do better?
    # If we compute AC and AD using Strassen, we use A.
    # If we compute BE and BF using Strassen, we use B.
    # Total rank 14? 
    # Let's check.
    # Strassen for A*C and A*D:
    # We need A. We can "pre-compute" A's Strassen factors? No, Strassen factors depend on both A and B (the other matrix).
    # So we cannot share factors between A*C and A*D easily unless we restructure.
    
    # However, we can use the "Winograd" scheme or similar.
    
    # Let's try a simpler heuristic:
    # Start with Schoolbook (56).
    # Try to remove terms by solving linear systems.
    # This is what the current program does, but it's weak.
    
    # New Strategy: 
    # 1. Generate a random rank 50 decomposition (by perturbing schoolbook and projecting to integers/half-integers).
    # 2. Use a "Sparse Recovery" algorithm (like L1 minimization or greedy) to find a sparse solution.
    #    Since the tensor is fixed, we are looking for a sparse representation.
    #    But the coefficients must be exact.
    
    # Actually, there is a known result: R(2,4,7) = 40? 
    # Let's try to construct rank 40.
    # Split 7 into 4+3.
    # R(2,4,4) <= 20?
    # R(2,4,3) <= 20?
    
    # Let's try to build a rank 40 decomposition manually.
    # M (2x4), N (4x7).
    # Let's use the decomposition of (2x2)*(2x2) with rank 7.
    # We have 2x4 * 4x7.
    # Split M into M1, M2 (2x2).
    # Split N into N1 (4x4), N2 (4x3).
    
    # For M1*N1 (2x2 * 4x4 -> 2x4):
    # N1 = [N1a, N1b] (2x2 blocks).
    # M1*N1 = M1*N1a + M1*N1b.
    # M1*N1a is 2x2 * 2x2. Rank 7.
    # M1*N1b is 2x2 * 2x2. Rank 7.
    # Total 14 for M1*N1.
    
    # Similarly for M2*N1:
    # M2*N1a (rank 7) + M2*N1b (rank 7).
    # Total 14 for M2*N1.
    
    # So M*N1 (2x4 * 4x4) can be done in rank 14+14=28?
    # Wait, M1 and M2 are different.
    # The product is [M1, M2] * [[N1a, N1b], [N2a, N2b]]? No.
    # N1 is 4x4.
    # N1 = [[C, D], [E, F]].
    # M*N1 = [M1*C + M2*E, M1*D + M2*F].
    # Term 1: M1*C + M2*E.
    # Term 2: M1*D + M2*F.
    
    # To compute M1*C + M2*E:
    # We can use Strassen on (M1, C) and (M2, E).
    # This requires 7 + 7 = 14 multiplications.
    # To compute M1*D + M2*F:
    # This requires 7 + 7 = 14 multiplications.
    # Total 28 for the 4x4 part.
    
    # For the 4x3 part (N2):
    # N2 = [[G, H, I], [J, K, L]].
    # M*N2 = [M1*G + M2*J, M1*H + M2*K, M1*I + M2*L].
    # Each term is 2x2 * 2x1 + 2x2 * 2x1.
    # R(2,2,1) = 4.
    # So each term takes 4+4=8 rank.
    # 3 terms * 8 = 24.
    
    # Total rank = 28 + 24 = 52.
    # This is better than 56.
    
    # Can we do better?
    # For the 4x3 part, can we share Strassen factors?
    # No, the matrices G, H, I are different.
    
    # What if we split 7 into 2+2+3?
    # R(2,4,2) + R(2,4,2) + R(2,4,3).
    # R(2,4,2):
    # M (2x4) * N (4x2).
    # M = [A, B]. N = [[C, D], [E, F]].
    # P = [A*C + B*E, A*D + B*F].
    # 2x2 * 2x2 results.
    # Each result is 2x2.
    # We can compute A*C, B*E, A*D, B*F.
    # Using Strassen: 4 * 7 = 28?
    # But wait, R(2,4,2) is smaller.
    # Actually, R(2,4,2) <= 14?
    # If we use the fact that we only need 2 columns of output.
    
    # Let's try to construct the Rank 52 decomposition explicitly.
    
    def strassen_2x2(A, B):
        # Returns U, V, W for 2x2 * 2x2 -> 2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices:
        # A: [[A00, A01], [A10, A11]] -> 0,1,2,3
        # B: [[B00, B01], [B10, B11]] -> 0,1,2,3
        # C: [[C00, C01], [C10, C11]] -> 0,1,2,3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Coefficients for U (A factors), V (B factors), W (C factors)
        
        # Col 0: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1 # C00=1, C11=1? No.
        # Wait, W is the output factor.
        # The tensor equation is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,j,k}
        # For 2x2*2x2, the tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
        # Output C_{i,k}.
        # So W should map (k, i) to the coefficient in C_{i,k}.
        
        # Let's define the mapping carefully.
        # i in 0..1, j in 0..1, k in 0..1.
        # U[i,j,c], V[j,k,c], W[k,i,c].
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1. (Indices 0, 3)
        # V: B00=1, B11=1. (Indices 0, 3)
        # W: Contributes to C00 and C11.
        # C00 = M1 - M4 + M5 + M7. So W[0,0] += 1. (k=0, i=0)
        # C11 = M1 - M2 + M3 + M6. So W[1,1] += 1. (k=1, i=1)
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 1] = 1
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1. (Indices 2, 3)
        # V: B00=1. (Index 0)
        # W: Contributes to C10 and C11.
        # C10 = M2 + M4. So W[0, 1] += 1. (k=0, i=1)
        # C11 = M1 - M2 + M3 + M6. So W[1, 1] += -1.
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = 1; W[1, 1] = -1
        
        # M3: A00(B01-B11)
        # U: A00=1. (Index 0)
        # V: B01=1, B11=-1. (Indices 1, 3)
        # W: Contributes to C01 and C11.
        # C01 = M3 + M5. So W[1, 0] += 1. (k=1, i=0)
        # C11 = M1 - M2 + M3 + M6. So W[1, 1] += 1.
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 0] = 1; W[1, 1] = 1
        
        # M4: A11(B10-B00)
        # U: A11=1. (Index 3)
        # V: B10=1, B00=-1. (Indices 2, 0)
        # W: Contributes to C00 and C10.
        # C00 = M1 - M4 + M5 + M7. So W[0, 0] += -1.
        # C10 = M2 + M4. So W[0, 1] += 1.
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 0] = -1; W[0, 1] = 1
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1. (Indices 0, 1)
        # V: B11=1. (Index 3)
        # W: Contributes to C00 and C01.
        # C00 = M1 - M4 + M5 + M7. So W[0, 0] += 1.
        # C01 = M3 + M5. So W[1, 0] += 1.
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 0] = 1; W[1, 0] = 1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1. (Indices 2, 0)
        # V: B00=1, B01=1. (Indices 0, 1)
        # W: Contributes to C11.
        # C11 = M1 - M2 + M3 + M6. So W[1, 1] += 1.
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 1] = 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1. (Indices 1, 3)
        # V: B10=1, B11=1. (Indices 2, 3)
        # W: Contributes to C00.
        # C00 = M1 - M4 + M5 + M7. So W[0, 0] += 1.
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 0] = 1
        
        return U, V, W

    def schoolbook_2x2_2x1(A, B):
        # 2x2 * 2x1 -> 2x1
        # Rank 4.
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 0: A00 B0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 1: A01 B1
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Term 2: A10 B0
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Term 3: A11 B1
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Construct the full decomposition
    # M (2x4) -> [M1 (2x2), M2 (2x2)]
    # N (4x7) -> [N1 (4x4), N2 (4x3)]
    
    # N1 = [[C, D], [E, F]]
    # N2 = [[G, H, I], [J, K, L]]
    
    # P = [M1*C + M2*E, M1*D + M2*F, M1*G + M2*J, M1*H + M2*K, M1*I + M2*L]
    
    # We will assemble U, V, W for the full 2x4 * 4x7 problem.
    # U shape (8, rank), V shape (28, rank), W shape (14, rank)
    
    # Rank calculation:
    # M1*C (2x2*2x2) -> 7
    # M2*E (2x2*2x2) -> 7
    # M1*D (2x2*2x2) -> 7
    # M2*F (2x2*2x2) -> 7
    # M1*G (2x2*2x1) -> 4
    # M2*J (2x2*2x1) -> 4
    # M1*H (2x2*2x1) -> 4
    # M2*K (2x2*2x1) -> 4
    # M1*I (2x2*2x1) -> 4
    # M2*L (2x2*2x1) -> 4
    
    # Total rank = 4*7 + 6*4 = 28 + 24 = 52.
    
    U_full = np.zeros((8, 52))
    V_full = np.zeros((28, 52))
    W_full = np.zeros((14, 52))
    
    col_offset = 0
    
    # Helper to add a sub-decomposition
    def add_decomp(U_sub, V_sub, W_sub, u_offset, v_offset, w_offset, col_start):
        U_full[u_offset:u_offset+U_sub.shape[0], col_start:col_start+U_sub.shape[1]] = U_sub
        V_full[v_offset:v_offset+V_sub.shape[0], col_start:col_start+V_sub.shape[1]] = V_sub
        W_full[w_offset:w_offset+W_sub.shape[0], col_start:col_start+W_sub.shape[1]] = W_sub
        return col_start + U_sub.shape[1]

    # M1 is rows 0,1 of M. M2 is rows 2,3 of M.
    # N1 is cols 0,1,2,3 of N. N2 is cols 4,5,6 of N.
    # N1 blocks:
    # C: N[0:2, 0:2]
    # D: N[0:2, 2:4]
    # E: N[2:4, 0:2]
    # F: N[2:4, 2:4]
    
    # N2 blocks:
    # G: N[0:2, 4:5]
    # H: N[0:2, 5:6]
    # I: N[0:2, 6:7]
    # J: N[2:4, 4:5]
    # K: N[2:4, 5:6]
    # L: N[2:4, 6:7]
    
    # Output P is 2x7.
    # P rows 0,1.
    # P cols 0,1 come from M1*C + M2*E.
    # P cols 2,3 come from M1*D + M2*F.
    # P cols 4,5,6 come from M1*G + M2*J, etc.
    
    # 1. M1*C
    U_sc, V_sc, W_sc = strassen_2x2(None, None)
    # M1 is U rows 0,1. C is V cols 0,1.
    # U_offset 0, V_offset 0, W_offset 0.
    col_offset = add_decomp(U_sc, V_sc, W_sc, 0, 0, 0, col_offset)
    
    # 2. M2*E
    # M2 is U rows 2,3. E is V cols 2,3 (rows 2,3 of N).
    # W_offset 0 (same output cols).
    col_offset = add_decomp(U_sc, V_sc, W_sc, 2, 2, 0, col_offset)
    
    # 3. M1*D
    # M1 is U rows 0,1. D is V cols 4,5 (rows 0,1 of N, cols 2,3).
    # W_offset 2 (output cols 2,3).
    col_offset = add_decomp(U_sc, V_sc, W_sc, 0, 4, 2, col_offset)
    
    # 4. M2*F
    # M2 is U rows 2,3. F is V cols 6,7 (rows 2,3 of N, cols 2,3).
    # W_offset 2.
    col_offset = add_decomp(U_sc, V_sc, W_sc, 2, 6, 2, col_offset)
    
    # 5. M1*G
    U_sg, V_sg, W_sg = schoolbook_2x2_2x1(None, None)
    # M1 is U rows 0,1. G is V cols 8,9 (rows 0,1 of N, col 4).
    # W_offset 4 (output col 4).
    col_offset = add_decomp(U_sg, V_sg, W_sg, 0, 8, 4, col_offset)
    
    # 6. M2*J
    # M2 is U rows 2,3. J is V cols 10,11 (rows 2,3 of N, col 4).
    # W_offset 4.
    col_offset = add_decomp(U_sg, V_sg, W_sg, 2, 10, 4, col_offset)
    
    # 7. M1*H
    # M1 is U rows 0,1. H is V cols 12,13 (rows 0,1 of N, col 5).
    # W_offset 5 (output col 5).
    col_offset = add_decomp(U_sg, V_sg, W_sg, 0, 12, 5, col_offset)
    
    # 8. M2*K
    # M2 is U rows 2,3. K is V cols 14,15 (rows 2,3 of N, col 5).
    # W_offset 5.
    col_offset = add_decomp(U_sg, V_sg, W_sg, 2, 14, 5, col_offset)
    
    # 9. M1*I
    # M1 is U rows 0,1. I is V cols 16,17 (rows 0,1 of N, col 6).
    # W_offset 6 (output col 6).
    col_offset = add_decomp(U_sg, V_sg, W_sg, 0, 16, 6, col_offset)
    
    # 10. M2*L
    # M2 is U rows 2,3. L is V cols 18,19 (rows 2,3 of N, col 6).
    # W_offset 6.
    col_offset = add_decomp(U_sg, V_sg, W_sg, 2, 18, 6, col_offset)
    
    # Verify the decomposition
    # Tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # i in 0..1, j in 0..3, k in 0..6.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(52):
                    val += U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook if construction failed
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
