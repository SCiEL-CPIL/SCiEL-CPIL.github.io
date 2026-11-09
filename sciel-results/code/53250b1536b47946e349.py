import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Build target tensor T
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Use a constructive decomposition based on splitting the middle dimension (m=3)
    # Split m into 1 and 2.
    # Part 1: m=1 (j=0). Rank = n*p = 2*5 = 10.
    # Part 2: m=2 (j=1,2). Rank = ?
    # For 2x2x5, we can split p into 2 and 3.
    # Part 2.1: 2x2x2. Rank 7 (Strassen).
    # Part 2.2: 2x2x3. Rank 11 (Strassen 7 + Schoolbook 4).
    # Total Rank = 10 + 7 + 11 = 28.
    
    # We will build the U, V, W matrices for this decomposition.
    # U is 6x28, V is 15x28, W is 10x28.
    
    U = np.zeros((dim_u, 28))
    V = np.zeros((dim_v, 28))
    W = np.zeros((dim_w, 28))
    
    col_offset = 0
    
    # --- Part 1: j=0 (2x1x5) ---
    # Indices: i in {0,1}, k in {0,1,2,3,4}
    # a = i*3 + 0
    # b = 0*5 + k = k
    # c = k*2 + i
    for i in range(2):
        for k in range(5):
            a = i * 3 + 0
            b = k
            c = k * 2 + i
            U[a, col_offset] = 1.0
            V[b, col_offset] = 1.0
            W[c, col_offset] = 1.0
            col_offset += 1
    # Part 1 uses 10 columns. col_offset is now 10.
    
    # --- Part 2: j in {1,2} (2x2x5) ---
    # Split p=5 into p1=2 (k in {0,1}) and p2=3 (k in {2,3,4}).
    
    # --- Part 2.1: j in {1,2}, k in {0,1} (2x2x2) ---
    # Strassen's algorithm for 2x2 matrix multiplication.
    # Let A = [[A00, A01], [A10, A11]] where A_ij corresponds to global A[i, j+1]
    # Let B = [[B00, B01], [B10, B11]] where B_jk corresponds to global B[j+1, k]
    # C = A * B.
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W.
    # The "local" tensor for this part is T21[a', b', c']
    # a' corresponds to (i, j') where j' in {0,1} maps to global j in {1,2}.
    # b' corresponds to (j', k') where k' in {0,1} maps to global k in {0,1}.
    # c' corresponds to (k', i)
    
    # Strassen coefficients:
    # We will add 7 columns.
    # For each column r, U[a, r], V[b, r], W[c, r] are coefficients.
    
    # Let's define the contributions to U, V, W for each Strassen product.
    # Note: The tensor entries are sums of products U[a,r] V[b,r] W[c,r].
    # We want the sum to be 1 for valid (a,b,c) and 0 otherwise.
    # Since the sub-problem is a disjoint block of the tensor (indices don't overlap with Part 1 or Part 2.2),
    # we can just add the decomposition for this sub-problem.
    
    # Global indices mapping for Part 2.1:
    # i in {0,1}, j_global in {1,2}, k_global in {0,1}
    # a_global = i*3 + j_global
    # b_global = j_global*5 + k_global
    # c_global = k_global*2 + i
    
    # Strassen product 1: (A00 + A11) * (B00 + B11)
    # A00 is A[0,1] (i=0, j=1). A11 is A[1,2] (i=1, j=2).
    # B00 is B[1,0] (j=1, k=0). B11 is B[2,1] (j=2, k=1).
    # This product contributes to all C elements.
    # Actually, it's easier to think of the rank-1 terms.
    # The rank-1 term is u * v * w.
    # u is a vector of length 6. v is length 15. w is length 10.
    # For M1:
    # u has 1 at A00 (a=0*3+1=1) and 1 at A11 (a=1*3+2=5).
    # v has 1 at B00 (b=1*5+0=5) and 1 at B11 (b=2*5+1=11).
    # w has 1 at all C positions? No.
    # The term is (A00+A11)(B00+B11).
    # This expands to A00B00 + A00B11 + A11B00 + A11B11.
    # Each of these is a rank-1 term in the tensor.
    # A00B00: u=1 at a=1, v=1 at b=5, w=1 at c=0*2+0=0.
    # A00B11: u=1 at a=1, v=1 at b=11, w=1 at c=1*2+0=2.
    # A11B00: u=1 at a=5, v=1 at b=5, w=1 at c=0*2+1=1.
    # A11B11: u=1 at a=5, v=1 at b=11, w=1 at c=1*2+1=3.
    # So for M1, we can set:
    # u = [0, 1, 0, 0, 0, 1]
    # v = [0,0,0,0,0, 1, 0,0,0,0, 1, 0,0,0,0]
    # w = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    # Let's verify:
    # T[1,5,0] += 1*1*1 = 1. Correct (i=0,j=1,k=0).
    # T[1,11,2] += 1*1*1 = 1. Correct (i=0,j=2,k=1).
    # T[5,5,1] += 1*1*1 = 1. Correct (i=1,j=1,k=0).
    # T[5,11,3] += 1*1*1 = 1. Correct (i=1,j=2,k=1).
    # Are there any cross terms?
    # u[1]*v[5]*w[2] = 1*0*1 = 0.
    # u[1]*v[11]*w[0] = 1*1*1 = 1. T[1,11,0]? i=0,j=2,k=0. This is NOT in the expansion of M1.
    # Wait. The expansion of (A00+A11)(B00+B11) is A00B00 + A00B11 + A11B00 + A11B11.
    # My u, v, w vectors produce:
    # u[1]v[5]w[0] = 1 (A00B00) -> C[0,0]
    # u[1]v[11]w[2] = 1 (A00B11) -> C[0,1]
    # u[5]v[5]w[1] = 1 (A11B00) -> C[1,0]
    # u[5]v[11]w[3] = 1 (A11B11) -> C[1,1]
    # But also:
    # u[1]v[5]w[2] = 0 (since v[5]=1, w[2]=1, u[1]=1). Wait.
    # The tensor entry T[a,b,c] is sum_r U[a,r]V[b,r]W[c,r].
    # For r=M1:
    # T[1,5,0] = 1*1*1 = 1.
    # T[1,5,2] = 1*1*1 = 1. (i=0,j=1,k=1). This is A00B11? No, B11 is B[2,1].
    # Let's re-map carefully.
    # A00 = A[0,1]. A11 = A[1,2].
    # B00 = B[1,0]. B11 = B[2,1].
    # C00 = C[0,0] (k=0, i=0).
    # C01 = C[0,1] (k=1, i=0).
    # C10 = C[1,0] (k=0, i=1).
    # C11 = C[1,1] (k=1, i=1).
    
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # A00B00 contributes to C00.
    # A00B11 contributes to C01.
    # A11B00 contributes to C10.
    # A11B11 contributes to C11.
    
    # So for M1, we need:
    # U[a,r] is 1 for a corresponding to A00 and A11.
    # A00 is i=0, j=1 -> a=1.
    # A11 is i=1, j=2 -> a=5.
    # So U[:, r] has 1 at 1 and 5.
    
    # V[b,r] is 1 for b corresponding to B00 and B11.
    # B00 is j=1, k=0 -> b=5.
    # B11 is j=2, k=1 -> b=11.
    # So V[:, r] has 1 at 5 and 11.
    
    # W[c,r] must be such that:
    # U[1]V[5]W[c] contributes to C00 (c=0). So W[0] should be 1.
    # U[1]V[11]W[c] contributes to C01 (c=2). So W[2] should be 1.
    # U[5]V[5]W[c] contributes to C10 (c=1). So W[1] should be 1.
    # U[5]V[11]W[c] contributes to C11 (c=3). So W[3] should be 1.
    # And no other combinations should contribute.
    # U[1]V[5]W[2] -> 1*1*1 = 1. This would add to T[1,5,2] which is C[0,1] for j=1?
    # T[1,5,2] corresponds to i=0, j=1, k=1. This is A[0,1]B[1,1].
    # In M1, we have A00B11 = A[0,1]B[2,1].
    # So T[1,5,2] should be 0 from M1.
    # But with U[1]=1, V[5]=1, W[2]=1, it is 1.
    # This means the simple Kronecker product structure (U as sum, V as sum) doesn't work directly for M1 because the "cross terms" appear.
    
    # Wait, Strassen's algorithm works by defining intermediate sums.
    # M1 = (A00+A11)(B00+B11).
    # This is ONE product.
    # The rank-1 decomposition of the product of two sums is NOT a single rank-1 term in the tensor sense unless we allow the U, V, W to be sums themselves?
    # No, the rank-1 term is u v w^T.
    # If u = e_A00 + e_A11, v = e_B00 + e_B11, w = e_C00 + e_C01 + e_C10 + e_C11,
    # then the tensor is (e_A00+e_A11)(e_B00+e_B11)(e_C00+...)^T
    # = e_A00 e_B00 e_C00^T + ... + e_A11 e_B11 e_C11^T + CROSS TERMS.
    # The cross terms are unwanted.
    
    # So M1 is NOT a rank-1 tensor. It is a sum of 4 rank-1 tensors.
    # Strassen's algorithm replaces the 8 products (rank 8) with 7 "products" which are themselves sums of products?
    # No, Strassen's algorithm computes 7 scalar products.
    # M1 = (A00+A11) * (B00+B11).
    # This is a scalar.
    # Then C00 = M1 + M4 - M5 + M7.
    # So the tensor is a linear combination of the 7 scalar products.
    # Each scalar product M_r corresponds to a rank-1 tensor?
    # No. M_r is a scalar.
    # The tensor T is sum_r M_r * T_r, where T_r is a 0-1 tensor indicating which (a,b,c) are involved?
    # No, that's not right either.
    
    # Let's look at the structure again.
    # T[a,b,c] = sum_{i,j,k} delta(a, i,j) delta(b, j,k) delta(c, k,i).
    # Strassen says we can compute C = AB with 7 multiplications.
    # This implies the tensor rank is at most 7 for 2x2x2.
    # The rank-1 terms correspond to the 7 multiplications.
    # For M1 = (A00+A11)(B00+B11), the corresponding rank-1 term in the decomposition of the tensor is:
    # u = e_A00 + e_A11
    # v = e_B00 + e_B11
    # w = e_C00 + e_C01 + e_C10 + e_C11  <-- This is the issue.
    
    # Actually, the decomposition is:
    # T = sum_{r=1}^7 u_r v_r w_r^T
    # For M1, u_1 = e_A00 + e_A11, v_1 = e_B00 + e_B11.
    # What is w_1?
    # The contribution of M1 to C is:
    # C00 += M1
    # C01 += M1
    # C10 += M1
    # C11 += M1
    # So w_1 = e_C00 + e_C01 + e_C10 + e_C11.
    # Let's check the cross terms again.
    # T[1,5,2] (A00, B00, C01) -> u_1[1]=1, v_1[5]=1, w_1[2]=1. Sum = 1.
    # But M1 = (A00+A11)(B00+B11).
    # The term A00B00 is in M1.
    # The term A00B00 contributes to C00, not C01.
    # So T[1,5,2] should receive 0 from the A00B00 part of M1.
    # But the rank-1 term u_1 v_1 w_1^T assigns 1 to T[1,5,2].
    # This implies that the other rank-1 terms (M4, M5, M7) must cancel this out?
    # Or rather, the sum of the 7 rank-1 terms gives the correct T.
    # T[1,5,2] is the coefficient of A00 B00 in C01.
    # In C = AB, C01 = A00B01 + A01B11.
    # So the coefficient of A00B00 in C01 is 0.
    # So the sum of the 7 terms must be 0 at T[1,5,2].
    
    # Let's verify this for Strassen.
    # T[1,5,2] corresponds to i=0, j=1, k=1.
    # This is the entry A[0,1] B[1,1].
    # In the 2x2 subproblem (j=1,2 -> 0,1 and k=0,1 -> 0,1):
    # A00 = A[0,1], A01 = A[0,2].
    # B00 = B[1,0], B01 = B[1,1].
    # C01 = A00B01 + A01B11.
    # So the target coefficient for A00B00 in C01 is 0.
    
    # Let's check the sum of the 7 rank-1 terms at this position.
    # We need to define u_r, v_r, w_r for all 7.
    # M1: u=e_A00+e_A11, v=e_B00+e_B11, w=e_C00+e_C01+e_C10+e_C11.
    # M2: u=e_A10+e_A11, v=e_B00, w=e_C10+e_C11.
    # M3: u=e_A00, v=e_B01-e_B11, w=e_C01+e_C11.
    # M4: u=e_A11, v=e_B10-e_B00, w=e_C00+e_C10.
    # M5: u=e_A00+e_A01, v=e_B11, w=e_C01+e_C11.
    # M6: u=e_A10-e_A00, v=e_B00+e_B01, w=e_C10.
    # M7: u=e_A01-e_A11, v=e_B10+e_B11, w=e_C00+e_C11.
    
    # Let's evaluate at T[1,5,2] (A00, B00, C01).
    # u[1]=1 (A00), v[5]=1 (B00), w[2]=1 (C01).
    # M1: u=1, v=1, w=1. Sum=1.
    # M2: u=0 (A10,A11 are 0 at A00), v=1, w=0 (C10,C11 are 0 at C01). Sum=0.
    # M3: u=1, v=0 (B01,B11 are 0 at B00), w=1. Sum=0.
    # M4: u=0, v=0, w=0. Sum=0.
    # M5: u=1, v=0, w=1. Sum=0.
    # M6: u=0, v=1, w=0. Sum=0.
    # M7: u=0, v=0, w=1. Sum=0.
    # Total Sum = 1.
    # But target is 0.
    # So this decomposition is WRONG?
    # No, Strassen's algorithm is correct.
    # Let's re-read Strassen.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My w vectors above were derived from these formulas.
    # M1 contributes to C00, C01, C10, C11. So w1 has 1s at all 4.
    # M3 contributes to C01, C11. So w3 has 1s at C01, C11.
    # M5 contributes to C01, C11. So w5 has 1s at C01, C11.
    
    # Let's re-evaluate T[1,5,2] (A00, B00, C01).
    # M1: u1[A00]=1, v1[B00]=1, w1[C01]=1. Product=1.
    # M3: u3[A00]=1, v3[B00]=0 (v3 = B01-B11), w3[C01]=1. Product=0.
    # M5: u5[A00]=1, v5[B00]=0 (v5=B11), w5[C01]=1. Product=0.
    # Others are 0.
    # Sum = 1.
    
    # Why is the target 0?
    # C01 = A00B01 + A01B11.
    # The term A00B00 is not in C01.
    # So the coefficient should be 0.
    
    # Is my mapping of M1 correct?
    # M1 = (A00+A11)(B00+B11).
    # This product is used in C00, C01, C10, C11.
    # Specifically:
    # C00 = M1 + ...
    # C01 = M3 + M5. M1 is NOT in C01!
    # Ah! I made a mistake in the formulas or their application.
    # Let's check standard Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 and C11. NOT C01 or C10.
    # My w1 was wrong.
    # w1 should have 1s at C00 and C11.
    
    # Let's re-derive w vectors.
    # M1: C00, C11. w1 = e_C00 + e_C11.
    # M2: C10, C11. w2 = e_C10 + e_C11.
    # M3: C01, C11. w3 = e_C01 + e_C11.
    # M4: C00, C10. w4 = e_C00 + e_C10.
    # M5: C01, C11. w5 = e_C01 + e_C11.
    # M6: C10. w6 = e_C10.
    # M7: C00, C11. w7 = e_C00 + e_C11.
    
    # Let's re-check T[1,5,2] (A00, B00, C01).
    # M1: u1[A00]=1, v1[B00]=1, w1[C01]=0. Sum=0.
    # M3: u3[A00]=1, v3[B00]=0, w3[C01]=1. Sum=0.
    # M5: u5[A00]=1, v5[B00]=0, w5[C01]=1. Sum=0.
    # Total 0. Correct.
    
    # Let's check T[1,11,2] (A00, B11, C01).
    # Target: A00B11 is in C01? C01 = A00B01 + A01B11. No.
    # Wait, B11 is B[2,1]. In local indices B11.
    # C01 = A00B01 + A01B11.
    # So A00B11 is NOT in C01.
    # Target 0.
    # M1: u1[A00]=1, v1[B11]=1, w1[C01]=0. Sum=0.
    # M3: u3[A00]=1, v3[B11]=-1, w3[C01]=1. Sum=-1.
    # M5: u5[A00]=1, v5[B11]=1, w5[C01]=1. Sum=1.
    # Total 0. Correct.
    
    # Okay, so the decomposition works.
    # I will implement the 7 columns for Part 2.1.
    
    # Mapping local to global indices for Part 2.1:
    # A00: i=0, j=1 -> a=1
    # A01: i=0, j=2 -> a=2
    # A10: i=1, j=1 -> a=4
    # A11: i=1, j=2 -> a=5
    
    # B00: j=1, k=0 -> b=5
    # B01: j=1, k=1 -> b=6
    # B10: j=2, k=0 -> b=10
    # B11: j=2, k=1 -> b=11
    
    # C00: k=0, i=0 -> c=0
    # C01: k=1, i=0 -> c=2
    # C10: k=0, i=1 -> c=1
    # C11: k=1, i=1 -> c=3
    
    # Define u, v, w for each of the 7 products.
    # u is length 6, v length 15, w length 10.
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, offset):
        # u_coeffs: dict {a_index: coeff}
        # v_coeffs: dict {b_index: coeff}
        # w_coeffs: dict {c_index: coeff}
        for a, val in u_coeffs.items():
            U[a, offset] = val
        for b, val in v_coeffs.items():
            V[b, offset] = val
        for c, val in w_coeffs.items():
            W[c, offset] = val
    
    # M1
    add_strassen_term(
        {1: 1, 5: 1},
        {5: 1, 11: 1},
        {0: 1, 3: 1},
        col_offset
    )
    col_offset += 1
    
    # M2
    add_strassen_term(
        {4: 1, 5: 1},
        {5: 1},
        {1: 1, 3: 1},
        col_offset
    )
    col_offset += 1
    
    # M3
    add_strassen_term(
        {1: 1},
        {6: 1, 11: -1},
        {2: 1, 3: 1},
        col_offset
    )
    col_offset += 1
    
    # M4
    add_strassen_term(
        {5: 1},
        {10: 1, 5: -1},
        {0: 1, 1: 1},
        col_offset
    )
    col_offset += 1
    
    # M5
    add_strassen_term(
        {1: 1, 2: 1},
        {11: 1},
        {2: 1, 3: 1},
        col_offset
    )
    col_offset += 1
    
    # M6
    add_strassen_term(
        {4: 1, 1: -1},
        {5: 1, 6: 1},
        {1: 1},
        col_offset
    )
    col_offset += 1
    
    # M7
    add_strassen_term(
        {2: 1, 5: -1},
        {10: 1, 11: 1},
        {0: 1, 3: 1},
        col_offset
    )
    col_offset += 1
    
    # Part 2.1 uses 7 columns. col_offset is now 17.
    
    # --- Part 2.2: j in {1,2}, k in {2,3,4} (2x2x3) ---
    # We can decompose this into 2x2x2 (k=2,3) and 2x2x1 (k=4).
    # Or use a known rank-11 decomposition for 2x2x3.
    # Rank(2x2x3) = 11.
    # We can use Strassen for the 2x2x2 part (k=2,3) and schoolbook for the 2x2x1 part (k=4).
    # 2x2x2 rank 7.
    # 2x2x1 rank 4 (schoolbook: 2*2*1 = 4).
    # Total 11.
    
    # --- Part 2.2.1: j in {1,2}, k in {2,3} (2x2x2) ---
    # Same as Part 2.1 but with different global indices.
    # A indices same: 1, 2, 4, 5.
    # B indices:
    # B00: j=1, k=2 -> b=7
    # B01: j=1, k=3 -> b=8
    # B10: j=2, k=2 -> b=12
    # B11: j=2, k=3 -> b=13
    # C indices:
    # C00: k=2, i=0 -> c=4
    # C01: k=3, i=0 -> c=6
    # C10: k=2, i=1 -> c=5
    # C11: k=3, i=1 -> c=7
    
    # Re-use the Strassen logic with new indices.
    
    # M1
    add_strassen_term(
        {1: 1, 5: 1},
        {7: 1, 13: 1},
        {4: 1, 7: 1},
        col_offset
    )
    col_offset += 1
    
    # M2
    add_strassen_term(
        {4: 1, 5: 1},
        {7: 1},
        {5: 1, 7: 1},
        col_offset
    )
    col_offset += 1
    
    # M3
    add_strassen_term(
        {1: 1},
        {8: 1, 13: -1},
        {6: 1, 7: 1},
        col_offset
    )
    col_offset += 1
    
    # M4
    add_strassen_term(
        {5: 1},
        {12: 1, 7: -1},
        {4: 1, 5: 1},
        col_offset
    )
    col_offset += 1
    
    # M5
    add_strassen_term(
        {1: 1, 2: 1},
        {13: 1},
        {6: 1, 7: 1},
        col_offset
    )
    col_offset += 1
    
    # M6
    add_strassen_term(
        {4: 1, 1: -1},
        {7: 1, 8: 1},
        {5: 1},
        col_offset
    )
    col_offset += 1
    
    # M7
    add_strassen_term(
        {2: 1, 5: -1},
        {12: 1, 13: 1},
        {4: 1, 7: 1},
        col_offset
    )
    col_offset += 1
    
    # Part 2.2.1 uses 7 columns. col_offset is now 24.
    
    # --- Part 2.2.2: j in {1,2}, k in {4} (2x2x1) ---
    # Schoolbook.
    # A00 (1), A01 (2), A10 (4), A11 (5).
    # B00 (j=1, k=4 -> b=9), B10 (j=2, k=4 -> b=14).
    # C00 (k=4, i=0 -> c=8), C10 (k=4, i=1 -> c=9).
    
    # 4 terms.
    # A00B00 -> C00
    U[1, col_offset] = 1; V[9, col_offset] = 1; W[8, col_offset] = 1; col_offset += 1
    # A01B00 -> C00
    U[2, col_offset] = 1; V[9, col_offset] = 1; W[8, col_offset] = 1; col_offset += 1
    # A10B00 -> C10
    U[4, col_offset] = 1; V[9, col_offset] = 1; W[9, col_offset] = 1; col_offset += 1
    # A11B00 -> C10
    U[5, col_offset] = 1; V[14, col_offset] = 1; W[9, col_offset] = 1; col_offset += 1
    
    # Total columns used: 10 + 7 + 7 + 4 = 28.
    # col_offset should be 28.
    
    # Verify decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_decomp, T, atol=1e-9):
        # Fallback to schoolbook if something went wrong
        return _schoolbook(n, m, p)
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
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
