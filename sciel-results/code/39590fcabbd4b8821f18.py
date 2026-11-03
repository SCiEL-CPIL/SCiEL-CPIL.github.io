"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    start_time = time.time()
    deadline = start_time + 110

    # WFTA Strategy for 2x4 * 4x6
    # Dimensions: n=2, m=4, p=6
    # We can view the multiplication as a convolution-like operation if we pad or structure the indices correctly.
    # However, direct WFTA on rectangular matrices requires careful handling of the group structure.
    
    # Alternative: Use a known low-rank decomposition for 2x4 * 4x6.
    # Rank 48 is trivial.
    # Rank 42 is Strassen-based (2 blocks of 2x2 * 2x3? No, 2x4 * 4x6 splits into 2x2 * 2x3 blocks? 
    # Actually, m=4 splits into 2+2. p=6 splits into 3+3 or 2+2+2.
    # If we split m=4 into 2+2 and p=6 into 3+3, we have 2x2 * 2x3 blocks.
    # Rank of 2x2 * 2x3 is 7 (Strassen-like).
    # 2 (n-split? no n=2) * 2 (p-split) = 4 blocks?
    # Let's check the structure:
    # A is 2x4. B is 4x6.
    # Split B's columns into two 3-col blocks: B1 (4x3), B2 (4x3).
    # C1 = A * B1 (2x3), C2 = A * B2 (2x3).
    # Rank of 2x4 * 4x3?
    # Split m=4 into 2+2.
    # A = [A1, A2] (2x2 each).
    # B1 = [B11; B12] (2x3 each).
    # C1 = A1*B11 + A2*B12.
    # This is a 2x2 * 2x3 multiplication.
    # Minimal rank for 2x2 * 2x3 is 7.
    # So C1 costs 7. C2 costs 7. Total 14?
    # Wait, A1 and A2 are shared? No, A is fixed.
    # The tensor is T_{i,j,k}.
    # If we split m into m1=2, m2=2.
    # C_{i,k} = sum_{j in m1} A_{i,j} B_{j,k} + sum_{j in m2} A_{i,j} B_{j,k}.
    # This is just a sum of two tensors.
    # Rank(T1 + T2) <= Rank(T1) + Rank(T2).
    # T1 is 2x2 * 2x6. Rank of 2x2 * 2x6?
    # Split p=6 into 3+3.
    # 2x2 * 2x3 has rank 7.
    # So 2x2 * 2x6 has rank 14?
    # Then total rank 14 + 14 = 28?
    # Let's verify if 2x2 * 2x6 can be done in less than 14.
    # 2x2 * 2x6 is equivalent to two 2x2 * 2x3 multiplications.
    # Since the A part is shared (2x2), maybe we can share the Strassen factors?
    # Strassen for 2x2 * 2x3:
    # M1 = (a00+a11)(b00+b11) -> contributes to C00, C01? No.
    # Let's look up rank of 2x2 * 2x3. It is 7.
    # Decomposition:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C02 = M6
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C12 = M7
    # Wait, standard Strassen is 2x2 * 2x2 (7 mults).
    # For 2x2 * 2x3, we can use 7 mults?
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # This covers 2x2 * 2x2.
    # For the 3rd column (b02, b12), we need:
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    # This requires 4 more mults? Or can we reuse?
    # Usually, rank(2x2 * 2x3) = 7 + 4 = 11? Or is it 7?
    # Actually, rank(2x2 * 2x3) is known to be 7? No, that's 2x2*2x2.
    # Rank(2x2 * 2x3) is 7? Let's check.
    # If we use 7 mults for the first 2 cols, we have 4 entries left.
    # C02, C12 depend on b02, b12.
    # We can add:
    # M8 = a00*b02
    # M9 = a11*b12
    # M10 = a01*b12
    # M11 = a10*b02
    # This adds 4. Total 11.
    # Can we do better?
    # Rank(2x2 * 2x3) is actually 7? No.
    # It is 7 for 2x2*2x2.
    # For 2x2*2x3, the lower bound is max(4, 6, 6) = 6?
    # Upper bound 12 (trivial).
    # Known rank is 7? No.
    # Let's assume rank 7 is not possible for 2x3.
    # However, we can treat the whole 2x4 * 4x6 as a single tensor.
    
    # Let's try a different split:
    # m=4 -> 2+2.
    # p=6 -> 2+2+2.
    # 3 blocks of 2x2 * 2x2.
    # Each 2x2 * 2x2 has rank 7.
    # Total 21?
    # But A is 2x4. A = [A1, A2].
    # B = [B1; B2] where B1 is 2x6, B2 is 2x6.
    # C = A1*B1 + A2*B2.
    # This is a sum of two 2x2 * 2x6 multiplications.
    # Rank(2x2 * 2x6)?
    # Split p=6 into 2+2+2.
    # 2x2 * 2x2 (rank 7) + 2x2 * 2x2 (rank 7) + 2x2 * 2x2 (rank 7) = 21?
    # But the A factors are the same for all 3 blocks!
    # A1 is 2x2.
    # B11, B12, B13 are 2x2.
    # C1 = A1*B11, C2 = A1*B12, C3 = A1*B13.
    # This is 3 separate 2x2*2x2 multiplications with the same A.
    # Can we share the Strassen factors for A?
    # Strassen factors for A:
    # a1 = a00+a11
    # a2 = a10+a11
    # a3 = a00
    # a4 = a11
    # a5 = a00+a01
    # a6 = a10-a00
    # a7 = a01-a11
    # These are 7 linear forms of A.
    # For each B block, we need 7 linear forms of B.
    # So for 3 B blocks, we need 3*7 = 21 linear forms of B.
    # And 7 linear forms of A.
    # Total rank 21?
    # Let's verify.
    # For block 1 (cols 0,1):
    # M1 = a1 * b1_1
    # M2 = a2 * b1_2
    # ...
    # For block 2 (cols 2,3):
    # M8 = a1 * b2_1
    # ...
    # Yes, the A factors are shared.
    # So rank = 7 (for A) + 7*3 (for B) ? No.
    # Rank is the number of rank-1 terms.
    # Each term is u_r * v_r * w_r.
    # u_r is from A. v_r is from B. w_r is for C.
    # For block 1, we have 7 terms.
    # For block 2, we have 7 terms.
    # For block 3, we have 7 terms.
    # Total 21 terms.
    # The u vectors are reused?
    # Term 1: u1, v1, w1.
    # Term 8: u1, v8, w8.
    # Yes, u1 is the same.
    # So the total rank is 21?
    # Let's check if this works.
    # C = sum_{r=1}^{21} u_r(A) v_r(B) w_r.
    # This decomposes the tensor.
    # Is 21 the minimal rank?
    # Lower bound: max(n*m, m*p, n*p) = max(8, 24, 12) = 24.
    # So 21 is impossible. The lower bound is 24.
    # My calculation of "sharing A factors" reduces the number of *distinct* A forms, but each term still requires a specific pairing.
    # Rank is the number of terms.
    # If I use 21 terms, the rank is 21.
    # But the lower bound is 24.
    # So 21 is impossible.
    # Why? Because the lower bound is tight?
    # Rank(2x2 * 2x2) = 7.
    # Rank(2x2 * 2x6) ?
    # If I split 2x6 into three 2x2, I get 3*7=21 terms.
    # But the lower bound for 2x2 * 2x6 is max(4, 12, 12) = 12?
    # No, lower bound is max(n*m, m*p, n*p) = max(4, 12, 12) = 12.
    # So 21 is possible.
    # But for 2x4 * 4x6, lower bound is 24.
    # So 21 is impossible for the whole matrix.
    # The error is in assuming the blocks are independent.
    # They are not, because the indices are coupled in the tensor definition?
    # No, matrix multiplication is a sum of products.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # If we split j and k, the terms are additive.
    # T = T_{j<2, k<2} + T_{j<2, 2<=k<4} + ...
    # Rank(T) <= sum Rank(T_part).
    # If we split m=4 into 2+2 and p=6 into 2+2+2.
    # 6 parts.
    # Each part is 2x2 * 2x2 (A block 2x2, B block 2x2).
    # Rank of each part is 7.
    # Total rank 42.
    # This matches the current program.
    
    # Can we do better?
    # Maybe use a different split.
    # m=4 -> 1+3?
    # p=6 -> 3+3?
    # 2x1 * 1x3 (rank 3) + 2x3 * 3x3 (rank ?)
    # Rank(2x3 * 3x3) ?
    # Lower bound max(6, 9, 6) = 9.
    # Upper bound 18.
    # Known rank?
    # If we use Strassen for 2x3 * 3x3?
    # This is getting complex.
    
    # Let's stick to the known rank 42 from the current program, but verify if it's correct.
    # The current program implements rank 42.
    # The lower bound is 24.
    # Is there a known rank between 24 and 42?
    # For 2x4 * 4x6, the rank is likely 30-40.
    # However, finding it is hard.
    
    # Let's try to implement the "Shared A" strategy for m=4, p=6.
    # Split m=4 into 2+2.
    # Split p=6 into 2+2+2.
    # 3 blocks of 2x2 * 2x2.
    # Rank 21?
    # Wait, the lower bound for 2x4 * 4x6 is 24.
    # So 21 is impossible.
    # Why is my block decomposition giving 21?
    # Because I am reusing the A factors.
    # But the tensor rank is not just the sum of ranks if the sub-tensors are not orthogonal in the support?
    # They are disjoint in the (j,k) indices.
    # So the tensor is a direct sum.
    # Rank(T1 + T2) = Rank(T1) + Rank(T2) if supports are disjoint.
    # So if I split into 6 disjoint 2x2*2x2 blocks, rank is 6*7 = 42.
    # If I group them into 3 blocks of (2x2 * 2x6), the supports are disjoint in k, but shared in j?
    # No, j is split.
    # Block 1: j in {0,1}, k in {0,1,2,3,4,5}.
    # Block 2: j in {2,3}, k in {0,1,2,3,4,5}.
    # These are disjoint in j.
    # So T = T1 + T2.
    # Rank(T1) = Rank(2x2 * 2x6).
    # Rank(T2) = Rank(2x2 * 2x6).
    # Total Rank = 2 * Rank(2x2 * 2x6).
    # If Rank(2x2 * 2x6) = 12, then Total = 24.
    # Is Rank(2x2 * 2x6) = 12?
    # Lower bound is 12.
    # Trivial is 12.
    # So yes, Rank(2x2 * 2x6) is 12.
    # So Total Rank = 24.
    
    # Let's construct the rank-24 decomposition.
    # T1: A1 (2x2) * B1 (2x6).
    # Trivial decomposition of 2x2 * 2x6:
    # C_{i,k} = sum_{j=0}^1 A1_{i,j} B1_{j,k}.
    # Terms: (i,j,k) for i in {0,1}, j in {0,1}, k in {0..5}.
    # 2*2*6 = 24 terms?
    # No, for T1, it's 2*2*6 = 24 terms?
    # Wait.
    # A1 is 2x2. B1 is 2x6.
    # i in {0,1}, j in {0,1}, k in {0..5}.
    # 2*2*6 = 24 terms for T1 alone?
    # No.
    # C_{i,k} = A1_{i,0}B1_{0,k} + A1_{i,1}B1_{1,k}.
    # For each i, k, there are 2 terms.
    # Total terms = 2 (i) * 6 (k) * 2 (j) = 24.
    # So T1 has rank 24?
    # But trivial rank of 2x2 * 2x6 is 2*2*6 = 24.
    # So T1 rank is 24.
    # T2 rank is 24.
    # Total rank 48.
    # This is the trivial decomposition.
    
    # So my "Shared A" idea for 2x2*2x6 using Strassen (rank 12) is the key.
    # Can 2x2 * 2x6 be done in rank 12?
    # Lower bound is 12.
    # Trivial is 12.
    # So yes, rank 12.
    # But Strassen for 2x2*2x2 is 7.
    # For 2x2*2x6, we can do 3 * 7 = 21?
    # Or 12?
    # If we use trivial, it's 12.
    # If we use Strassen on 2x2*2x2 blocks, it's 21.
    # So trivial is better for 2x2*2x6.
    
    # So for 2x4 * 4x6:
    # Split m=4 into 2+2.
    # T = T1 + T2.
    # T1 is 2x2 * 2x6. Rank 12 (trivial).
    # T2 is 2x2 * 2x6. Rank 12 (trivial).
    # Total 24.
    
    # Let's implement the rank-24 trivial decomposition for the split blocks.
    # This is effectively the same as the global trivial decomposition (rank 48) but with sharing?
    # No, global trivial is 48.
    # Split trivial is 24.
    # Why?
    # Global trivial:
    # C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k}.
    # 2*4*6 = 48 terms.
    # Split trivial:
    # C_{i,k} = sum_{j=0}^1 A1_{i,j} B1_{j,k} + sum_{j=2}^3 A2_{i,j-2} B2_{j-2,k}.
    # Term 1: A1_{i,0} B1_{0,k}.
    # Term 2: A1_{i,1} B1_{1,k}.
    # Term 3: A2_{i,0} B2_{0,k}.
    # Term 4: A2_{i,1} B2_{1,k}.
    # Total 4 terms per (i,k).
    # 2*6*4 = 48 terms.
    # So split trivial is also 48.
    
    # So where did I get 24?
    # I said Rank(2x2 * 2x6) = 12.
    # 2*2*6 = 24.
    # So Rank(2x2 * 2x6) is 24?
    # No, trivial is 24.
    # Is it possible to do 2x2 * 2x6 in less than 24?
    # Lower bound 12.
    # Known rank?
    # If I use Strassen for the 2x2 part?
    # A1 is 2x2.
    # B1 is 2x6.
    # C = A1 * B1.
    # This is a linear map from B1 to C.
    # For each column k of B1, we do 2x2 * 2x1 multiplication.
    # Rank of 2x2 * 2x1 is 4? (Trivial 4).
    # Can it be done in 3?
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    # 3 mults?
    # M1 = (a00+a11)(b0+b1)
    # M2 = (a10+a11)b0
    # M3 = a00(b1-b0)
    # C0 = M1 + M3
    # C1 = M1 - M2 + M3?
    # Let's check.
    # M1 + M3 = (a00+a11)(b0+b1) + a00(b1-b0)
    # = a00b0 + a00b1 + a11b0 + a11b1 + a00b1 - a00b0
    # = 2a00b1 + a11b0 + a11b1.
    # Not C0.
    # So rank 3 is not trivial.
    # Actually, rank of 2x2 * 2x1 is 4.
    # So 2x2 * 2x6 is 6 * 4 = 24.
    # So trivial is optimal for 2x2 * 2x6?
    # Likely.
    
    # So rank 42 is likely close to optimal.
    # The current program has rank 42.
    # I will return the rank 42 decomposition.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            # M1
            U[0*4+j0, r] = 1; U[1*4+j0+1, r] = 1
            V[j0*6+k0, r] = 1; V[(j0+1)*6+(k0+1), r] = 1
            W[k0*2+0, r] = 1; W[(k0+1)*2+0, r] = 1
            W[k0*2+1, r] = 1; W[(k0+1)*2+1, r] = 1
            r += 1
            # M2
            U[1*4+j0, r] = 1; U[1*4+j0+1, r] = 1
            V[j0*6+k0, r] = 1
            W[k0*2+1, r] = 1
            W[(k0+1)*2+1, r] = -1
            r += 1
            # M3
            U[0*4+j0, r] = 1
            V[j0*6+(k0+1), r] = 1; V[(j0+1)*6+(k0+1), r] = -1
            W[(k0+1)*2+0, r] = 1
            W[(k0+1)*2+1, r] = 1
            r += 1
            # M4
            U[1*4+j0+1, r] = 1
            V[(j0+1)*6+k0, r] = 1; V[j0*6+k0, r] = -1
            W[k0*2+0, r] = 1
            W[k0*2+1, r] = 1
            r += 1
            # M5
            U[0*4+j0, r] = 1; U[0*4+j0+1, r] = 1
            V[(j0+1)*6+(k0+1), r] = 1
            W[k0*2+0, r] = -1
            W[(k0+1)*2+0, r] = 1
            r += 1
            # M6
            U[1*4+j0, r] = 1; U[0*4+j0, r] = -1
            V[j0*6+k0, r] = 1; V[j0*6+(k0+1), r] = 1
            W[(k0+1)*2+1, r] = 1
            r += 1
            # M7
            U[0*4+j0+1, r] = 1; U[1*4+j0+1, r] = -1
            V[(j0+1)*6+k0, r] = 1; V[(j0+1)*6+(k0+1), r] = 1
            W[k0*2+0, r] = 1
            r += 1

    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=50):
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                u_val = np.dot(U_mat[:, r], A.flatten())
                v_val = np.dot(V_mat[:, r], B.flatten())
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += u_val * v_val * W_mat[k*n+i, r]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    if verify(U, V, W, n, m, p, num_tests=50):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank_t = n * m * p
    U_t = [[0] * rank_t for _ in range(n * m)]
    V_t = [[0] * rank_t for _ in range(m * p)]
    W_t = [[0] * rank_t for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_t[i * m + j][r] = 1
                V_t[j * p + k][r] = 1
                W_t[k * n + i][r] = 1
    return {"U": U_t, "V": V_t, "W": W_t}
# EVOLVE-BLOCK-END
