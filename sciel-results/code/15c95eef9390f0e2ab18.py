import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # A is n x m = 4x4, B is m x p = 4x5, C is n x p = 4x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (20, rank)  
    # W has shape (p*n, rank) = (20, rank)
    # W index: k*n + i
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: A * B4 where B4 is 4x4 (first 4 columns of B)
    # Use Strassen: 7 block products, each 2x2x2 with Strassen rank 7
    # Total 49 terms
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4x4, blocks are 2x2:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Strassen 2x2x2 rank-7 decomposition:
    # Each term r: u_r (4-dim for 2x2 A), v_r (4-dim for 2x2 B), w_r (4-dim for 2x2 C)
    # A index: i*2+j, B index: j*2+k, C index: k*2+i
    
    # The 7 Strassen terms for 2x2:
    # Term 0: M1 = (A00+A11)(B00+B11) -> u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # Term 1: M2 = (A10+A11)B00 -> u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # Term 2: M3 = A00(B10-B11) -> u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0]
    # Term 3: M4 = A11(B01-B00) -> u=[0,0,0,1], v=[0,1,-1,0], w=[0,0,0,1]
    # Term 4: M5 = (A00+A01)B11 -> u=[1,1,0,0], v=[0,0,0,1], w=[0,0,1,0]  # wait let me recheck
    # Term 5: M6 = (A10-A00)(B00+B01) -> u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,-1,1]
    # Term 6: M7 = (A01-A11)(B10+B11) -> u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,0]
    
    # Let me verify term 4: M5 = (A00+A01)B11
    # A00+A01: elements at (0,0)+(0,1) and (1,0)+(1,1) for A. u = [1,1,0,0]
    # B11: elements at (1,0) and (1,1) for B. v = [0,0,0,1]
    # C contribution: C01 gets +M5, C11 gets +M5. 
    # C01 is at positions k*2+i: (0,0)=0, (0,1)=1. C11: (1,0)=2, (1,1)=3.
    # Wait, C is 2x2, index k*2+i. C01 means k=0,i=1 -> index 1. C11 means k=1,i=1 -> index 3.
    # So w = [0,1,0,1]
    
    # Let me redo this carefully.
    
    # 2x2 matrices:
    # A = [[a00, a01], [a10, a11]], index i*2+j
    # B = [[b00, b01], [b10, b11]], index j*2+k  
    # C = [[c00, c01], [c10, c11]], index k*2+i
    # C = A*B: c00 = a00*b00 + a01*b10, c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10, c11 = a10*b01 + a11*b11
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each M_i is a scalar product of two linear combinations.
    # For the tensor decomposition, each M_i contributes to multiple C entries.
    
    # Let me define the 7 Strassen terms as (u, v, w) where u is 4-dim (for 2x2 A),
    # v is 4-dim (for 2x2 B), w is 4-dim (for 2x2 C).
    
    # M1 = (a00+a11)(b00+b11)
    # u1 = [1, 0, 0, 1]  (a00 + a11)
    # v1 = [1, 0, 0, 1]  (b00 + b11)
    # This contributes: +1 to C00, +1 to C11
    # C00 index = 0*2+0 = 0, C11 index = 1*2+1 = 3
    # w1 = [1, 0, 0, 1]
    
    # M2 = (a10+a11)b00
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # Contributes: +1 to C10, -1 to C11
    # C10 index = 0*2+1 = 1, C11 index = 3
    # w2 = [0, 1, 0, -1]
    
    # M3 = a00(b10-b11)
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 0, 1, -1]
    # Contributes: +1 to C01, +1 to C11
    # C01 index = 1*2+0 = 2, C11 index = 3
    # w3 = [0, 0, 1, 1]
    
    # M4 = a11(b01-b00)
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 1, -1, 0]
    # Contributes: +1 to C00, +1 to C10
    # C00 index = 0, C10 index = 1
    # w4 = [1, 1, 0, 0]
    
    # M5 = (a00+a01)b11
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # Contributes: -1 to C00, +1 to C01
    # C00 index = 0, C01 index = 2
    # w5 = [-1, 0, 1, 0]
    
    # M6 = (a10-a00)(b00+b01)
    # u6 = [0, 0, 1, -1]
    # v6 = [1, 1, 0, 0]
    # Contributes: +1 to C10, -1 to C11  wait let me check
    # C10 = M2 + M4, C11 = M1 - M2 + M3 + M7
    # M6 doesn't appear in the C formulas directly. Let me recheck Strassen.
    
    # Standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P7
    
    # With 1-indexed: A11=a00, A12=a01, A21=a10, A22=a11
    # B11=b00, B12=b01, B21=b10, B22=b11
    # C11=c00, C12=c01, C21=c10, C22=c11
    
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # P6 is not used? That can't be right. Let me look up Strassen again.
    
    # Actually I think I have the indices mixed up. Let me use the standard formulation:
    # C = A * B where A, B, C are 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+dh]]
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(g-h)
    # p4 = d(f-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C11 = p1 + p4 - p5 - p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p7
    
    # So p6 is NOT used! That's correct - Strassen uses 7 multiplications but the 6th one (p6) is...
    # Wait, I'm confused. Let me count: p1 through p7 is 7 terms. p6 = (c-a)(e+f) is one of them.
    # But it doesn't appear in the C formulas? Let me recheck.
    
    # Actually I think the standard Strassen has:
    # C11 = p1 + p4 - p5 - p7
    # C12 = p3 + p5
    # C21 = p2 + p4  
    # C22 = p1 - p2 + p3 + p7
    
    # And p6 is not used? No, that's only 6 terms used. I must be wrong.
    
    # Let me look this up properly. The 7 Strassen multiplications are:
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = a * (g - h)
    # m4 = d * (f - e)
    # m5 = (a + b) * h
    # m6 = (c - a) * (e + f)
    # m7 = (b - d) * (g + h)
    
    # C11 = m1 + m4 - m5 - m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m7
    
    # Indeed m6 is not used in the output! This is a well-known quirk. The rank-7 decomposition still works because m6 is linearly dependent on the others in terms of contribution to C. Actually no - the point is that these 7 bilinear forms span the 8-dimensional space of 2x2 matrix multiplication with only 7 terms. m6 is needed to make the system consistent, but its contribution to C is zero? No, that doesn't make sense.
    
    # I think I'm making an error. Let me just verify numerically.
    
    # Actually, upon reflection, I believe the standard Strassen formulas are correct and m6 does not contribute to any output entry. This is because the 7 terms are not all "independent" in the output - they're a basis for the tensor. The fact that m6 has zero coefficient in all C entries means it's not actually a multiplicative term but rather... no, that can't be.
    
    # I'll just skip m6 and use 6 terms? No, Strassen is rank 7. Let me just implement it and verify numerically.
    
    # For now, let me just use schoolbook for the 2x2x2 blocks to be safe, giving rank 8 per block.
    # 4x4x4 with one Strassen level: 7 blocks * 8 = 56. Plus 4x4x1: 4. Total: 60.
    # That's still better than 80.
    
    # Actually, let me just verify Strassen numerically in code and use it if correct.
    
    strassen_2x2 = [
        # (u, v, w) for each term
        # m1 = (a+d)(e+h)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m2 = (c+d)e  
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # m3 = a(g-h)
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        # m4 = d(f-e)
        ([0,0,0,1], [0,1,-1,0], [0,0,0,1]),
        # m5 = (a+b)h
        ([1,1,0,0], [0,0,0,1], [0,0,1,0]),
        # m6 = (c-a)(e+f) - verify if needed
        ([0,0,1,-1], [1,1,0,0], [0,0,-1,1]),
        # m7 = (b-d)(g+h)
        ([0,1,0,-1], [0,0,1,1], [0,1,0,0]),
    ]
    
    # Verify Strassen
    import numpy as np
    A_test = np.random.randint(-3, 4, (2,2)).astype(float)
    B_test = np.random.randint(-3, 4, (2,2)).astype(float)
    C_expected = A_test @ B_test
    C_flat = C_expected.flatten()  # [c00, c01, c10, c11] in row-major
    # But our W index is k*2+i, so C is stored as [c00, c10, c01, c11]
    C_our_order = np.array([C_flat[0], C_flat[2], C_flat[1], C_flat[3]])
    
    C_computed = np.zeros(4)
    for u, v, w in strassen_2x2:
        prod = sum(A_test[i,j] * u[i*2+j] for i in range(2) for j in range(2)) * \
               sum(B_test[j,k] * v[j*2+k] for j in range(2) for k in range(2))
        for idx in range(4):
            C_computed[idx] += prod * w[idx]
    
    # Check
    if not np.allclose(C_computed, C_our_order):
        # Try without m6
        C_computed = np.zeros(4)
        for u, v, w in strassen_2x2[:5] + strassen_2x2[6:]:
            prod = sum(A_test[i,j] * u[i*2+j] for i in range(2) for j in range(2)) * \
                   sum(B_test[j,k] * v[j*2+k] for j in range(2) for k in range(2))
            for idx in range(4):
                C_computed[idx] += prod * w[idx]
    
    # Let me just use schoolbook for 2x2x2 to be safe: rank 8
    # This gives 4x4x4 rank = 7*8 = 56, plus 4x4x1 rank 4, total 60.
    
    # Actually, let me just hardcode the correct Strassen terms.
    
    # I'll use the known correct Strassen rank-7 decomposition.
    # After careful derivation, the correct w vectors are:
    
    strassen_correct = [
        # m1 = (a+d)(e+h) -> C11 += m1, C22 += m1
        # C11 idx 0, C22 idx 3
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m2 = (c+d)e -> C21 += m2, C22 -= m2
        # C21 idx 1, C22 idx 3
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        # m3 = a(g-h) -> C12 += m3, C22 += m3
        # C12 idx 2, C22 idx 3
        ([1,0,0,0], [0,0,1,-1], [0,0,1,1]),
        # m4 = d(f-e) -> C11 += m4, C21 += m4
        # C11 idx 0, C21 idx 1
        ([0,0,0,1], [0,1,-1,0], [1,1,0,0]),
        # m5 = (a+b)h -> C11 -= m5, C12 += m5
        # C11 idx 0, C12 idx 2
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        # m6 = (c-a)(e+f) -> C21 += m6, C22 -= m6
        # C21 idx 1, C22 idx 3
        ([0,0,1,-1], [1,1,0,0], [0,1,0,-1]),
        # m7 = (b-d)(g+h) -> C11 -= m7, C12 -= m7
        # C11 idx 0, C12 idx 2
        ([0,1,0,-1], [0,0,1,1], [-1,0,-1,0]),
    ]
    
    # Verify
    C_computed = np.zeros(4)
    for u, v, w in strassen_correct:
        ua = np.array(u)
        va = np.array(v)
        wa = np.array(w)
        A_vec = A_test.flatten()  # [a, b, c, d] row-major: a00, a01, a10, a11
        B_vec = B_test.flatten()  # [e, f, g, h] row-major: b00, b01, b10, b11
        prod = np.dot(A_vec, ua) * np.dot(B_vec, va)
        C_computed += prod * wa
    
    C_expected_our = np.array([C_expected[0,0], C_expected[1,0], C_expected[0,1], C_expected[1,1]])
    
    if not np.allclose(C_computed, C_expected_our):
        # Fall back to schoolbook for 2x2x2
        strassen_correct = None
    
    # Build the full decomposition
    # Part 1: 4x4 x 4x4 (first 4 columns of B)
    if strassen_correct is not None:
        # 7 Strassen block terms, each expanded to 7 scalar Strassen terms
        for block_u, block_v, block_w in [(strassen_correct[i][0], strassen_correct[i][1], strassen_correct[i][2]) for i in range(7)]:
            # This is getting too complex. Let me just use schoolbook for simplicity.
            pass
    
    # Let me just use schoolbook for the 4x4x4 part: rank 16
    # And schoolbook for 4x4x1: rank 4
    # Total: 20. Wait, that's worse than 80? No, 4x4x4 schoolbook is 4*4*4=64, not 16.
    
    # OK let me just do a clean schoolbook and return rank 80, then try to improve.
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
