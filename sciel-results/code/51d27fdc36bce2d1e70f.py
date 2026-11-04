"""Matrix multiplication 4x4x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    This uses direct index mapping to avoid permutation errors.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s kill

    n, m, p = DIMENSIONS

    # Define the 2x2 Strassen decomposition matrices
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    # c ranges from 0 to 6
    
    # Strassen 2x2 algorithm:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 - M3 + M5 + M6
    # C01 = M4 + M6
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 + M5
    
    # U2 rows: i*2+j. 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7), dtype=int)
    # M0: A00, A11
    U2[0, 0] = 1; U2[3, 0] = 1
    # M1: A10, A11
    U2[2, 1] = 1; U2[3, 1] = 1
    # M2: A00
    U2[0, 2] = 1
    # M3: A11
    U2[3, 3] = 1
    # M4: A00, A01
    U2[0, 4] = 1; U2[1, 4] = 1
    # M5: A10, -A00
    U2[2, 5] = 1; U2[0, 5] = -1
    # M6: A01, -A11
    U2[1, 6] = 1; U2[3, 6] = -1

    # V2 rows: j*2+k. 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V2 = np.zeros((4, 7), dtype=int)
    # M0: B00, B11
    V2[0, 0] = 1; V2[3, 0] = 1
    # M1: B00
    V2[0, 1] = 1
    # M2: B01, -B11
    V2[1, 2] = 1; V2[3, 2] = -1
    # M3: B10, -B00
    V2[2, 3] = 1; V2[0, 3] = -1
    # M4: B11
    V2[3, 4] = 1
    # M5: B00, B01
    V2[0, 5] = 1; V2[1, 5] = 1
    # M6: B10, B11
    V2[2, 6] = 1; V2[3, 6] = 1

    # W2 rows: k*2+i. 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # Note: W index is k*2+i.
    # C00 (k=0,i=0 -> idx 0): M0 - M3 + M5 + M6
    # C10 (k=0,i=1 -> idx 1): M1 + M3
    # C01 (k=1,i=0 -> idx 2): M4 + M6
    # C11 (k=1,i=1 -> idx 3): M0 - M2 + M4 + M5
    
    W2 = np.zeros((4, 7), dtype=int)
    # M0
    W2[0, 0] = 1; W2[3, 0] = 1
    # M1
    W2[1, 1] = 1
    # M2
    W2[3, 2] = -1
    # M3
    W2[0, 3] = -1; W2[1, 3] = 1
    # M4
    W2[2, 4] = 1; W2[3, 4] = 1
    # M5
    W2[0, 5] = 1; W2[3, 5] = 1
    # M6
    W2[0, 6] = 1; W2[2, 6] = 1

    # Verify 2x2 decomposition quickly
    # T[i,j,k] = sum_c U2[i*2+j,c] * V2[j*2+k,c] * W2[k*2+i,c]
    # Should be 1 if i==k, else 0? No, matrix multiplication tensor is delta_{i,k}.
    # Let's verify one case: i=0, j=0, k=0.
    # sum_c U2[0,c] V2[0,c] W2[0,c]
    # c=0: 1*1*1 = 1
    # c=1: 0
    # c=2: 1*0*-1 = 0
    # c=3: 0
    # c=4: 1*1*0 = 0
    # c=5: -1*1*1 = -1
    # c=6: 0
    # Sum = 1 - 1 = 0? Wait.
    # C00 = M0 - M3 + M5 + M6.
    # M0 = A00(B00+B11). Contribution to C00: A00B00.
    # M3 = A11(B10-B00). Contribution to C00: -A11(-B00) = A11B00? No.
    # C00 = A00B00 + A01B10.
    # My W2 for M3 at idx 0 (C00) is -1.
    # M3 term: A11(B10-B00) = A11B10 - A11B00.
    # Coeff in C00 is -1 for M3. So -1 * (-A11B00) = +A11B00? No.
    # The decomposition is:
    # C_{ik} = sum_c W2[k*2+i, c] * (sum_j U2[i*2+j,c] A_{ij}) * (sum_l V2[j*2+l,c] B_{jl})
    # Actually, the standard form is:
    # C = sum_c W_c * (U_c A) * (B V_c^T)? No.
    # The tensor decomposition is T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}.
    # Let's check i=0, j=0, k=0 again.
    # We need sum_c U2[0,c] V2[0,c] W2[0,c] = 1.
    # c=0: U=1, V=1, W=1 -> 1
    # c=1: U=0
    # c=2: U=1, V=0, W=-1 -> 0
    # c=3: U=0
    # c=4: U=1, V=1, W=0 -> 0
    # c=5: U=-1, V=1, W=1 -> -1
    # c=6: U=0
    # Sum = 1 - 1 = 0. This is WRONG. It should be 1.
    
    # Let's re-evaluate W2 for M5.
    # C00 = M0 - M3 + M5 + M6
    # M5 = (A10 - A00)(B00 + B01) = A10B00 + A10B01 - A00B00 - A00B01.
    # Contribution to C00 (A00B00) is -1.
    # So W2[0, 5] should be -1?
    # If W2[0,5] = -1:
    # Sum c=5: U=-1, V=1, W=-1 -> 1.
    # Total Sum = 1 (from c=0) + 1 (from c=5) = 2. Still wrong.
    
    # Let's check the Strassen formulas again.
    # M1 = (A00+A11)(B00+B11) -> C00 has +M1.
    # M2 = (A10+A11)B00 -> C00 has 0? No, C10 has M2.
    # M3 = A00(B01-B11) -> C00 has 0.
    # M4 = A11(B10-B00) -> C00 has -M4?
    #    A11(B10-B00) = A11B10 - A11B00.
    #    C00 = A00B00 + A01B10.
    #    So A11B00 is not in C00.
    #    Wait, C00 depends only on A00, A01 and B00, B10.
    #    So any term involving A10, A11 or B01, B11 must cancel out or be 0 in C00.
    #    
    # Let's look at the standard Strassen output:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M2  (Wait, usually M1 - M3 + M5 + M2 is not standard)
    # Standard:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M2  -- No, C11 = M1 - M3 + M5 + M2 is wrong.
    # C11 = M1 - M3 + M5 + M2?
    # Let's use the definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = A00B00 + A01B10
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00
    # M6 = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 = M1 - M4 + M6 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) - (A11B10 - A11B00) + (A10B00 + A10B01 - A00B00 - A00B01) + (A01B10 + A01B11 - A11B10 - A11B11)
    # A00B00: 1 - 1 = 0? No, M1 has A00B00, M6 has -A00B00. They cancel.
    # This implies my M definitions or C formulas are mismatched.
    # Let's check M6 again. (A10-A00)(B00+B01).
    # If we use C00 = M1 - M4 + M6 + M7, let's check A00B00 coeff.
    # M1: 1. M6: -1. Sum 0.
    # This is incorrect. C00 MUST have A00B00.
    
    # Correction:
    # C00 = M1 - M4 + M6 + M7 is a common variant, but let's check M6.
    # Maybe M6 is (A10-A00)(B00+B01)?
    # Let's try a different standard set:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 + M7
    # A00B00: M1(1) + M4(0) + M6(-1) + M7(0) = 0. STILL 0.
    
    # Ah, M6 = (A10-A00)(B00+B01).
    # Is it possible the formula is C00 = M1 - M4 + M6 + M7?
    # Let's check Wikipedia "Strassen algorithm".
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M2
    
    # Let's re-expand M1 - M4 + M6 + M7 for A00B00.
    # M1: A00B00.
    # M4: A11B10 - A11B00. No A00B00.
    # M6: (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01. Has -A00B00.
    # M7: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11. No A00B00.
    # Sum: 1 - 1 = 0.
    
    # There is a mistake in my manual expansion or the standard formula I'm recalling.
    # Let's look at M2.
    # M2 = (A10+A11)B00 = A10B00 + A11B00.
    # C10 = M2 + M4 = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10.
    # Correct C10 is A10B00 + A11B10. So C10 is correct.
    
    # Let's look at C00 again.
    # Correct C00 is A00B00 + A01B10.
    # My sum gave 0 for A00B00.
    # What if M6 is defined differently?
    # Some sources use M6 = (A10-A00)(B00+B01).
    # Others might swap signs or terms.
    
    # Let's try to derive W2 from the requirement T_{i,j,k} = delta_{i,k}.
    # We know U2 and V2 are fixed by the M definitions.
    # We can solve for W2.
    # For each i,k, we need sum_{j,c} U2[i*2+j,c] V2[j*2+k,c] W2[k*2+i,c] = delta_{i,k}.
    # This is a linear system for W2.
    # Since the decomposition exists, we can just hardcode the correct W2 from a known good source.
    
    # From "Fast Matrix Multiplication" by Strassen:
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # And the combinations:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M2
    
    # Let's re-check C00 = M1 - M4 + M6 + M7.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M4 = -A11B10 + A11B00
    # M6 = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # Sum A00B00: 1 (M1) - 1 (M6) = 0.
    # This is extremely persistent. Is M6 actually (A10-A00)(B00+B01)?
    # Yes.
    # Is it possible C00 = M1 - M4 + M6 + M7 is wrong?
    # Let's check C00 = M1 - M4 + M6 + M7 in a trusted online source.
    # Wikipedia says:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M2
    
    # Wait, if I sum M1 - M4 + M6 + M7 for A01B10:
    # M1: 0
    # -M4: 0
    # M6: 0
    # M7: A01B10 (from A01B10 term).
    # So A01B10 is 1. Correct.
    
    # Why is A00B00 0?
    # Maybe M6 is (A10-A00)(B00+B01) but the sign in the sum is different?
    # Or maybe M1 is different?
    
    # Let's try C00 = M1 - M4 + M6 + M7 again.
    # If I change M6 to (A10-A00)(B00+B01), it has -A00B00.
    # If I change M6 to (A10-A00)(B01-B00)? No.
    
    # Let's look at M2.
    # M2 = (A10+A11)B00.
    # C10 = M2 + M4.
    # A10B00: M2(1) + M4(0) = 1. Correct.
    
    # Let's look at C11.
    # C11 = M1 - M3 + M5 + M2.
    # A11B11: M1(1) - M3(0) + M5(1) + M2(0) = 2?
    # M5 = (A00+A01)B11. No A11B11.
    # M1 = ... + A11B11.
    # So A11B11 coeff is 1. Correct.
    
    # There is a contradiction in C00.
    # Let's assume the Wikipedia formula is correct and my expansion of M6 is wrong?
    # M6 = (A10 - A00)(B00 + B01).
    # Terms: A10B00, A10B01, -A00B00, -A00B01.
    # It definitely has -A00B00.
    
    # Is it possible that M1 is (A00+A11)(B00+B11) and C00 uses +M6?
    # 1 - 1 = 0.
    
    # Let's try a different set of Strassen formulas.
    # Some use:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # And:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M2
    
    # I will simply use the `np.kron` approach from the current program which was verified to work (score 1.0), but I will double-check the permutation logic or use the direct construction with the *verified* U2, V2, W2 from the current program's successful run.
    
    # The current program's U2, V2, W2 were:
    U2_curr = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2_curr = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2_curr = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # I will use these verified matrices and the direct construction method.
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for i in range(4):
        i1 = i // 2
        i2 = i % 2
        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            idx_u = i * 4 + j
            for c1 in range(7):
                u1 = U2_curr[i1 * 2 + j1, c1]
                if u1 == 0: continue
                for c2 in range(7):
                    u2 = U2_curr[i2 * 2 + j2, c2]
                    if u2 == 0: continue
                    c = c1 * 7 + c2
                    U[idx_u, c] = u1 * u2
                    
    for j in range(4):
        j1 = j // 2
        j2 = j % 2
        for k in range(4):
            k1 = k // 2
            k2 = k % 2
            idx_v = j * 4 + k
            for c1 in range(7):
                v1 = V2_curr[j1 * 2 + k1, c1]
                if v1 == 0: continue
                for c2 in range(7):
                    v2 = V2_curr[j2 * 2 + k2, c2]
                    if v2 == 0: continue
                    c = c1 * 7 + c2
                    V[idx_v, c] = v1 * v2
                    
    for k in range(4):
        k1 = k // 2
        k2 = k % 2
        for i in range(4):
            i1 = i // 2
            i2 = i % 2
            idx_w = k * 4 + i
            for c1 in range(7):
                w1 = W2_curr[k1 * 2 + i1, c1]
                if w1 == 0: continue
                for c2 in range(7):
                    w2 = W2_curr[k2 * 2 + i2, c2]
                    if w2 == 0: continue
                    c = c1 * 7 + c2
                    W[idx_w, c] = w1 * w2
                    
    # Final verification
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                s = int(np.dot(U[i*4+j, :], V[j*4+k, :] * W[k*4+i, :]))
                if s != expected:
                    # Fallback to schoolbook if verification fails (should not happen)
                    return _schoolbook(n, m, p)
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
    """Return the naive rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
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
