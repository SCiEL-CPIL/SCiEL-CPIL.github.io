import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    # We will build the U, V, W matrices as lists of columns (terms).
    # Each term is a rank-1 tensor U_i, V_i, W_i.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict {(i, j): val} for U[i, j] where index is i*m+j
        v_vals: dict {(j, k): val} for V[j, k] where index is j*p+k
        w_vals: dict {(k, i): val} for W[k, i] where index is k*n+i
        """
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        
        for (i, j), val in u_vals.items():
            U_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            V_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            W_vec[k * n + i] = val
            
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)

    # Strassen decomposition for 2x2 * 2x2
    # Indices: A rows i0, i1; cols j0, j1
    #           B rows j0, j1; cols k0, k1
    #           C rows i0, i1; cols k0, k1
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        # M1 = (A_i0 + A_i1)(B_j0 + B_j1) -> C_i0k0 += M1, C_i1k1 += M1
        # M2 = (A_i0 + A_i1)B_j1 -> C_i0k1 -= M2, C_i1k1 += M2
        # M3 = A_i0(B_j0 + B_j1) -> C_i0k0 += M3, C_i0k1 += M3
        # M4 = A_i1(B_j0 + B_j1) -> C_i1k0 += M4, C_i1k1 += M4
        # M5 = (A_i0 - A_i1)B_j0 -> C_i0k0 += M5, C_i1k0 -= M5
        # M6 = (A_i0 - A_i1)B_j1 -> C_i0k1 += M6, C_i1k1 -= M6
        # M7 = (A_i1 - A_i0)(B_j1 - B_j0) -> C_i1k0 += M7, C_i0k1 -= M7
        
        # Note: The standard Strassen algorithm has 7 multiplications.
        # Let's use the standard formulation:
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
        
        # 1. M1
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        # 2. M2
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1})
        # 3. M3
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1})
        # 4. M4
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1})
        # 5. M5
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k1, i0): 1, (k1, i1): -1}) # Wait, M5 contributes to C00 and C01?
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My previous add_term calls:
        # 1. M1: U=(i0,j0)+(i1,j1), V=(j0,k0)+(j1,k1), W=(k0,i0)+(k1,i1). Correct for C00 and C11.
        # 2. M2: U=(i1,j0)+(i1,j1), V=(j0,k0), W=(k0,i1). Correct for C10 and C11 (with -1).
        #    Wait, M2 is added to C10 and subtracted from C11.
        #    So W should be (k0,i1):1 and (k0,i1):-1? No, one term per multiplication.
        #    The term is M2 * (e_i1, e_j0) ... no.
        #    M2 = (A10+A11)B00.
        #    Contribution to C10: +M2. Contribution to C11: -M2.
        #    So we need two terms? No, Strassen is 7 terms.
        #    The "term" in tensor decomposition corresponds to one multiplication.
        #    The U, V, W vectors define the inputs and outputs.
        #    For M2:
        #    U: A10 and A11 are multiplied by B00.
        #    V: B00.
        #    W: The result M2 is added to C10 and subtracted from C11.
        #    So W should have (k0, i1): 1 and (k0, i1): -1? No, k is the column of C.
        #    C is 2x2. Indices k0, k1.
        #    M2 contributes to C[1,0] and C[1,1].
        #    So W[k0, i1] += 1 and W[k1, i1] -= 1.
        
        # Let's restart the Strassen terms carefully.
        
        # Reset local lists for this block? No, add_term appends to global.
        # I will define the 7 terms explicitly.
        
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        
        # M2 = (A10 + A11)B00
        # C10 += M2, C11 -= M2
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        
        # M3 = A00(B01 - B11)
        # C01 += M3, C11 += M3
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        
        # M4 = A11(B10 - B00)
        # C00 += M4, C10 += M4
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        
        # M5 = (A00 + A01)B11
        # C00 -= M5, C01 += M5
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k1, i0): 1, (k1, i1): -1}) # Wait, C00 is (k0, i0).
        # M5 contributes to C00 (index k0, i0) with -1 and C01 (index k1, i0) with +1.
        # So W should be {(k0, i0): -1, (k1, i0): 1}.
        # My previous add_term had {(k1, i0): 1, (k1, i1): -1}. This is wrong.
        
        # Let's do this systematically.
        # I will clear the last few terms added if I made a mistake, but it's easier to just write the correct ones.
        # Since I can't easily "undo", I will assume the previous calls were part of the logic.
        # Actually, I can just overwrite the logic. I will define the terms correctly below.
        
        # To be safe, I will not use the previous add_term calls in this function body 
        # if I'm rewriting the function. I am rewriting the whole construct().
        
        pass # The code above is just thought process. The actual code below will be clean.

    # Correct Strassen 2x2
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        
        # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        
        # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k0, i0): -1, (k1, i0): 1})
        
        # M6 = (A10 - A00)(B00 + B01) -> C01 += M6, C11 += M6
        add_term({(i1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, i1): 1, (k1, i0): 1}) # Wait.
        # M6 = (A10 - A00)(B00 + B01).
        # C01 += M6? No.
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check M6 contribution.
        # M6 = (A10 - A00)(B00 + B01).
        # It contributes to C01? No, C01 = M3 + M5.
        # It contributes to C11? Yes, C11 = ... + M6.
        # Does it contribute to C00? No.
        # Does it contribute to C10? No.
        # So M6 only affects C11?
        # Let's re-read a reliable source.
        # Wikipedia:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C00+, C11+
        # M2: C10+, C11-
        # M3: C01+, C11+
        # M4: C00+, C10+
        # M5: C00-, C01+
        # M6: C11+ (Only C11? Let's check C00/C01/C10. No M6 in them.)
        # M7: C00+, C01- (Wait. C00 has +M7. C01 has -M7? No, C01 = M3+M5. C00 = M1+M4-M5+M7. C11 = M1-M2+M3+M6. C10 = M2+M4.)
        # Where is M7 in C01? It's not.
        # Where is M7 in C11? It's not.
        # Where is M7 in C10? It's not.
        # So M7 only affects C00?
        # Let's check C00 again. M1 + M4 - M5 + M7.
        # Let's check C01. M3 + M5.
        # Let's check C10. M2 + M4.
        # Let's check C11. M1 - M2 + M3 + M6.
        
        # This implies M6 and M7 are single-output terms?
        # M6 = (A10 - A00)(B00 + B01).
        # If M6 only adds to C11, then W for M6 should be {(k1, i1): 1}.
        # M7 = (A01 - A11)(B10 + B11).
        # If M7 only adds to C00, then W for M7 should be {(k0, i0): 1}.
        
        # Let's verify M6.
        # (A10 - A00)(B00 + B01) = A10B00 + A10B01 - A00B00 - A00B01.
        # C11 = A10B01 + A11B11.
        # This doesn't look like it sums to C11 alone.
        
        # I will use the known correct 7-term decomposition from the previous program which worked.
        # The previous program used:
        # 1. (i0,j0)+(i1,j1) * (j0,k0)+(j1,k1) -> (k0,i0)+(k1,i1)
        # 2. (i1,j0)+(i1,j1) * (j0,k0) -> (k0,i1)-(k1,i1)  [Wait, prev had (k0,i1):1, (k1,i1):-1]
        # 3. (i0,j0) * (j0,k1)-(j1,k1) -> (k1,i0)+(k1,i1) [Prev had (k1,i0):1, (k1,i1):1]
        # 4. (i1,j1) * (j1,k0)-(j0,k0) -> (k0,i0)+(k0,i1) [Prev had (k0,i0):1, (k0,i1):1]
        # 5. (i0,j0)+(i0,j1) * (j1,k1) -> (k0,i0)-(k1,i0) [Prev had (k0,i0):-1, (k1,i0):1]
        # 6. (i1,j0)-(i0,j0) * (j0,k0)+(j0,k1) -> (k1,i1) [Prev had (k1,i1):1]
        # 7. (i0,j1)-(i1,j1) * (j1,k0)+(j1,k1) -> (k0,i0) [Prev had (k0,i0):1]
        
        # Let's check if this set covers C.
        # C00 (k0,i0): T1(1) + T4(1) + T5(-1) + T7(1) = M1+M4-M5+M7. Correct.
        # C01 (k1,i0): T3(1) + T5(1) = M3+M5. Correct.
        # C10 (k0,i1): T2(1) + T4(1) = M2+M4. Correct.
        # C11 (k1,i1): T1(1) + T2(-1) + T3(1) + T6(1) = M1-M2+M3+M6. Correct.
        
        # So the previous terms were correct. I will reuse them.

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k0, i0): -1, (k1, i0): 1})
        add_term({(i1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, i1): 1})
        add_term({(i0, j1): 1, (i1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, i0): 1})

    def mm_2x1_1x2(i0, i1, j, k0, k1):
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def mm_1x2_2x1(i, j0, j1, k):
        for j in [j0, j1]:
            add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def mm_2x3_3x2(i0, i1, j0, j1, j2, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        mm_2x1_1x2(i0, i1, j2, k0, k1)

    def mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})

    def mm_2x3_3x3(i0, i1, j0, j1, j2, k0, k1, k2):
        mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2)
        for i in [i0, i1]:
            for k in [k0, k1, k2]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x2_2x2(i0, i1, i2, j0, j1, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    def mm_3x3_3x2(i0, i1, i2, j0, j1, j2, k0, k1):
        mm_3x2_2x2(i0, i1, i2, j0, j1, k0, k1)
        for i in [i0, i1, i2]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x3_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2)
        for i in [i0, i1, i2]:
            for k in [k0, k1, k2]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2)
        for j in [j0, j1]:
            for k in [k0, k1, k2]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    # Block decomposition 2+3
    strassen_2x2(0, 1, 0, 1, 0, 1)
    mm_2x3_3x2(0, 1, 2, 3, 4, 0, 1)
    mm_2x2_2x3(0, 1, 0, 1, 2, 3, 4)
    mm_2x3_3x3(0, 1, 2, 3, 4, 2, 3, 4)
    mm_3x2_2x2(2, 3, 4, 0, 1, 0, 1)
    mm_3x3_3x2(2, 3, 4, 2, 3, 4, 0, 1)
    mm_3x2_2x3(2, 3, 4, 0, 1, 2, 3, 4)
    mm_3x3_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    rank = U.shape[1]

    # Quick verification
    if time.time() - t0 < deadline - 5:
        ok = True
        # Sample a few entries to save time? No, verify all to be safe.
        # 125 entries * 120 rank = 15000 dot products. Fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Fallback to naive
            U = np.zeros((n*m, n*m*p))
            V = np.zeros((m*p, n*m*p))
            W = np.zeros((p*n, n*m*p))
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        idx = i2*m*p + j2*p + k2
                        U[i2*m+j2, idx] = 1
                        V[j2*p+k2, idx] = 1
                        W[k2*n+i2, idx] = 1
            rank = n*m*p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
