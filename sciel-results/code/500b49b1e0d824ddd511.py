def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We will build lists of terms first.
    # Each term is a dict or sparse representation.
    # Since rank is small, we can use dense lists or dicts.
    
    terms = []
    
    # Define the 7 Strassen terms for a 2x2x2 block
    # Indices: A(0,0), A(0,1), A(1,0), A(1,1)
    #          B(0,0), B(0,1), B(1,0), B(1,1)
    #          C(0,0), C(0,1), C(1,0), C(1,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)B00 -> C10 += 1
    # M3 = A00(B01-B11) -> C01 += 1
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    # M7 = (A01-A11)(B10+B11) -> C00 += 1, C11 += 1
    
    # Let's verify the C combination:
    # C00: M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? 
    # Wait, Strassen formula:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # My previous Ccombos in the code were:
    # C[0,0]: M1(1), M4(1), M5(-1), M7(1) -> Sum 2?
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # If A,B are 2x2.
    # Let's check M1: (A11+A22)(B11+B22)
    # If A=I, B=I.
    # M1 = (1+1)(1+1) = 4.
    # M4 = A22(B21-B11) = 1(0-1) = -1.
    # M5 = (A11+A12)B22 = (1+0)1 = 1.
    # M7 = (A12-A22)(B21+B22) = (0-1)(0+1) = -1.
    # C11 = 4 - 1 - 1 - 1 = 1. Correct.
    
    # So the coefficients for W are:
    # Term 1 (M1): W[C00]=1, W[C11]=1
    # Term 2 (M2): W[C10]=1
    # Term 3 (M3): W[C01]=1
    # Term 4 (M4): W[C00]=1, W[C10]=1
    # Term 5 (M5): W[C00]=-1, W[C01]=1
    # Term 6 (M6): W[C11]=1
    # Term 7 (M7): W[C00]=1, W[C11]=1
    
    # Wait, M7 in C11?
    # C11 = M1 + M4 - M5 + M7. Yes.
    # C22 = M1 - M2 + M3 + M6.
    # So M1 contributes to C00 and C11.
    # M2 contributes to C10 and C11 (with -1).
    # M3 contributes to C01 and C11.
    # M4 contributes to C00 and C10.
    # M5 contributes to C00 (-1) and C01.
    # M6 contributes to C11.
    # M7 contributes to C00.
    
    # Let's list W for each term:
    # T1: {C00:1, C11:1}
    # T2: {C10:1, C11:-1}
    # T3: {C01:1, C11:1}
    # T4: {C00:1, C10:1}
    # T5: {C00:-1, C01:1}
    # T6: {C11:1}
    # T7: {C00:1}
    
    # And U, V:
    # T1: U{A00:1, A11:1}, V{B00:1, B11:1}
    # T2: U{A10:1, A11:1}, V{B00:1}
    # T3: U{A00:1}, V{B01:1, B11:-1}
    # T4: U{A11:1}, V{B10:1, B00:-1}
    # T5: U{A00:1, A01:1}, V{B11:1}
    # T6: U{A10:1, A00:-1}, V{B00:1, B01:1}
    # T7: U{A01:1, A11:-1}, V{B10:1, B11:1}

    # We will iterate over the 6 blocks and add these 7 terms.
