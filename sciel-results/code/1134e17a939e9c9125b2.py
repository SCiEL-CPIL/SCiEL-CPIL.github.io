import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The minimal rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
    # This is a known lower bound and achievable upper bound.
    # We construct the Strassen decomposition with rank 7.
    rank = 7
    n = 2
    m = 2
    p = 2
    
    # Initialize matrices with integer entries
    U = [[0] * rank for _ in range(n * m)]  # Shape: (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # Shape: (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # Shape: (4, 7)
    
    # Convention:
    # U[i*m+j, r] = coefficient of A_{i,j} in the r-th product
    # V[j*p+k, r] = coefficient of B_{j,k} in the r-th product
    # W[k*n+i, r] = coefficient of C_{i,k} in the r-th product (output factor)
    #
    # The tensor decomposition requires:
    # T_{i,j,k,l} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T_{i,j,k,l} = 1 if the term A_{i,j} * B_{j,k} appears in C_{i,k}, else 0.
    # More precisely, for matrix multiplication:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor is: T[i,j,k,l] = delta_{j,k} (where l is the row index of C, i.e., l=i)
    # Actually, let's be more careful. The multiplication tensor T has components:
    # T_{i,j,k,l} = 1 if j==k (the inner index matches) and we're computing C_{i,l}
    # Wait, the standard indexing: A is n x m, B is m x p, C is n x p.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is in V^* (for A) tensor V^* (for B) tensor V (for C)
    # Basis: A_{i,j} for i in [0,n), j in [0,m)
    # Basis: B_{j,k} for j in [0,m), k in [0,p)
    # Basis: C_{i,k} for i in [0,n), k in [0,p)
    # T(A_{i,j}, B_{j',k}, C_{i',k'}) = 1 if i==i' and k==k' and j==j', else 0
    #
    # With the trace convention W[k*n+i, r], the index is k*n + i where k is the column of C and i is the row of C.
    # So W[k*n+i, r] is the coefficient for C_{i,k} in the r-th product.
    
    # Strassen's 7 products:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # And the reconstruction:
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's verify this:
    # C00 = A00*B00 + A01*B10
    # P1 - P4 - P5 + P7 = (A00+A11)(B00+B11) - A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B10 + A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + (A00*B11 - A00*B11) + (A11*B00 + A11*B00) + (A11*B11 - A11*B11) - A11*B10 + A11*B00 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # Wait, let me redo this more carefully:
    # P1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # P4 = A11*B10 - A11*B00
    # P5 = A00*B11 + A01*B11
    # P7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # P1 - P4 - P5 + P7 = (A00*B00 + A00*B11 + A11*B00 + A11*B11) - (A11*B10 - A11*B00) - (A00*B11 + A01*B11) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B10 + A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + (A00*B11 - A00*B11) + (A11*B00 + A11*B00) + (A11*B11 - A11*B11) + (-A11*B10 - A11*B10) + (-A01*B11 + A01*B11) + A01*B10
    # = A00*B00 + 2*A11*B00 - 2*A11*B10 + A01*B10
    # This doesn't look right. Let me look up the correct Strassen formula.
    
    # Correct Strassen reconstruction:
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let me re-derive P1 - P4 - P5 + P7:
    # P1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # P4 = A11(B10-B00) = A11*B10 - A11*B00
    # P5 = (A00+A01)B11 = A00*B11 + A01*B11
    # P7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # 
    # P1 - P4 - P5 + P7 = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B10 + A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + (A00*B11 - A00*B11) + (A11*B00 + A11*B00) + (A11*B11 - A11*B11) + (-A11*B10 - A11*B10) + (-A01*B11 + A01*B11) + A01*B10
    # = A00*B00 + 2*A11*B00 - 2*A11*B10 + A01*B10
    # 
    # This is not C00 = A00*B00 + A01*B10. There's an error in my Strassen formula.
    # 
    # Let me look up the correct Strassen algorithm again:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # 
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # Let me check C11 = P1 - P2 + P3 + P6:
    # P1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # P2 = (A10+A11)B00 = A10*B00 + A11*B00
    # P3 = A00(B01-B11) = A00*B01 - A00*B11
    # P6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    # 
    # P1 - P2 + P3 + P6 = (A00*B00 + A00*B11 + A11*B00 + A11*B11) - (A10*B00 + A11*B00) + (A00*B01 - A00*B11) + (A10*B00 + A10*B01 - A00*B00 - A00*B01)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A10*B00 + A10*B01 - A00*B00 - A00*B01
    # = (A00*B00 - A00*B00) + (A00*B11 - A00*B11) + (A11*B00 - A11*B00) + A11*B11 + (-A10*B00 + A10*B00) + (A00*B01 - A00*B01) + A10*B01
    # = A11*B11 + A10*B01
    # 
    # But C11 = A10*B01 + A11*B11. Yes! That's correct.
    
    # Now let me check C00 again. Maybe I made an arithmetic error.
    # P1 - P4 - P5 + P7:
    # P1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # P4 = A11*B10 - A11*B00
    # P5 = A00*B11 + A01*B11
    # P7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # 
    # P1 - P4 - P5 + P7 = P1 + (-P4) + (-P5) + P7
    # = (A00*B00 + A00*B11 + A11*B00 + A11*B11) + (-A11*B10 + A11*B00) + (-A00*B11 - A01*B11) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    # 
    # Let me group terms:
    # A00*B00: 1
    # A00*B11: 1 - 1 = 0
    # A11*B00: 1 + 1 = 2
    # A11*B11: 1 - 1 = 0
    # A11*B10: -1 - 1 = -2
    # A01*B11: -1 + 1 = 0
    # A01*B10: 1
    # 
    # So C00 = A00*B00 + 2*A11*B00 - 2*A11*B10 + A01*B10
    # 
    # This is still not A00*B00 + A01*B10. There must be an error in the Strassen formula I'm using.
    # 
    # Let me try a different set of Strassen products. The standard Strassen algorithm is:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # 
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # I think the issue is that I'm not using the correct Strassen formula. Let me look it up properly.
    # 
    # Actually, I think the correct formula is:
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # But my calculation shows C00 is not correct. Let me try with specific values.
    # Let A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]
    # Then C = [[19, 22], [43, 50]]
    # 
    # A00=1, A01=2, A10=3, A11=4
    # B00=5, B01=6, B10=7, B11=8
    # 
    # P1 = (1+4)(5+8) = 5*13 = 65
    # P2 = (3+4)*5 = 7*5 = 35
    # P3 = 1*(6-8) = 1*(-2) = -2
    # P4 = 4*(7-5) = 4*2 = 8
    # P5 = (1+2)*8 = 3*8 = 24
    # P6 = (3-1)*(5+6) = 2*11 = 22
    # P7 = (2-4)*(7+8) = (-2)*15 = -30
    # 
    # C00 = P1 - P4 - P5 + P7 = 65 - 8 - 24 + (-30) = 65 - 8 - 24 - 30 = 3
    # But C00 should be 19. So there's definitely an error.
    # 
    # Let me try the correct Strassen formula. I think the reconstruction is:
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # But this gives C00 = 3, not 19. So the formula is wrong.
    # 
    # Let me look up the correct Strassen algorithm. The correct formula is:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # 
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # I'm getting the same formula. Let me try a different source.
    # 
    # Actually, I think the correct Strassen algorithm uses:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # 
    # And the reconstruction is:
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # But this is not working. Let me try yet another source.
    # 
    # I found that the correct Strassen algorithm is:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # 
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # 
    # I'm going in circles. Let me just use the known correct Strassen decomposition from the current program and verify it.
    
    # The current program uses:
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # Term 2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    # Term 3: A00(B01-B11) -> C01 += 1, C11 += 1
    # Term 4: A11(B10-B00) -> C00 += 1, C10 += 1
    # Term 5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    # Term 6: (A10-A00)(B00+B01) -> C11 += 1
    # Term 7: (A01-A11)(B10+B11) -> C00 += 1
    
    # Let's verify this with the example:
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # C = [[19, 22], [43, 50]]
    
    # Term 1: (1+4)(5+8) = 5*13 = 65
    #   C00 += 65, C11 += 65
    # Term 2: (3+4)*5 = 7*5 = 35
    #   C10 += 35, C11 -= 35
    # Term 3: 1*(6-8) = 1*(-2) = -2
    #   C01 += -2, C11 += -2
    # Term 4: 4*(7-5) = 4*2 = 8
    #   C00 += 8, C10 += 8
    # Term 5: (1+2)*8 = 3*8 = 24
    #   C00 -= 24, C01 += 24
    # Term 6: (3-1)*(5+6) = 2*11 = 22
    #   C11 += 22
    # Term 7: (2-4)*(7+8) = (-2)*15 = -30
    #   C00 += -30
    
    # C00 = 65 + 8 - 24 - 30 = 19 ✓
    # C01 = -2 + 24 = 22 ✓
    # C10 = 35 + 8 = 43 ✓
    # C11 = 65 - 35 - 2 + 22 = 50 ✓
    
    # Great! So the current program's Strassen decomposition is correct.
    # The reconstruction is:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Now let's fill in the matrices.
    # U[i*m+j, r] where i is row, j is column of A
    # V[j*p+k, r] where j is row, k is column of B
    # W[k*n+i, r] where k is column, i is row of C (trace convention)
    
    # Term 1: (A00+A11)(B00+B11)
    # A00: i=0, j=0 -> index 0
    # A11: i=1, j=1 -> index 3
    # B00: j=0, k=0 -> index 0
    # B11: j=1, k=1 -> index 3
    # C00: i=0, k=0 -> index 0*2+0 = 0
    # C11: i=1, k=1 -> index 1*2+1 = 3
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # Term 2: (A10+A11)B00
    # A10: i=1, j=0 -> index 2
    # A11: i=1, j=1 -> index 3
    # B00: j=0, k=0 -> index 0
    # C10: i=1, k=0 -> index 0*2+1 = 1
    # C11: i=1, k=1 -> index 1*2+1 = 3
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[1][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # Term 3: A00(B01-B11)
    # A00: i=0, j=0 -> index 0
    # B01: j=0, k=1 -> index 1
    # B11: j=1, k=1 -> index 3
    # C01: i=0, k=1 -> index 1*2+0 = 2
    # C11: i=1, k=1 -> index 1*2+1 = 3
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[2][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # Term 4: A11(B10-B00)
    # A11: i=1, j=1 -> index 3
    # B10: j=1, k=0 -> index 2
    # B00: j=0, k=0 -> index 0
    # C00: i=0, k=0 -> index 0
    # C10: i=1, k=0 -> index 1
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00
    W[1][3] = 1  # C10
    
    # Term 5: (A00+A01)B11
    # A00: i=0, j=0 -> index 0
    # A01: i=0, j=1 -> index 1
    # B11: j=1, k=1 -> index 3
    # C00: i=0, k=0 -> index 0
    # C01: i=0, k=1 -> index 2
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00
    W[2][4] = 1  # C01
    
    # Term 6: (A10-A00)(B00+B01)
    # A10: i=1, j=0 -> index 2
    # A00: i=0, j=0 -> index 0
    # B00: j=0, k=0 -> index 0
    # B01: j=0, k=1 -> index 1
    # C11: i=1, k=1 -> index 3
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11
    
    # Term 7: (A01-A11)(B10+B11)
    # A01: i=0, j=1 -> index 1
    # A11: i=1, j=1 -> index 3
    # B10: j=1, k=0 -> index 2
    # B11: j=1, k=1 -> index 3
    # C00: i=0, k=0 -> index 0
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    # Verify the time constraint
    elapsed = time.time() - start_time
    if elapsed > 145:  # Safety margin before 150s limit
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
