"""Matrix multiplication 2x5x5 via Hybrid Strassen-Schoolbook decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    # U: (10, rank)
    # V: (25, rank)
    # W: (10, rank)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_indices, v_indices, w_indices):
        """
        u_indices: list of (index, coeff) for U
        v_indices: list of (index, coeff) for V
        w_indices: list of (index, coeff) for W
        """
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        for idx, coeff in u_indices:
            U_row[idx] += coeff
        for idx, coeff in v_indices:
            V_row[idx] += coeff
        for idx, coeff in w_indices:
            W_row[idx] += coeff
            
        U_terms.append(U_row)
        V_terms.append(V_row)
        W_terms.append(W_row)

    # Helper to get index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # 1. Strassen part for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # B indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A00 + A11) * (B00 + B11)
    add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], 
             [(v_idx(0,0), 1), (v_idx(1,1), 1)], 
             [(w_idx(0,0), 1)]) # C00 gets +1
             
    # M2 = (A10 + A11) * B00
    add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], 
             [(v_idx(0,0), 1)], 
             [(w_idx(0,0), -1), (w_idx(1,0), 1)]) # C00 gets -1, C10 gets +1

    # M3 = A00 * (B01 - B11)
    add_term([(u_idx(0,0), 1)], 
             [(v_idx(0,1), 1), (v_idx(1,1), -1)], 
             [(w_idx(0,1), 1), (w_idx(1,1), 1)]) # C01 gets +1, C11 gets +1

    # M4 = A11 * (B10 - B00)
    add_term([(u_idx(1,1), 1)], 
             [(v_idx(1,0), 1), (v_idx(0,0), -1)], 
             [(w_idx(0,0), 1), (w_idx(1,0), 1)]) # C00 gets +1, C10 gets +1

    # M5 = (A00 + A01) * B11
    add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], 
             [(v_idx(1,1), 1)], 
             [(w_idx(0,0), -1), (w_idx(0,1), 1)]) # C00 gets -1, C01 gets +1

    # M6 = (A10 - A00) * (B00 + B01)
    add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], 
             [(v_idx(0,0), 1), (v_idx(0,1), 1)], 
             [(w_idx(1,1), 1)]) # C11 gets +1

    # M7 = (A01 - A11) * (B10 + B11)
    add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], 
             [(v_idx(1,0), 1), (v_idx(1,1), 1)], 
             [(w_idx(0,0), 1), (w_idx(1,1), -1)]) # C00 gets +1, C11 gets -1

    # Check C00: M1(1) + M2(-1) + M4(1) + M5(-1) + M7(1) = 1. Correct.
    # Check C01: M3(1) + M5(1) = 2? 
    # Wait. C01 = A00*B01 + A01*B11.
    # M3 = A00(B01-B11) = A00*B01 - A00*B11.
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11.
    # Sum M3+M5 = A00*B01 + A01*B11. Correct.
    # My W assignment for M3: C01 gets 1, C11 gets 1?
    # M3 contributes to C01 and C11?
    # M3 = A00(B01-B11). 
    # A00 is row 0. B01 is col 1 row 0. B11 is col 1 row 1.
    # Product A00*B01 contributes to C01.
    # Product A00*B11 contributes to C01? No, A00 (row 0) * B11 (row 1) -> C01?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
    # So A00*B01 is C01. A00*B11 is NOT in C01. A00*B11 is in C01? No.
    # A[0,0] * B[1,1] contributes to C[0,1]?
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
    # So A[0,0]B[1,1] is 0 in C01.
    # So M3 = A00(B01-B11) = A00*B01 - A00*B11.
    # A00*B01 is C01.
    # A00*B11 is... wait. B11 is B[1,1].
    # A[0,0] * B[1,1] does not appear in C[0,1].
    # It appears in C[0,1]? No.
    # C[0,1] uses B[0,1] and B[1,1].
    # A[0,0] multiplies B[0,1].
    # A[0,1] multiplies B[1,1].
    # So A[0,0]B[1,1] is not in C[0,1].
    # So M3 is not a valid bilinear form for the tensor?
    # Strassen works because the combinations cancel out in the final sum.
    # But each individual M_k is a product of linear forms.
    # The linear form in B for M3 is (B01 - B11).
    # This linear form multiplies A00.
    # The contribution to C is A00 * (B01 - B11).
    # This adds A00*B01 to C01 and -A00*B11 to C01?
    # No, A00*B11 is not a term in C01.
    # So how does Strassen work?
    # The term -A00*B11 must be canceled by another term.
    # In M5 = (A00+A01)B11 = A00*B11 + A01*B11.
    # So M3 + M5 = A00*B01 - A00*B11 + A00*B11 + A01*B11 = A00*B01 + A01*B11.
    # This is exactly C01.
    # So the intermediate terms can have "ghost" contributions that cancel out.
    # My W matrix must reflect the final contribution to C.
    # For M3, the product is A00*(B01-B11).
    # This product value is added to the sum for C01?
    # No, the decomposition is sum_c U_c V_c W_c.
    # The term c=3 is U3 V3 W3.
    # U3 = A00. V3 = B01-B11.
    # The product is A00(B01-B11).
    # This scalar value is multiplied by W3.
    # W3 is a vector of length 10.
    # We want this scalar to contribute to C01 and C11?
    # C01 needs +A00*B01.
    # C11 needs +A00*B01? No.
    # C11 = A00*B11 + A01*B21...
    # Let's re-verify Strassen assignments.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My indices:
    # A11=A00, A12=A01, A21=A10, A22=A11.
    # B11=B00, B12=B01, B21=B10, B22=B11.
    # C11=C00, C12=C01, C21=C10, C22=C11.
    
    # M3 = A00(B01-B11).
    # Contributes to C01 (via +M3) and C11 (via +M3).
    # So W for M3 should be [0, 1, 0, 1, ...] (indices 1 and 3).
    # C01 index 1. C11 index 3.
    # So my previous W assignment for M3 was correct: [(w_idx(0,1), 1), (w_idx(1,1), 1)].
    
    # Let's re-verify M5.
    # M5 = (A00+A01)B11.
    # Contributes to C00 (via -M5) and C01 (via +M5).
    # W for M5: [(w_idx(0,0), -1), (w_idx(0,1), 1)].
    # Previous: [(w_idx(0,0), -1), (w_idx(0,1), 1)]. Correct.
    
    # Let's re-verify M7.
    # M7 = (A01-A11)(B10+B11).
    # Contributes to C00 (via +M7) and C11 (via -M7).
    # W for M7: [(w_idx(0,0), 1), (w_idx(1,1), -1)].
    # Previous: [(w_idx(0,0), 1), (w_idx(1,1), -1)]. Correct.

    # 2. Schoolbook for A[0:2, 0:2] * B[0:2, 2:5]
    # A rows 0,1. Cols 0,1.
    # B rows 0,1. Cols 2,3,4.
    # C cols 2,3,4.
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term([(u_idx(i,j), 1)], 
                         [(v_idx(j,k), 1)], 
                         [(w_idx(k,i), 1)])

    # 3. Schoolbook for A[0:2, 2:5] * B[2:5, 0:2]
    # A rows 0,1. Cols 2,3,4.
    # B rows 2,3,4. Cols 0,1.
    # C cols 0,1.
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_term([(u_idx(i,j), 1)], 
                         [(v_idx(j,k), 1)], 
                         [(w_idx(k,i), 1)])

    # 4. Schoolbook for A[0:2, 2:5] * B[2:5, 2:5]
    # A rows 0,1. Cols 2,3,4.
    # B rows 2,3,4. Cols 2,3,4.
    # C cols 2,3,4.
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term([(u_idx(i,j), 1)], 
                         [(v_idx(j,k), 1)], 
                         [(w_idx(k,i), 1)])

    rank = len(U_terms)
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
