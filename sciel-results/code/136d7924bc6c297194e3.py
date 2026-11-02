import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    
    # We will construct a decomposition of rank 27.
    # Split j into j1 (0,1) and j2 (2).
    # Part 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x7)
    # Split k into 3 blocks of 2 and 1 block of 1.
    # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:7]
    
    # Part 2: A[:, 2:3] * B[2:3, :] (2x1 * 1x7)
    # This is 2 terms.
    
    U = np.zeros((n * m, 27))
    V = np.zeros((m * p, 27))
    W = np.zeros((p * n, 27))
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We will map these to the global indices.
    # U index: i*m + j. For Part 1, j is 0 or 1.
    # V index: j*p + k. For Part 1, j is 0 or 1.
    # W index: k*n + i.
    
    def add_strassen_block(u_offset, v_offset, w_offset, r_start):
        # u_offset: index in U for i=0, j=0 is u_offset. i=1, j=0 is u_offset+2?
        # No, U index is i*m + j.
        # For j=0: i=0 -> 0, i=1 -> 2.
        # For j=1: i=0 -> 1, i=1 -> 3.
        # So U indices for A block are 0, 1, 2, 3.
        # V index is j*p + k.
        # For j=0: k=0..1 -> 0, 1. k=2..3 -> 2, 3.
        # For j=1: k=0..1 -> 7, 8. k=2..3 -> 9, 10.
        # W index is k*n + i.
        # For k=0: i=0 -> 0, i=1 -> 1.
        # For k=1: i=0 -> 2, i=1 -> 3.
        
        # We need to define the mapping for a specific block of B.
        # Let's say we are handling B[j=0..1, k=k0..k0+1].
        # A is always the same 2x2 block (j=0..1).
        
        # Actually, the Strassen algorithm reuses A.
        # We can't just add 7 terms per block independently because the A coefficients are shared?
        # No, in the tensor decomposition, each term is independent.
        # We are adding 7 rank-1 tensors for each 2x2 block of B.
        # The A part is the same 2x2 block.
        
        # Let's define the 7 terms for a 2x2 * 2x2 multiplication.
        # The A indices are fixed: (0,0), (0,1), (1,0), (1,1).
        # The B indices are variable: (j, k0), (j, k0+1), (j+1, k0), (j+1, k0+1).
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 (idx 0), A11 (idx 3).
        # V: B00 (idx j*p+k0), B11 (idx (j+1)*p+k0+1).
        # W: C00 (idx k0*n+0), C11 (idx (k0+1)*n+1).
        
        # We need to be careful with the W indices.
        # The output C is 2x7.
        # For a block of B at k0, the output is C[:, k0] and C[:, k0+1].
        
        # Let's write a helper to add a term.
        def add_term(u_indices, v_indices, w_indices, coeffs, r):
            for idx, coeff in zip(u_indices, coeffs):
                U[idx, r] += coeff
            for idx, coeff in zip(v_indices, coeffs):
                V[idx, r] += coeff
            for idx, coeff in zip(w_indices, coeffs):
                W[idx, r] += coeff

        # Strassen Terms for 2x2 * 2x2
        # A indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
        # B indices for block starting at k0, j=0:
        # B00: 0*p + k0 = k0
        # B01: 0*p + (k0+1) = k0+1
        # B10: 1*p + k0 = 7 + k0
        # B11: 1*p + (k0+1) = 7 + k0 + 1
        
        # W indices for output C at k0, k0+1:
        # C00: k0*n + 0 = 2*k0
        # C01: k0*n + 1 = 2*k0 + 1
        # C10: (k0+1)*n + 0 = 2*k0 + 2
        # C11: (k0+1)*n + 1 = 2*k0 + 3
        
        # Term 1: (A00+A11)(B00+B11)
        # U: 0:1, 3:1
        # V: k0:1, 7+k0+1:1
        # W: 2*k0:1, 2*k0+3:1
        # This term contributes to C00 and C11.
        # But Strassen M1 is used to compute C00 and C11 via combinations.
        # The decomposition is a sum of products.
        # We need to express C = sum M_r.
        # C00 = M1 + M4 - M5 - M7
        # C11 = M1 - M2 + M3 + M6
        # C01 = M3 + M5 + M7
        # C10 = M2 - M4 - M6
        
        # This means we cannot just add the rank-1 tensors directly.
        # We need to assign the coefficients in W to match the linear combinations.
        # Actually, the tensor T[i, j, k] is the coefficient of A[i,j]B[j,k] in C[i,k].
        # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k].
        # For Strassen, the "multiplications" are the M_r.
        # The entries of U, V, W are the coefficients of A and B in M_r.
        # The entries of W are the coefficients of M_r in C.
        
        # So for Term 1 (M1):
        # U[0, r] = 1 (A00), U[3, r] = 1 (A11)
        # V[k0, r] = 1 (B00), V[7+k0+1, r] = 1 (B11)
        # W[2*k0, r] = 1 (C00 coeff), W[2*k0+3, r] = 1 (C11 coeff)
        # But C00 = M1 + ... and C11 = M1 + ...
        # So W[2*k0, r] should be 1, and W[2*k0+3, r] should be 1.
        # This works if we add other terms with negative coefficients in W.
        
        add_term([0, 3], [k0, 7+k0+1], [2*k0, 2*k0+3], [1, 1, 1, 1], r_start)
        
        # Term 2: (A10+A11)B00
        # U: 2:1, 3:1
        # V: k0:1
        # W: 2*k0+2:1 (C10), 2*k0+3:1 (C11)?
        # C10 = M2 - M4 - M6. So W[2*k0+2, r] = 1.
        # C11 = M1 - M2 + M3 + M6. So W[2*k0+3, r] = -1.
        add_term([2, 3], [k0], [2*k0+2, 2*k0+3], [1, 1, 1, -1], r_start + 1)
        
        # Term 3: A00(B01-B11)
        # U: 0:1
        # V: k0+1:1, 7+k0+1:-1
        # W: 2*k0:1 (C00), 2*k0+1:1 (C01)?
        # C00 = M1 + M4 - M5 - M7. M3 not in C00.
        # C01 = M3 + M5 + M7. So W[2*k0+1, r] = 1.
        # C11 = M1 - M2 + M3 + M6. So W[2*k0+3, r] = 1.
        add_term([0], [k0+1, 7+k0+1], [2*k0+1, 2*k0+3], [1, 1, -1, 1, 1], r_start + 2)
        # Wait, U has 1 entry, V has 2, W has 2. Coeffs must match length?
        # My add_term assumes parallel lists.
        # U: [0] -> [1]
        # V: [k0+1, 7+k0+1] -> [1, -1]
        # W: [2*k0+1, 2*k0+3] -> [1, 1]
        
        # Let's re-verify the Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5 + M7
        # C10 = M2 - M4 - M6
        # C11 = M1 - M2 + M3 + M6
        
        # Term 3 (M3):
        # U: A00 (0) -> 1
        # V: B01 (k0+1) -> 1, B11 (7+k0+1) -> -1
        # W: C01 (2*k0+1) -> 1, C11 (2*k0+3) -> 1
        
        # Term 4 (M4): A11(B10-B00)
        # U: A11 (3) -> 1
        # V: B10 (7+k0) -> 1, B00 (k0) -> -1
        # W: C00 (2*k0) -> 1, C10 (2*k0+2) -> -1
        
        # Term 5 (M5): (A00+A01)B11
        # U: A00 (0) -> 1, A01 (1) -> 1
        # V: B11 (7+k0+1) -> 1
        # W: C00 (2*k0) -> -1, C01 (2*k0+1) -> 1
        
        # Term 6 (M6): (A10-A00)(B00+B01)
        # U: A10 (2) -> 1, A00 (0) -> -1
        # V: B00 (k0) -> 1, B01 (k0+1) -> 1
        # W: C10 (2*k0+2) -> -1, C11 (2*k0+3) -> 1
        
        # Term 7 (M7): (A01-A11)(B10+B11)
        # U: A01 (1) -> 1, A11 (3) -> -1
        # V: B10 (7+k0) -> 1, B11 (7+k0+1) -> 1
        # W: C00 (2*k0) -> -1, C01 (2*k0+1) -> 1
        
        # Let's implement these carefully.
        
        # We already added Term 1 and 2. Let's restart the function logic in the main loop.
        pass

    # Correct Implementation:
    U = np.zeros((n * m, 27))
    V = np.zeros((m * p, 27))
    W = np.zeros((p * n, 27))
    
    r = 0
    # 3 Strassen blocks for k=0..1, 2..3, 4..5
    for k0 in [0, 2, 4]:
        # Term 1
        U[0, r] = 1; U[3, r] = 1
        V[k0, r] = 1; V[7+k0+1, r] = 1
        W[2*k0, r] = 1; W[2*k0+3, r] = 1
        r += 1
        
        # Term 2
        U[2, r] = 1; U[3, r] = 1
        V[k0, r] = 1
        W[2*k0+2, r] = 1; W[2*k0+3, r] = -1
        r += 1
        
        # Term 3
        U[0, r] = 1
        V[k0+1, r] = 1; V[7+k0+1, r] = -1
        W[2*k0+1, r] = 1; W[2*k0+3, r] = 1
        r += 1
        
        # Term 4
        U[3, r] = 1
        V[7+k0, r] = 1; V[k0, r] = -1
        W[2*k0, r] = 1; W[2*k0+2, r] = -1
        r += 1
        
        # Term 5
        U[0, r] = 1; U[1, r] = 1
        V[7+k0+1, r] = 1
        W[2*k0, r] = -1; W[2*k0+1, r] = 1
        r += 1
        
        # Term 6
        U[2, r] = 1; U[0, r] = -1
        V[k0, r] = 1; V[k0+1, r] = 1
        W[2*k0+2, r] = -1; W[2*k0+3, r] = 1
        r += 1
        
        # Term 7
        U[1, r] = 1; U[3, r] = -1
        V[7+k0, r] = 1; V[7+k0+1, r] = 1
        W[2*k0, r] = -1; W[2*k0+1, r] = 1
        r += 1

    # 1 Schoolbook block for k=6
    # A 2x2 * B 2x1
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # B indices: k=6.
    # B00: 0*7+6 = 6
    # B10: 1*7+6 = 13
    
    # Term 8: A00 B00
    U[0, r] = 1
    V[6, r] = 1
    W[2*6, r] = 1 # C0
    r += 1
    
    # Term 9: A01 B10
    U[1, r] = 1
    V[13, r] = 1
    W[2*6, r] = 1 # C0
    r += 1
    
    # Term 10: A10 B00
    U[2, r] = 1
    V[6, r] = 1
    W[2*6+1, r] = 1 # C1
    r += 1
    
    # Term 11: A11 B10
    U[3, r] = 1
    V[13, r] = 1
    W[2*6+1, r] = 1 # C1
    r += 1
    
    # Part 2: A[:, 2] * B[2, :]
    # A indices: i=0, j=2 -> 0*3+2=2. i=1, j=2 -> 1*3+2=5.
    # B indices: j=2, k=0..6 -> 2*7+k = 14..20.
    # W indices: k*2+i.
    
    # Term 12: i=0 part
    U[2, r] = 1
    for k in range(7):
        V[14+k, r] = 1
        W[2*k+0, r] = 1
    r += 1
    
    # Term 13: i=1 part
    U[5, r] = 1
    for k in range(7):
        V[14+k, r] = 1
        W[2*k+1, r] = 1
    r += 1
    
    # Total r = 21 + 2 = 23?
    # 3 Strassen blocks * 7 = 21.
    # 1 Schoolbook block * 4 = 4.
    # 2 Outer product terms = 2.
    # Total 27.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
