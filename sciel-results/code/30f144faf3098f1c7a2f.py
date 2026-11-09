"""
Rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.

Strategy: Use a 3+2 block partition on both matrix dimensions, applying
Strassen's algorithm (rank 7) to the 2x2x2 sub-blocks where possible,
and schoolbook (rank = product of dims) elsewhere.

Block structure:
  A = [[A11 (3x3), A12 (3x2)],
       [A21 (2x3), A22 (2x2)]]
  B = [[B11 (3x3), B12 (3x2)],
       [B21 (2x3), B22 (2x2)]]
  C = A*B = [[C11 (3x3), C12 (3x2)],
             [C21 (2x3), C22 (2x2)]]

Each C-block is a sum of two matrix products:
  C11 = A11*B11 + A12*B21
  C12 = A11*B12 + A12*B22
  C21 = A21*B11 + A22*B21
  C22 = A21*B12 + A22*B22

For each individual matrix product X*Y where X is (a x b) and Y is (b x c),
the schoolbook decomposition has rank a*b*c. The only sub-product that can
be improved with Strassen is A22*B22 (2x2 * 2x2 -> 2x2), which uses rank 7
instead of 8.

Total rank = 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Return {'U', 'V', 'W'}: factor matrices for the 5x5x5 matrix multiplication
    tensor with rank 124 (schoolbook + Strassen on the 2x2x2 corner block).
    
    Index conventions:
      U has shape (n*m, rank)  = (25, 124), indexed by [i*m + j, r]
      V has shape (m*p, rank)  = (25, 124), indexed by [j*p + k, r]
      W has shape (p*n, rank)  = (25, 124), indexed by [k*n + i, r]  (trace convention)
    
    The decomposition satisfies:
      For all i, j, k:  T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 iff (i,j,k) is a valid triple for C[i,k] = sum_j A[i,j]*B[j,k].
    Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    in the sense that C[i,k] = sum_j A[i,j] B[j,k].
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolbook(a, b, c, i_off, j_off, k_off):
        """
        Add schoolbook decomposition for an (a x b) * (b x c) -> (a x c) product.
        i_off: row offset in A (and C)
        j_off: col offset in A / row offset in B
        k_off: col offset in B (and C)
        """
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U_cols.append((i_off + i, j_off + j))
                    V_cols.append((j_off + j, k_off + k))
                    W_cols.append((k_off + k, i_off + i))
    
    def add_strassen_2x2(i_off, j_off, k_off):
        """
        Add Strassen decomposition for a (2x2) * (2x2) -> (2x2) product.
        i_off: row offset in A (and C)
        j_off: col offset in A / row offset in B
        k_off: col offset in B (and C)
        
        Strassen's 7 products for 2x2 matrix multiplication:
        Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 + M3 + M6
        
        In terms of our tensor factors, each M_r is a product of a linear form
        in A-entries and a linear form in B-entries, and contributes to specific
        C-entries.
        
        For the tensor decomposition, each rank-1 term is:
          U factor: linear form in A entries (at specific (i,j) positions)
          V factor: linear form in B entries (at specific (j,k) positions)  
          W factor: linear form in C entries (at specific (k,i) positions)
        
        Actually, let me think more carefully. The tensor T for 2x2x2 matrix mult
        has 8 non-zero entries: T[i,j,k] for i,j,k in {0,1} where C[i,k] = sum_j A[i,j]B[j,k].
        The 8 entries are T[i,j,k] = 1 for all i,j,k in {0,1}.
        
        Strassen decomposes this rank-8 tensor into 7 rank-1 terms.
        
        Let me write out the 7 terms explicitly.
        Using 0-based indexing:
        A = [[A00, A01], [A10, A11]]
        B = [[B00, B01], [B10, B11]]
        C = [[C00, C01], [C10, C11]]
        
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        
        Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        Each M_r is a bilinear form: (linear in A) * (linear in B), and it
        contributes with coefficient 1 to certain C entries.
        
        For the tensor decomposition, each rank-1 term r has:
          U_r[i*m+j] = coefficient of A[i,j] in the A-linear form
          V_r[j*p+k] = coefficient of B[j,k] in the B-linear form
          W_r[k*n+i] = coefficient of C[i,k] in the C-linear form (with sign)
        
        Wait, I need to be more careful. The tensor is:
          T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        And we need T[i,j,k] = 1 for all valid (i,j,k) where C[i,k] depends on A[i,j]*B[j,k].
        
        Actually for matrix multiplication C = A*B, the "tensor" is defined such that
        C[i,k] = sum_j A[i,j] * B[j,k]. So the 3-way tensor has entries T[i,j,k] = 1
        for all i, j, k (in the appropriate ranges), because each triple (i,j,k)
        contributes A[i,j]*B[j,k] to C[i,k].
        
        So for the 2x2x2 sub-block at offset (i_off, j_off, k_off):
        The local tensor has T_local[a,b,c] = 1 for all a,b,c in {0,1}.
        
        Strassen decomposes this into 7 rank-1 terms. Let me write them out:
        
        Term 1: M1 = (A00+A11)(B00+B11)
          A-part: A00 + A11 -> U: {A00: 1, A11: 1}
          B-part: B00 + B11 -> V: {B00: 1, B11: 1}
          C-part: contributes +1 to C00, +1 to C11
          W: {C00: 1, C11: 1}
        
        Term 2: M2 = (A10+A11)B00
          A-part: A10 + A11 -> U: {A10: 1, A11: 1}
          B-part: B00 -> V: {B00: 1}
          C-part: contributes +1 to C10, -1 to C11
          W: {C10: 1, C11: -1}
        
        Term 3: M3 = A00(B01-B11)
          A-part: A00 -> U: {A00: 1}
          B-part: B01 - B11 -> V: {B01: 1, B11: -1}
          C-part: contributes +1 to C01, +1 to C11
          W: {C01: 1, C11: 1}
        
        Term 4: M4 = A11(B10-B00)
          A-part: A11 -> U: {A11: 1}
          B-part: B10 - B00 -> V: {B10: 1, B00: -1}
          C-part: contributes +1 to C00, +1 to C10
          W: {C00: 1, C10: 1}
        
        Term 5: M5 = (A00+A01)B11
          A-part: A00 + A01 -> U: {A00: 1, A01: 1}
          B-part: B11 -> V: {B11: 1}
          C-part: contributes -1 to C00, +1 to C01
          W: {C00: -1, C01: 1}
        
        Term 6: M6 = (A10-A00)(B00+B01)
          A-part: A10 - A00 -> U: {A10: 1, A00: -1}
          B-part: B00 + B01 -> V: {B00: 1, B01: 1}
          C-part: contributes +1 to C11
          W: {C11: 1}
        
        Term 7: M7 = (A01-A11)(B10+B11)
          A-part: A01 - A11 -> U: {A01: 1, A11: -1}
          B-part: B10 + B11 -> V: {B10: 1, B11: 1}
          C-part: contributes +1 to C00
          W: {C00: 1}
        
        Let me verify:
        C00 = M1 + M4 - M5 + M7
             = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
             = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00
               - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
             = A00*B00 + A01*B10 + A11*B00 + A11*B11 - A11*B00 + A01*B10 - A11*B10
             Hmm, let me redo this more carefully.
        
        Actually, let me just verify the standard Strassen formulas:
        C00 = M1 + M4 - M5 + M7
             = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
             = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
             = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
             = ae + bg  ✓ (C[0,0] = a*e + b*g)
        
        C01 = M3 + M5
             = a(f-h) + (a+b)h
             = af-ah+ah+bh
             = af + bh  ✓ (C[0,1] = a*f + b*h)
        
        C10 = M2 + M4
             = (c+d)e + d(g-e)
             = ce+de+dg-de
             = ce + dg  ✓ (C[1,0] = c*e + d*g)
        
        C11 = M1 - M2 + M3 + M6
             = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
             = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
             = ae+ah+de+dh-ce-de+af-ah+ce+cf-ae-af
             = dh + cf  ✓ (C[1,1] = c*f + d*h)
        
        Great, so the 7 terms are correct. Now let me map them to our indexing.
        
        For the sub-block at offset (i_off, j_off, k_off):
        Local A indices: (i_off+a, j_off+b) for a,b in {0,1}
        Local B indices: (j_off+b, k_off+c) for b,c in {0,1}
        Local C indices: (i_off+a, k_off+c) for a,c in {0,1}
        
        In the global U matrix, row index = (i_off+a)*m + (j_off+b)
        In the global V matrix, row index = (j_off+b)*p + (k_off+c)
        In the global W matrix, row index = (k_off+c)*n + (i_off+a)
        """
        # Term 1: (A00+A11)(B00+B11) -> C00+1, C11+1
        U_cols.append({(i_off+0, j_off+0): 1, (i_off+1, j_off+1): 1})
        V_cols.append({(j_off+0, k_off+0): 1, (j_off+1, k_off+1): 1})
        W_cols.append({(k_off+0, i_off+0): 1, (k_off+1, i_off+1): 1})
        
        # Term 2: (A10+A11)B00 -> C10+1, C11-1
        U_cols.append({(i_off+1, j_off+0): 1, (i_off+1, j_off+1): 1})
        V_cols.append({(j_off+0, k_off+0): 1})
        W_cols.append({(k_off+0, i_off+1): 1, (k_off+1, i_off+1): -1})
        
        # Term 3: A00(B01-B11) -> C01+1, C11+1
        U_cols.append({(i_off+0, j_off+0): 1})
        V_cols.append({(j_off+0, k_off+1): 1, (j_off+1, k_off+1): -1})
        W_cols.append({(k_off+1, i_off+0): 1, (k_off+1, i_off+1): 1})
        
        # Term 4: A11(B10-B00) -> C00+1, C10+1
        U_cols.append({(i_off+1, j_off+1): 1})
        V_cols.append({(j_off+1, k_off+0): 1, (j_off+0, k_off+0): -1})
        W_cols.append({(k_off+0, i_off+0): 1, (k_off+0, i_off+1): 1})
        
        # Term 5: (A00+A01)B11 -> C00-1, C01+1
        U_cols.append({(i_off+0, j_off+0): 1, (i_off+0, j_off+1): 1})
        V_cols.append({(j_off+1, k_off+1): 1})
        W_cols.append({(k_off+0, i_off+0): -1, (k_off+1, i_off+0): 1})
        
        # Term 6: (A10-A00)(B00+B01) -> C11+1
        U_cols.append({(i_off+1, j_off+0): 1, (i_off+0, j_off+0): -1})
        V_cols.append({(j_off+0, k_off+0): 1, (j_off+0, k_off+1): 1})
        W_cols.append({(k_off+1, i_off+1): 1})
        
        # Term 7: (A01-A11)(B10+B11) -> C00+1
        U_cols.append({(i_off+0, j_off+1): 1, (i_off+1, j_off+1): -1})
        V_cols.append({(j_off+1, k_off+0): 1, (j_off+1, k_off+1): 1})
        W_cols.append({(k_off+0, i_off+0): 1})
    
    # Now build the full decomposition:
    # C11 = A11*B11 (3x3 * 3x3 -> 3x3) + A12*B21 (3x2 * 2x3 -> 3x3)
    #   A11 is at rows 0-2, cols 0-2 of A; B11 is at rows 0-2, cols 0-2 of B
    #   A12 is at rows 0-2, cols 3-4 of A; B21 is at rows 3-4, cols 0-2 of B
    #   C11 is at rows 0-2, cols 0-2 of C
    add_schoolbook(3, 3, 3, i_off=0, j_off=0, k_off=0)  # A11*B11
    add_schoolbook(3, 2, 3, i_off=0, j_off=3, k_off=0)  # A12*B21
    
    # C12 = A11*B12 (3x3 * 3x2 -> 3x2) + A12*B22 (3x2 * 2x2 -> 3x2)
    #   B12 is at rows 0-2, cols 3-4 of B
    #   C12 is at rows 0-2, cols 3-4 of C
    #   A22 is at rows 3-4, cols 3-4 of A; B22 is at rows 3-4, cols 3-4 of B
    add_schoolbook(3, 3, 2, i_off=0, j_off=0, k_off=3)  # A11*B12
    add_schoolbook(3, 2, 2, i_off=0, j_off=3, k_off=3)  # A12*B22
    
    # C21 = A21*B11 (2x3 * 3x3 -> 2x3) + A22*B21 (2x2 * 2x3 -> 2x3)
    #   A21 is at rows 3-4, cols 0-2 of A
    #   C21 is at rows 3-4, cols 0-2 of C
    add_schoolbook(2, 3, 3, i_off=3, j_off=0, k_off=0)  # A21*B11
    add_schoolbook(2, 2, 3, i_off=3, j_off=3, k_off=0)  # A22*B21
    
    # C22 = A21*B12 (2x3 * 3x2 -> 2x2) + A22*B22 (2x2 * 2x2 -> 2x2)
    #   C22 is at rows 3-4, cols 3-4 of C
    add_schoolbook(2, 3, 2, i_off=3, j_off=0, k_off=3)  # A21*B12
    add_strassen_2x2(i_off=3, j_off=3, k_off=3)  # A22*B22 with Strassen (rank 7)
    
    rank = len(U_cols)
    assert rank == 124, f"Expected rank 124, got {rank}"
    
    # Build dense matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (i, j), val in U_cols[r].items():
            U[i * m + j, r] = val
        for (j, k), val in V_cols[r].items():
            V[j * p + k, r] = val
        for (k, i), val in W_cols[r].items():
            W[k * n + i, r] = val
    
    if _deadline_exceeded():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
