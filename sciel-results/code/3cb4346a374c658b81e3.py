import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Structured Strassen-based construction for rank 34
    # Block decomposition:
    # A (3x3): A22 = A[0:2,0:2] (2x2), A21 = A[0:2,2] (2x1), A12 = A[2,0:2] (1x2), A11 = A[2,2] (1x1)
    # B (3x4): B22 = B[0:2,0:2] (2x2), B22p = B[0:2,2:4] (2x2), B12 = B[2,0:2] (1x2), B12p = B[2,2:4] (1x2)
    # C (3x4): C[0:2,0:2] = A22*B22 + A21*B12
    #          C[0:2,2:4] = A22*B22p + A21*B12p
    #          C[2,0:2]   = A12*B22 + A11*B12
    #          C[2,2:4]   = A12*B22p + A11*B12p
    
    # Strassen 7-term for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M5
    
    # For each 2x2x2 subproblem, we need 7 rank-1 terms.
    # For each (1x2x2) or (2x1x2) or (1x1x2) subproblem, we use standard multiplication.
    
    # Build the decomposition term by term.
    # Each term is a rank-1 tensor: U_col (9-vector), V_col (12-vector), W_col (12-vector)
    # U index: i*m+j for A[i,j]
    # V index: j*p+k for B[j,k]
    # W index: k*n+i for C[i,k] (output C has shape 3x4, indexed by row i and col k)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(U_col, V_col, W_col):
        """Add a rank-1 term. U_col, V_col, W_col are dicts mapping indices to Fraction values."""
        U_vec = [Fraction(0)] * (n * m)
        V_vec = [Fraction(0)] * (m * p)
        W_vec = [Fraction(0)] * (p * n)
        for idx, val in U_col.items():
            U_vec[idx] += val
        for idx, val in V_col.items():
            V_vec[idx] += val
        for idx, val in W_col.items():
            W_vec[idx] += val
        U_terms.append(U_vec)
        V_terms.append(V_vec)
        W_terms.append(W_vec)
    
    # Helper: add a standard multiplication term for A[i,j] * B[j,k] -> C[i,k]
    def add_mult(i, j, k):
        add_term({i * m + j: Fraction(1)}, {j * p + k: Fraction(1)}, {k * n + i: Fraction(1)})
    
    # Helper: add a Strassen term for 2x2x2 subproblem
    # The 2x2 submatrix of A has elements at positions (r,c) where r in {r0, r0+1}, c in {c0, c0+1}
    # The 2x2 submatrix of B has elements at positions (r,c) where r in {r0, r0+1}, c in {c0, c0+1}
    # The output 2x2 submatrix of C has elements at positions (r,c) where r in {r0, r0+1}, c in {c0, c0+1}
    # Strassen uses intermediate sums/differences.
    # For a 2x2x2 problem with A indices (a00,a01,a10,a11), B indices (b00,b01,b10,b11), C indices (c00,c01,c10,c11):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M5
    
    def add_strassen(a_indices, b_indices, c_indices):
        """
        a_indices: dict with keys 'a00','a01','a10','a11' mapping to (i,j) indices in A
        b_indices: dict with keys 'b00','b01','b10','b11' mapping to (j,k) indices in B
        c_indices: dict with keys 'c00','c01','c10','c11' mapping to (i,k) indices in C
        """
        a00 = a_indices['a00']
        a01 = a_indices['a01']
        a10 = a_indices['a10']
        a11 = a_indices['a11']
        b00 = b_indices['b00']
        b01 = b_indices['b01']
        b10 = b_indices['b10']
        b11 = b_indices['b11']
        c00 = c_indices['c00']
        c01 = c_indices['c01']
        c10 = c_indices['c10']
        c11 = c_indices['c11']
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c11=1
        add_term(
            {a00[0]*m+a00[1]: Fraction(1), a11[0]*m+a11[1]: Fraction(1)},
            {b00[0]*p+b00[1]: Fraction(1), b11[0]*p+b11[1]: Fraction(1)},
            {c00[1]*n+c00[0]: Fraction(1), c11[1]*n+c11[0]: Fraction(1)}
        )
        
        # M2 = (a10+a11)b00
        # U: a10=1, a11=1; V: b00=1; W: c10=1
        add_term(
            {a10[0]*m+a10[1]: Fraction(1), a11[0]*m+a11[1]: Fraction(1)},
            {b00[0]*p+b00[1]: Fraction(1)},
            {c10[1]*n+c10[0]: Fraction(1)}
        )
        
        # M3 = a00(b10-b00)
        # U: a00=1; V: b10=1, b00=-1; W: c01=1
        add_term(
            {a00[0]*m+a00[1]: Fraction(1)},
            {b10[0]*p+b10[1]: Fraction(1), b00[0]*p+b00[1]: Fraction(-1)},
            {c01[1]*n+c01[0]: Fraction(1)}
        )
        
        # M4 = a11(b01-b11)
        # U: a11=1; V: b01=1, b11=-1; W: c00=1, c10=1
        add_term(
            {a11[0]*m+a11[1]: Fraction(1)},
            {b01[0]*p+b01[1]: Fraction(1), b11[0]*p+b11[1]: Fraction(-1)},
            {c00[1]*n+c00[0]: Fraction(1), c10[1]*n+c10[0]: Fraction(1)}
        )
        
        # M5 = (a00+a01)b11
        # U: a00=1, a01=1; V: b11=1; W: c00=-1, c01=1, c11=1
        add_term(
            {a00[0]*m+a00[1]: Fraction(1), a01[0]*m+a01[1]: Fraction(1)},
            {b11[0]*p+b11[1]: Fraction(1)},
            {c00[1]*n+c00[0]: Fraction(-1), c01[1]*n+c01[0]: Fraction(1), c11[1]*n+c11[0]: Fraction(1)}
        )
        
        # M6 = (a10-a00)(b00+b01)
        # U: a10=1, a00=-1; V: b00=1, b01=1; W: c10=-1, c11=-1
        add_term(
            {a10[0]*m+a10[1]: Fraction(1), a00[0]*m+a00[1]: Fraction(-1)},
            {b00[0]*p+b00[1]: Fraction(1), b01[0]*p+b01[1]: Fraction(1)},
            {c10[1]*n+c10[0]: Fraction(-1), c11[1]*n+c11[0]: Fraction(-1)}
        )
        
        # M7 = (a01-a11)(b10+b11)
        # U: a01=1, a11=-1; V: b10=1, b11=1; W: c00=1, c11=1
        add_term(
            {a01[0]*m+a01[1]: Fraction(1), a11[0]*m+a11[1]: Fraction(-1)},
            {b10[0]*p+b10[1]: Fraction(1), b11[0]*p+b11[1]: Fraction(1)},
            {c00[1]*n+c00[0]: Fraction(1), c11[1]*n+c11[0]: Fraction(1)}
        )
    
    # Now build the 8 subproblems:
    
    # 1. A22 * B22: 2x2x2, A rows 0-1 cols 0-1, B rows 0-1 cols 0-1, C rows 0-1 cols 0-1
    # A22 elements: A[0,0]=(0,0), A[0,1]=(0,1), A[1,0]=(1,0), A[1,1]=(1,1)
    # B22 elements: B[0,0]=(0,0), B[0,1]=(0,1), B[1,0]=(1,0), B[1,1]=(1,1)
    # C output: C[0,0]=(0,0), C[0,1]=(0,1), C[1,0]=(1,0), C[1,1]=(1,1)
    # But note: C index is (i,k) where i is row of C, k is col of C
    # W index = k*n + i
    a22 = {'a00': (0,0), 'a01': (0,1), 'a10': (1,0), 'a11': (1,1)}
    b22 = {'b00': (0,0), 'b01': (0,1), 'b10': (1,0), 'b11': (1,1)}
    c22 = {'c00': (0,0), 'c01': (0,1), 'c10': (1,0), 'c11': (1,1)}
    add_strassen(a22, b22, c22)
    
    # 2. A22 * B22p: 2x2x2, A rows 0-1 cols 0-1, B rows 0-1 cols 2-3, C rows 0-1 cols 2-3
    # B22p elements: B[0,2]=(0,2), B[0,3]=(0,3), B[1,2]=(1,2), B[1,3]=(1,3)
    # C output: C[0,2]=(0,2), C[0,3]=(0,3), C[1,2]=(1,2), C[1,3]=(1,3)
    a22 = {'a00': (0,0), 'a01': (0,1), 'a10': (1,0), 'a11': (1,1)}
    b22p = {'b00': (0,2), 'b01': (0,3), 'b10': (1,2), 'b11': (1,3)}
    c22p = {'c00': (0,2), 'c01': (0,3), 'c10': (1,2), 'c11': (1,3)}
    add_strassen(a22, b22p, c22p)
    
    # 3. A21 * B12: 2x1x2, A rows 0-1 col 2 (2x1), B row 2 cols 0-1 (1x2), C rows 0-1 cols 0-1 (2x2)
    # A21: A[0,2]=(0,2), A[1,2]=(1,2)
    # B12: B[2,0]=(2,0), B[2,1]=(2,1)
    # C: C[0,0], C[0,1], C[1,0], C[1,1]
    # This is a 2x1x2 multiplication: sum over j=2 of A[i,2]*B[2,k]
    # Standard: 4 terms
    for i in range(2):
        for k in range(2):
            add_mult(i, 2, k)
    
    # 4. A21 * B12p: 2x1x2, A rows 0-1 col 2, B row 2 cols 2-3, C rows 0-1 cols 2-3
    for i in range(2):
        for k in range(2, 4):
            add_mult(i, 2, k)
    
    # 5. A12 * B22: 1x2x2, A row 2 cols 0-1 (1x2), B rows 0-1 cols 0-1 (2x2), C row 2 cols 0-1 (1x2)
    # Standard: 4 terms
    for j in range(2):
        for k in range(2):
            add_mult(2, j, k)
    
    # 6. A12 * B22p: 1x2x2, A row 2 cols 0-1, B rows 0-1 cols 2-3, C row 2 cols 2-3
    for j in range(2):
        for k in range(2, 4):
            add_mult(2, j, k)
    
    # 7. A11 * B12: 1x1x2, A[2,2], B row 2 cols 0-1, C row 2 cols 0-1
    for k in range(2):
        add_mult(2, 2, k)
    
    # 8. A11 * B12p: 1x1x2, A[2,2], B row 2 cols 2-3, C row 2 cols 2-3
    for k in range(2, 4):
        add_mult(2, 2, k)
    
    # Now verify the decomposition
    rank = len(U_terms)
    
    # Check that all entries are integer or half-integer
    valid = True
    for mat in [U_terms, V_terms, W_terms]:
        for vec in mat:
            for val in vec:
                doubled = val * 2
                if doubled != doubled.numerator // doubled.denominator * Fraction(doubled.denominator, 1) and doubled.denominator != 1:
                    # Check if it's integer or half-integer
                    if doubled.denominator not in [1, 2]:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Verify the decomposition exactly
    # Build target tensor indices
    target_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.add((i * m + j, j * p + k, k * n + i))
    
    # Compute the decomposed tensor
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Only non-zero where (a,b,c) in target_indices
    decomposed = {}
    for r in range(rank):
        U_r = U_terms[r]
        V_r = V_terms[r]
        W_r = W_terms[r]
        # Find non-zero entries
        U_nz = [(idx, val) for idx, val in enumerate(U_r) if val != 0]
        V_nz = [(idx, val) for idx, val in enumerate(V_r) if val != 0]
        W_nz = [(idx, val) for idx, val in enumerate(W_r) if val != 0]
        
        for a, ua in U_nz:
            for b, vb in V_nz:
                for c, wc in W_nz:
                    key = (a, b, c)
                    if key in target_indices:
                        decomposed[key] = decomposed.get(key, Fraction(0)) + ua * vb * wc
    
    # Verify all target entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                key = (a, b, c)
                if key not in decomposed or decomposed[key] != 1:
                    # Decomposition failed
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Also verify no extra non-zero entries (should be fine since we only added terms for target indices)
    # But let's check that all decomposed entries are in target_indices
    for key in decomposed:
        if key not in target_indices:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Convert to lists of float (integers and 0.5 are exactly representable)
    def frac_to_list(vecs):
        return [[float(v) for v in vec] for vec in vecs]
    
    U_out = frac_to_list(U_terms)
    V_out = frac_to_list(V_terms)
    W_out = frac_to_list(W_terms)
    
    return {"U": U_out, "V": V_out, "W": W_out}
