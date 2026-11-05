import time

def _validate(U, V, W, n=3, m=3, p=4, verbose=False):
    r = len(U[0])
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    ok = False
                    if verbose:
                        print(f"FAIL at i={i},j={j},k={k}: sum={s}")
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _add_strassen_block(U, V, W, col_offset, i1, i2, j1, j2, k1, k2):
    """
    Adds a Strassen decomposition for the 2x2x2 block defined by
    i in {i1, i2}, j in {j1, j2}, k in {k1, k2}.
    """
    # Strassen products M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices in local 2x2 block
    # A indices: (0,0)->i1,j1; (0,1)->i1,j2; (1,0)->i2,j1; (1,1)->i2,j2
    # B indices: (0,0)->j1,k1; (0,1)->j1,k2; (1,0)->j2,k1; (1,1)->j2,k2
    # C indices: (0,0)->i1,k1; (0,1)->i1,k2; (1,0)->i2,k1; (1,1)->i2,k2
    
    # We define 7 columns.
    # For each product Ms, we specify which U, V, W entries are non-zero.
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # M1: (A11+A22)(B11+B22)
    # A11: i1,j1. A22: i2,j2.
    # B11: j1,k1. B22: j2,k2.
    # This product contributes to all 4 C entries?
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1?
    # No, Strassen formulas are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to distribute the M's into the tensor.
    # Each M is a single product of linear combinations.
    # We add one column for each M.
    
    # M1: U has A11, A22. V has B11, B22. W has all 4?
    # No, the W factor multiplies the result to the correct C position.
    # The term is U[i,j] V[j,k] W[k,i].
    # For M1 to contribute to C11, we need W[k1,i1] to be 1.
    # For M1 to contribute to C12, we need W[k2,i1] to be 1.
    # So for M1, we set W entries for all 4 (k,i) pairs to 1?
    # Yes, because M1 is added to all 4 C entries (with coefficients +1, +1, +1, +1? No, check formulas).
    # C11: +1. C12: 0? No, C12 = M3+M5. M1 is not in C12.
    # Wait, Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C11 (+1) and C22 (+1).
    # So for column c corresponding to M1:
    # U[i1*m+j1, c] = 1
    # U[i2*m+j2, c] = 1
    # V[j1*p+k1, c] = 1
    # V[j2*p+k2, c] = 1
    # W[k1*n+i1, c] = 1  # for C11
    # W[k2*n+i2, c] = 1  # for C22
    
    # Let's define the 7 products explicitly.
    
    # Helper to add a product
    def add_prod(c, u_coeffs, v_coeffs, w_coeffs):
        for idx, val in u_coeffs:
            U[idx][c] += val
        for idx, val in v_coeffs:
            V[idx][c] += val
        for idx, val in w_coeffs:
            W[idx][c] += val
            
    # M1
    c = col_offset
    add_prod(c, 
             [(i1*m+j1, 1), (i2*m+j2, 1)],
             [(j1*p+k1, 1), (j2*p+k2, 1)],
             [(k1*n+i1, 1), (k2*n+i2, 1)])
             
    # M2 = (A21+A22)B11
    # A21: i2,j1. A22: i2,j2. B11: j1,k1.
    # Contributes to C21 (+1) and C22 (-1).
    c = col_offset + 1
    add_prod(c,
             [(i2*m+j1, 1), (i2*m+j2, 1)],
             [(j1*p+k1, 1)],
             [(k1*n+i2, 1), (k2*n+i2, -1)])
             
    # M3 = A11(B12-B22)
    # A11: i1,j1. B12: j1,k2. B22: j2,k2.
    # Contributes to C12 (+1) and C22 (+1).
    c = col_offset + 2
    add_prod(c,
             [(i1*m+j1, 1)],
             [(j1*p+k2, 1), (j2*p+k2, -1)],
             [(k2*n+i1, 1), (k2*n+i2, 1)])
             
    # M4 = A22(B21-B11)
    # A22: i2,j2. B21: j2,k1. B11: j1,k1.
    # Contributes to C11 (+1) and C21 (+1).
    c = col_offset + 3
    add_prod(c,
             [(i2*m+j2, 1)],
             [(j2*p+k1, 1), (j1*p+k1, -1)],
             [(k1*n+i1, 1), (k1*n+i2, 1)])
             
    # M5 = (A11+A12)B22
    # A11: i1,j1. A12: i1,j2. B22: j2,k2.
    # Contributes to C11 (-1) and C12 (+1).
    c = col_offset + 4
    add_prod(c,
             [(i1*m+j1, 1), (i1*m+j2, 1)],
             [(j2*p+k2, 1)],
             [(k1*n+i1, -1), (k2*n+i1, 1)])
             
    # M6 = (A21-A11)(B11+B12)
    # A21: i2,j1. A11: i1,j1. B11: j1,k1. B12: j1,k2.
    # Contributes to C22 (+1).
    c = col_offset + 5
    add_prod(c,
             [(i2*m+j1, 1), (i1*m+j1, -1)],
             [(j1*p+k1, 1), (j1*p+k2, 1)],
             [(k2*n+i2, 1)])
             
    # M7 = (A12-A22)(B21+B22)
    # A12: i1,j2. A22: i2,j2. B21: j2,k1. B22: j2,k2.
    # Contributes to C11 (+1).
    c = col_offset + 6
    add_prod(c,
             [(i1*m+j2, 1), (i2*m+j2, -1)],
             [(j2*p+k1, 1), (j2*p+k2, 1)],
             [(k1*n+i1, 1)])

def construct():
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. T2: i in 0..2, j=2, k in 0..3
    # 12 terms
    for i in range(3):
        for k in range(4):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # 2. T1b: i=2, j in 0..1, k in 0..3
    # 8 terms
    for j in range(2):
        for k in range(4):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
    # 3. T1a1: i in 0..1, j in 0..1, k in 0..1
    # Strassen 2x2x2
    _add_strassen_block(U, V, W, col, 0, 1, 0, 1, 0, 1)
    col += 7
    
    # 4. T1a2: i in 0..1, j in 0..1, k in 2..3
    # Strassen 2x2x2
    _add_strassen_block(U, V, W, col, 0, 1, 0, 1, 2, 3)
    col += 7
    
    # Verify
    assert _validate(U, V, W, n, m, p), "Validation failed"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
