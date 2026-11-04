def _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries):
    """Add a rank-1 term. a_entries, b_entries, c_entries are lists of (index, coeff)."""
    for idx, coeff in a_entries:
        U[idx][col] += coeff
    for idx, coeff in b_entries:
        V[idx][col] += coeff
    for idx, coeff in c_entries:
        W[idx][col] += coeff

def construct():
    n, m, p = 4, 4, 5
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # 7 Strassen terms for 4x4x4, each with 4 schoolbook sub-terms
    # A[i,k] index: i*4+k
    # B[k,j] index: k*5+j
    # C[i,j] index: j*4+i (W convention: W[col*n+row])
    
    for i in range(2):
        for k in range(2):
            for j in range(2):
                # M1: (A00+A11)*(B00+B11) → C00(+), C11(+)
                a_entries = [(i*4+k, 1), ((i+2)*4+(k+2), 1)]
                b_entries = [(k*5+j, 1), ((k+2)*5+(j+2), 1)]
                c_entries = [(j*4+i, 1), ((j+2)*4+(i+2), 1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
                
                # M2: (A10+A11)*B00 → C10(+), C11(-)
                a_entries = [((i+2)*4+k, 1), ((i+2)*4+(k+2), 1)]
                b_entries = [(k*5+j, 1)]
                c_entries = [(j*4+(i+2), 1), ((j+2)*4+(i+2), -1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
                
                # M3: A00*(B01-B11) → C01(+), C11(+)
                a_entries = [(i*4+k, 1)]
                b_entries = [(k*5+(j+2), 1), ((k+2)*5+(j+2), -1)]
                c_entries = [((j+2)*4+i, 1), ((j+2)*4+(i+2), 1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
                
                # M4: A11*(B10-B00) → C00(+), C10(+)
                a_entries = [((i+2)*4+(k+2), 1)]
                b_entries = [((k+2)*5+j, 1), (k*5+j, -1)]
                c_entries = [(j*4+i, 1), (j*4+(i+2), 1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
                
                # M5: (A00+A01)*B11 → C00(-), C01(+)
                a_entries = [(i*4+k, 1), (i*4+(k+2), 1)]
                b_entries = [((k+2)*5+(j+2), 1)]
                c_entries = [(j*4+i, -1), ((j+2)*4+i, 1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
                
                # M6: (A10-A00)*(B00+B01) → C11(+)
                a_entries = [((i+2)*4+k, 1), (i*4+k, -1)]
                b_entries = [(k*5+j, 1), (k*5+(j+2), 1)]
                c_entries = [((j+2)*4+(i+2), 1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
                
                # M7: (A01-A11)*(B10+B11) → C00(+), C11(+)
                a_entries = [(i*4+(k+2), 1), ((i+2)*4+(k+2), -1)]
                b_entries = [((k+2)*5+j, 1), ((k+2)*5+(j+2), 1)]
                c_entries = [(j*4+i, 1), ((j+2)*4+(i+2), 1)]
                _add_rank1_term(U, V, W, col, a_entries, b_entries, c_entries)
                col += 1
    
    # 16 schoolbook terms for 4x4x1 (fifth column)
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    return {"U": U, "V": V, "W": W}
