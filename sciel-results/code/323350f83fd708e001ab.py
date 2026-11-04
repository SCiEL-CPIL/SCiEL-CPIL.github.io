<<<<<<< SEARCH
def _add_strassen_2x2x2(U, V, W, i0, j0, k0, rank_counter):
    """Add 7 rank-1 terms for a 2x2x2 Strassen sub-problem.
    
    i0: starting row of A (and corresponding rows in C)
    j0: starting col of A / starting row of B
    k0: starting col of B (and corresponding cols in C)
    rank_counter: list with [current_rank] to track rank
    """
    n, m, p = 4, 5, 5
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    
    def add_term(U_indices, V_indices, W_indices, coeffs=1):
        """Add one rank-1 term."""
        r = rank_counter[0]
        rank_counter[0] += 1
        for idx in U_indices:
            U[idx][r] = coeffs
        for idx in V_indices:
            V[idx][r] = coeffs
        for idx in W_indices:
            W[idx][r] = coeffs
        return r
    
    # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
    # Contributes: C[k0,i0] += 1, C[k1,i1] += 1
    add_term(
        [i0*m+j0, i1*m+j1],
        [j0*p+k0, j1*p+k1],
        [k0*n+i0, k1*n+i1],
        1
    )
    
    # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
    # Contributes: C[k0,i1] += 1
    add_term(
        [i1*m+j0, i1*m+j1],
        [j0*p+k0],
        [k0*n+i1],
        1
    )
    
    # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
    # Contributes: C[k0,i0] += 1, C[k1,i0] += -1
    add_term(
        [i0*m+j0],
        [j1*p+k0],
        [k0*n+i0],
        1
    )
    add_term(
        [i0*m+j0],
        [j1*p+k1],
        [k0*n+i0],
        -1
    )
    
    # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
    # Contributes: C[k0,i0] += -1, C[k1,i0] += 1, C[k0,i1] += 1
    add_term(
        [i1*m+j1],
        [j0*p+k1],
        [k0*n+i0],
        -1
    )
    add_term(
        [i1*m+j1],
        [j0*p+k0],
        [k0*n+i0],
        1
    )
    add_term(
        [i1*m+j1],
        [j0*p+k1],
        [k0*n+i1],
        1
    )
    
    # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
    # Contributes: C[k0,i0] += -1, C[k1,i0] += 1
    add_term(
        [i0*m+j0],
        [j1*p+k1],
        [k0*n+i0],
        -1
    )
    add_term(
        [i0*m+j1],
        [j1*p+k1],
        [k0*n+i0],
        -1
    )
    add_term(
        [i0*m+j0],
        [j1*p+k1],
        [k1*n+i0],
        1
    )
    add_term(
        [i0*m+j1],
        [j1*p+k1],
        [k1*n+i0],
        1
    )
    
    # M6 = (A[i1,j1] - A[i0,j1]) * (B[j0,k0] + B[j0,k1])
    # Contributes: C[k1,i1] += 1
    add_term(
        [i1*m+j1],
        [j0*p+k0],
        [k1*n+i1],
        1
    )
    add_term(
        [i1*m+j1],
        [j0*p+k1],
        [k1*n+i1],
        1
    )
    add_term(
        [i0*m+j1],
        [j0*p+k0],
        [k1*n+i1],
        -1
    )
    add_term(
        [i0*m+j1],
        [j0*p+k1],
        [k1*n+i1],
        -1
    )
    
    # M7 = (A[i0,j1] - A[i1,j1]) * (B[j0,k1] + B[j1,k1])
    # Contributes: C[k0,i0] += 1
    add_term(
        [i0*m+j1],
        [j0*p+k1],
        [k0*n+i0],
        1
    )
    add_term(
        [i0*m+j1],
        [j1*p+k1],
        [k0*n+i0],
        1
    )
    add_term(
        [i1*m+j1],
        [j0*p+k1],
        [k0*n+i0],
        -1
    )
    add_term(
        [i1*m+j1],
        [j1*p+k1],
        [k0*n+i0],
        -1
    )
    
    return rank_counter[0]
=======
def _add_strassen_2x2x2(U, V, W, i0, j0, k0, rank_counter):
    """Add 7 rank-1 terms for a 2x2x2 Strassen sub-problem.
    
    Uses standard Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A10)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 4, 5, 5
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    
    def add_term(U_entries, V_entries, W_entries):
        """Add one rank-1 term. Each entry is (index, coefficient)."""
        r = rank_counter[0]
        rank_counter[0] += 1
        for idx, coeff in U_entries:
            U[idx][r] += coeff
        for idx, coeff in V_entries:
            V[idx][r] += coeff
        for idx, coeff in W_entries:
            W[idx][r] += coeff
        return r
    
    # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
    # C00 += 1, C11 += 1
    add_term(
        [(i0*m+j0, 1), (i1*m+j1, 1)],
        [(j0*p+k0, 1), (j1*p+k1, 1)],
        [(k0*n+i0, 1), (k1*n+i1, 1)]
    )
    
    # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
    # C10 += 1
    add_term(
        [(i1*m+j0, 1), (i1*m+j1, 1)],
        [(j0*p+k0, 1)],
        [(k0*n+i1, 1)]
    )
    
    # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
    # C01 += 1, C11 += 1
    add_term(
        [(i0*m+j0, 1)],
        [(j1*p+k0, 1), (j1*p+k1, -1)],
        [(k1*n+i0, 1), (k1*n+i1, 1)]
    )
    
    # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
    # C00 += 1, C10 += 1
    add_term(
        [(i1*m+j1, 1)],
        [(j0*p+k1, 1), (j0*p+k0, -1)],
        [(k0*n+i0, 1), (k0*n+i1, 1)]
    )
    
    # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
    # C01 += 1, C00 -= 1
    add_term(
        [(i0*m+j0, 1), (i0*m+j1, 1)],
        [(j1*p+k1, 1)],
        [(k1*n+i0, 1), (k0*n+i0, -1)]
    )
    
    # M6 = (A[i1,j1] - A[i0,j1]) * (B[j0,k0] + B[j0,k1])
    # C11 += 1
    add_term(
        [(i1*m+j1, 1), (i0*m+j1, -1)],
        [(j0*p+k0, 1), (j0*p+k1, 1)],
        [(k1*n+i1, 1)]
    )
    
    # M7 = (A[i0,j1] - A[i1,j1]) * (B[j0,k1] + B[j1,k1])
    # C00 += 1
    add_term(
        [(i0*m+j1, 1), (i1*m+j1, -1)],
        [(j0*p+k1, 1), (j1*p+k1, 1)],
        [(k0*n+i0, 1)]
    )
    
    return rank_counter[0]
>>>>>>> REPLACE

Now let me add a verification function and fix the construct function to include verification:

