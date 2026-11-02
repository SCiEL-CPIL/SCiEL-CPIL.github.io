import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        for idx in u_indices:
            U[idx, col] += coeff
        for idx in v_indices:
            V[idx, col] += coeff
        for idx in w_indices:
            W[idx, col] += coeff
        col += 1

    # Part 2: A[3x1] * B[1x4]
    # A col 2, B row 2
    # A indices: (i, 2) -> i*3 + 2
    # B indices: (2, k) -> 2*4 + k
    # C indices: (k, i) -> k*3 + i
    for i in range(3):
        for k in range(4):
            add_term(
                [i * 3 + 2],
                [2 * 4 + k],
                [k * 3 + i]
            )
            
    # Part 1: A[3x2] * B[2x4]
    # A cols 0,1. B rows 0,1.
    # Split A rows into 0,1 and 2.
    # Split B cols into 0,1 and 2,3.
    
    # Sub-part 1a: A[2x2] * B[2x2] (top-left)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1.
    # C rows 0,1 cols 0,1.
    # Strassen
    # M1 = (A00+A11)(B00+B11)
    add_term(
        [0*3+0, 1*3+1],
        [0*4+0, 1*4+1],
        [0*3+0, 1*3+1] # C00, C11
    )
    # M2 = (A10+A11)B00
    add_term(
        [1*3+0, 1*3+1],
        [0*4+0],
        [0*3+1, 1*3+1] # C10, C11
    )
    # M3 = A00(B01-B11)
    add_term(
        [0*3+0],
        [0*4+1, 1*4+1],
        [0*3+0, 0*3+1] # C00, C01
    )
    # Wait, M3 = A00(B01-B11).
    # V indices: 0*4+1 (B01) with +1, 1*4+1 (B11) with -1.
    # We can't just add indices with different signs in one term easily unless we allow negative coeffs.
    # The add_term function uses a single coeff.
    # We should modify add_term to accept lists of (index, coeff) or just call it multiple times?
    # No, a rank-1 term has a single scalar multiplier for the whole product?
    # No, $U_{i,j,r}$ can be any number.
    # So we can set U, V, W entries directly.
    
    # Let's rewrite the helper to accept coefficients.
    
    # Reset and restart with a better helper
    col = 0
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(u_data, v_data, w_data):
        nonlocal col
        for idx, val in u_data:
            U[idx, col] += val
        for idx, val in v_data:
            V[idx, col] += val
        for idx, val in w_data:
            W[idx, col] += val
        col += 1

    # Part 2: A[3x1] * B[1x4]
    for i in range(3):
        for k in range(4):
            add_term(
                [(i * 3 + 2, 1)],
                [(2 * 4 + k, 1)],
                [(k * 3 + i, 1)]
            )
            
    # Part 1: A[3x2] * B[2x4]
    
    # Sub-part 1a: A[2x2] (rows 0,1; cols 0,1) * B[2x2] (rows 0,1; cols 0,1)
    # C[2x2] (rows 0,1; cols 0,1)
    # Strassen
    # M1 = (A00+A11)(B00+B11)
    add_term(
        [(0*3+0, 1), (1*3+1, 1)],
        [(0*4+0, 1), (1*4+1, 1)],
        [(0*3+0, 1), (1*3+1, 1)]
    )
    # M2 = (A10+A11)B00
    add_term(
        [(1*3+0, 1), (1*3+1, 1)],
        [(0*4+0, 1)],
        [(0*3+1, 1), (1*3+1, 1)]
    )
    # M3 = A00(B01-B11)
    add_term(
        [(0*3+0, 1)],
        [(0*4+1, 1), (1*4+1, -1)],
        [(0*3+0, 1), (0*3+1, 1)]
    )
    # M4 = A11(B10-B00)
    add_term(
        [(1*3+1, 1)],
        [(1*4+0, 1), (0*4+0, -1)],
        [(0*3+1, 1), (1*3+1, 1)]
    )
    # M5 = (A00+A01)B11
    add_term(
        [(0*3+0, 1), (0*3+1, 1)],
        [(1*4+1, 1)],
        [(0*3+0, -1), (0*3+1, 1)]
    )
    # M6 = (A10-A00)(B00+B01)
    add_term(
        [(1*3+0, 1), (0*3+0, -1)],
        [(0*4+0, 1), (0*4+1, 1)],
        [(1*3+1, 1)]
    )
    # M7 = (A01-A11)(B10+B11)
    add_term(
        [(0*3+1, 1), (1*3+1, -1)],
        [(1*4+0, 1), (1*4+1, 1)],
        [(0*3+0, 1)]
    )

    # Sub-part 1b: A[1x2] (row 2; cols 0,1) * B[2x2] (rows 0,1; cols 0,1)
    # C[1x2] (row 2; cols 0,1)
    # Schoolbook
    # C20 = A20*B00 + A21*B10
    add_term(
        [(2*3+0, 1)],
        [(0*4+0, 1)],
        [(0*3+2, 1)]
    )
    add_term(
        [(2*3+1, 1)],
        [(1*4+0, 1)],
        [(0*3+2, 1)]
    )
    # C21 = A20*B01 + A21*B11
    add_term(
        [(2*3+0, 1)],
        [(0*4+1, 1)],
        [(1*3+2, 1)]
    )
    add_term(
        [(2*3+1, 1)],
        [(1*4+1, 1)],
        [(1*3+2, 1)]
    )

    # Sub-part 1c: A[2x2] (rows 0,1; cols 0,1) * B[2x2] (rows 0,1; cols 2,3)
    # C[2x2] (rows 0,1; cols 2,3)
    # Strassen
    # M1 = (A00+A11)(B02+B13)
    add_term(
        [(0*3+0, 1), (1*3+1, 1)],
        [(0*4+2, 1), (1*4+3, 1)],
        [(2*3+0, 1), (3*3+1, 1)]
    )
    # M2 = (A10+A11)B02
    add_term(
        [(1*3+0, 1), (1*3+1, 1)],
        [(0*4+2, 1)],
        [(2*3+1, 1), (3*3+1, 1)]
    )
    # M3 = A00(B03-B13)
    add_term(
        [(0*3+0, 1)],
        [(0*4+3, 1), (1*4+3, -1)],
        [(2*3+0, 1), (3*3+0, 1)]
    )
    # M4 = A11(B12-B02)
    add_term(
        [(1*3+1, 1)],
        [(1*4+2, 1), (0*4+2, -1)],
        [(2*3+1, 1), (3*3+1, 1)]
    )
    # M5 = (A00+A01)B13
    add_term(
        [(0*3+0, 1), (0*3+1, 1)],
        [(1*4+3, 1)],
        [(2*3+0, -1), (3*3+0, 1)]
    )
    # M6 = (A10-A00)(B02+B03)
    add_term(
        [(1*3+0, 1), (0*3+0, -1)],
        [(0*4+2, 1), (0*4+3, 1)],
        [(3*3+1, 1)]
    )
    # M7 = (A01-A11)(B12+B13)
    add_term(
        [(0*3+1, 1), (1*3+1, -1)],
        [(1*4+2, 1), (1*4+3, 1)],
        [(2*3+0, 1)]
    )

    # Sub-part 1d: A[1x2] (row 2; cols 0,1) * B[2x2] (rows 0,1; cols 2,3)
    # C[1x2] (row 2; cols 2,3)
    # Schoolbook
    # C22 = A20*B02 + A21*B12
    add_term(
        [(2*3+0, 1)],
        [(0*4+2, 1)],
        [(2*3+2, 1)]
    )
    add_term(
        [(2*3+1, 1)],
        [(1*4+2, 1)],
        [(2*3+2, 1)]
    )
    # C23 = A20*B03 + A21*B13
    add_term(
        [(2*3+0, 1)],
        [(0*4+3, 1)],
        [(3*3+2, 1)]
    )
    add_term(
        [(2*3+1, 1)],
        [(1*4+3, 1)],
        [(3*3+2, 1)]
    )

    return {"U": U, "V": V, "W": W}
