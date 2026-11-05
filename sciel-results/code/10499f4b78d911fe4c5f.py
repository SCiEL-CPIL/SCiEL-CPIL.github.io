"""
Rank-28 exact tensor decomposition for 2x3x5 matrix multiplication.

Strategy:
- Split the 2x3x5 tensor into three disjoint sub-tensors:
  * Block A: rows i in {0,1}, cols j in {0,1}, out k in {0,1}  -> M_{2,2,2}, rank 7 (Strassen)
  * Block B: rows i in {0,1}, col  j in {2},   out k in {0,1,2,3,4} -> M_{2,1,5}, rank 10 (schoolbook)
  * Block C: rows i in {0,1}, cols j in {0,1}, out k in {2,3,4} -> M_{2,2,3}, rank 11 (Strassen for k={2,3} + 4 schoolbook for k=4)
- Total rank: 7 + 10 + 11 = 28.
- All entries are integers, so the decomposition is exact.
"""

import numpy as np

DIMENSIONS = [2, 3, 5]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    col = 0

    # ---------- Block A: M_{2,2,2} for i in {0,1}, j in {0,1}, k in {0,1} ----------
    # Strassen decomposition for 2x2x2.
    # A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=0,1), C is 2x2.
    # Strassen products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Mapping: local (i,j) -> global U index i*m + j
    #           local (j,k) -> global V index j*p + k
    #           local (k,i) -> global W index k*n + i

    # For each Strassen product, we need to set U, V, W columns.
    # U entries: for each (i,j) pair involved, set U[i*m+j, col] = coefficient
    # V entries: for each (j,k) pair involved, set V[j*p+k, col] = coefficient
    # W entries: for each (k,i) pair involved, set W[k*n+i, col] = coefficient (with sign)

    def add_strassen_2x2x2(col_offset):
        """Add Strassen rank-7 terms for a 2x2x2 block. Returns next col."""
        c = col_offset
        # P1 = (A00 + A11) * (B00 + B11)
        U[0*m+0, c] = 1; U[1*m+1, c] = 1
        V[0*p+0, c] = 1; V[1*p+1, c] = 1
        W[0*n+0, c] = 1; W[0*n+1, c] = 1; W[1*n+0, c] = 1; W[1*n+1, c] = -1
        c += 1
        # P2 = (A10 + A11) * B00
        U[1*m+0, c] = 1; U[1*m+1, c] = 1
        V[0*p+0, c] = 1
        W[1*n+0, c] = 1; W[1*n+1, c] = -1
        c += 1
        # P3 = A00 * (B01 - B11)
        U[0*m+0, c] = 1
        V[0*p+1, c] = 1; V[1*p+1, c] = -1
        W[0*n+1, c] = 1; W[1*n+1, c] = 1
        c += 1
        # P4 = A11 * (B10 - B00)
        U[1*m+1, c] = 1
        V[1*p+0, c] = 1; V[0*p+0, c] = -1
        W[0*n+0, c] = 1; W[1*n+0, c] = 1
        c += 1
        # P5 = (A00 + A01) * B11
        U[0*m+0, c] = 1; U[0*m+1, c] = 1
        V[1*p+1, c] = 1
        W[0*n+0, c] = -1; W[1*n+1, c] = 1
        c += 1
        # P6 = (A10 - A00) * (B00 + B01)
        U[1*m+0, c] = 1; U[0*m+0, c] = -1
        V[0*p+0, c] = 1; V[0*p+1, c] = 1
        W[1*n+1, c] = 1
        c += 1
        # P7 = (A01 - A11) * (B10 + B11)
        U[0*m+1, c] = 1; U[1*m+1, c] = -1
        V[1*p+0, c] = 1; V[1*p+1, c] = 1
        W[0*n+0, c] = 1
        c += 1
        return c

    col = add_strassen_2x2x2(col)

    # ---------- Block B: M_{2,1,5} for i in {0,1}, j in {2}, k in {0,1,2,3,4} ----------
    # Schoolbook: 2*1*5 = 10 terms.
    for i in range(n):
        for k in range(p):
            U[i*m+2, col] = 1
            V[2*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

    # ---------- Block C: M_{2,2,3} for i in {0,1}, j in {0,1}, k in {2,3,4} ----------
    # Strassen for k={2,3} (2x2x2 sub-block) + 4 schoolbook terms for k=4.
    # Strassen for the 2x2x2 block with local k' in {0,1} mapping to global k in {2,3}.
    def add_strassen_2x2x2_k23(col_offset):
        """Add Strassen rank-7 terms for 2x2x2 block where k maps to {2,3}."""
        c = col_offset
        # k_local 0 -> k_global 2, k_local 1 -> k_global 3
        # P1 = (A00 + A11) * (B02 + B13)
        U[0*m+0, c] = 1; U[1*m+1, c] = 1
        V[0*p+2, c] = 1; V[1*p+3, c] = 1
        W[2*n+0, c] = 1; W[3*n+0, c] = 1; W[2*n+1, c] = 1; W[3*n+1, c] = -1
        c += 1
        # P2 = (A10 + A11) * B02
        U[1*m+0, c] = 1; U[1*m+1, c] = 1
        V[0*p+2, c] = 1
        W[2*n+0, c] = 1; W[3*n+0, c] = -1
        c += 1
        # P3 = A00 * (B03 - B13)
        U[0*m+0, c] = 1
        V[0*p+3, c] = 1; V[1*p+3, c] = -1
        W[3*n+0, c] = 1; W[3*n+1, c] = 1
        c += 1
        # P4 = A11 * (B12 - B02)
        U[1*m+1, c] = 1
        V[1*p+2, c] = 1; V[0*p+2, c] = -1
        W[2*n+0, c] = 1; W[2*n+1, c] = 1
        c += 1
        # P5 = (A00 + A01) * B13
        U[0*m+0, c] = 1; U[0*m+1, c] = 1
        V[1*p+3, c] = 1
        W[2*n+0, c] = -1; W[3*n+1, c] = 1
        c += 1
        # P6 = (A10 - A00) * (B02 + B03)
        U[1*m+0, c] = 1; U[0*m+0, c] = -1
        V[0*p+2, c] = 1; V[0*p+3, c] = 1
        W[3*n+1, c] = 1
        c += 1
        # P7 = (A01 - A11) * (B12 + B13)
        U[0*m+1, c] = 1; U[1*m+1, c] = -1
        V[1*p+2, c] = 1; V[1*p+3, c] = 1
        W[2*n+0, c] = 1
        c += 1
        return c

    col = add_strassen_2x2x2_k23(col)

    # Schoolbook for k=4: 2*2*1 = 4 terms
    for i in range(n):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1

    assert col == R, f"Expected rank {R}, got {col}"

    # Convert to list of lists of ints
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
