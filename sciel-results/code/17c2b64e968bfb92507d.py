"""Matrix multiplication 4x5x6: Strassen-based decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    rank = 108  # 3*12 + 3*12 + 1*36 + 1*24 + 1*24 + 1*12 = 108

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def strassen22(U_base_i, U_base_j, V_base_j, V_base_k, W_base_k, W_base_i, col):
        """Add Strassen 2x2x2 starting at col. Local indices 0,1 -> offset."""
        # M1 = (a00+a11)(b00+b11)
        U[U_base_i + 0][col] = 1; U[U_base_i + 1][col] = 1
        V[V_base_j + 0][col] = 1; V[V_base_k + 0][col] = 1
        W[W_base_k + 0][col] = 1; W[W_base_i + 1][col] = 1
        col += 1
        # M2 = (a10+a11)(b00)
        U[U_base_i + 1][col] = 1  # a10 (j offset 1)
        U[U_base_i + 1][col] = 2  # a11 (j offset 1) -- wait, this is wrong
        # Let me redo: U indices are (i*m + j). For 2x2 block at rows i0,i0+1 and cols j0,j0+1:
        # U[i0*m + j0], U[i0*m + j0+1], U[(i0+1)*m + j0], U[(i0+1)*m + j0+1]
        # Let me restart this function properly.
        return col

    # Let me just hardcode the Strassen components properly.
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen22(i0, j0, k0, col):
        """Add Strassen 2x2x2 for subblock A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2].
        U rows: i0..i0+1, cols j0..j0+1 -> indices i*m+j
        V rows: j0..j0+1, cols k0..k0+1 -> indices j*p+k
        W rows: k0..k0+1, cols i0..i0+1 -> indices k*n+i
        """
        # M1 = (a00+a11)(b00+b11)
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        # M2 = (a10+a11)(b00)
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1
        # M3 = a00(b10)
        U[i0*m + j0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        # M4 = a11(b11)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        # M5 = (a00+a01)(b11)
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        # M6 = (a10-a00)(b00+b01)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        # M7 = (a01-a11)(b10+b11)
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        return col

    def add_schoolbook_strip(n_strip, m_strip, p_strip, i0, j0, k0, col):
        """Add schoolbook for A[i0:i0+n_strip, j0:j0+m_strip] * B[j0:j0+m_strip, k0:k0+p_strip]."""
        for ii in range(n_strip):
            for jj in range(m_strip):
                for kk in range(p_strip):
                    U[(i0+ii)*m + (j0+jj)][col] = 1
                    V[(j0+jj)*p + (k0+kk)][col] = 1
                    W[(k0+kk)*n + (i0+ii)][col] = 1
                    col += 1
        return col

    col = 0
    # Block 1: rows 0-1, cols 0-1, k 0-1: Strassen
    col = add_strassen22(0, 0, 0, col)
    # Block 2: rows 0-1, cols 0-1, k 2-3: Strassen
    col = add_strassen22(0, 0, 2, col)
    # Block 3: rows 0-1, cols 0-1, k 4-5: Strassen
    col = add_strassen22(0, 0, 4, col)
    # Block 4: rows 2-3, cols 0-1, k 0-1: Strassen
    col = add_strassen22(2, 0, 0, col)
    # Block 5: rows 2-3, cols 0-1, k 2-3: Strassen
    col = add_strassen22(2, 0, 2, col)
    # Block 6: rows 2-3, cols 0-1, k 4-5: Strassen
    col = add_strassen22(2, 0, 4, col)
    # Remaining: rows 0-1, cols 2-4, all k: schoolbook 2x3x6 = 36
    col = add_schoolbook_strip(2, 3, 6, 0, 2, 0, col)
    # Remaining: rows 2-3, cols 2-4, all k: schoolbook 2x3x6 = 36
    col = add_schoolbook_strip(2, 3, 6, 2, 2, 0, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
