def add_virtual_2x2(i_offsets, j_offsets, k_offsets, 
                    A_coeffs, B_coeffs, C_coeffs,
                    U, V, W, col_offset):
    """
    Add a virtual 2×2×2 multiplication.
    i_offsets: list of 2 row offsets for A and C
    j_offsets: list of 2 col offsets for A and row offsets for B
    k_offsets: list of 2 col offsets for B and C
    A_coeffs: 2×2 matrix of coefficients for A entries
    B_coeffs: 2×2 matrix of coefficients for B entries
    C_coeffs: 2×2 matrix of coefficients for C entries
    """
