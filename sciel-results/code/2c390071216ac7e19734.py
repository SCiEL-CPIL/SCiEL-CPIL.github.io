def add_term(U, V, W, u_coeffs, v_coeffs, w_coeffs):
    # u_coeffs: dict {U_index: coeff}
    # v_coeffs: dict {V_index: coeff}
    # w_coeffs: dict {W_index: coeff}
    col = U.shape[1] # This won't work if we pre-allocate.
