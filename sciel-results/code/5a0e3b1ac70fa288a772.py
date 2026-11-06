    # Verification and scaling
    expected_f = apply_laplacian_2d(Y0).flatten()
    actual_f = f(t0, y0)
    
    # Estimate scaling factor in case h != 1
    # Assume A_actual = scale * A_h1
    dot_actual = np.dot(actual_f, actual_f)
    dot_expected = np.dot(expected_f, expected_f)
    if dot_expected > 1e-15:
        scale_factor = np.dot(actual_f, expected_f) / dot_expected
    else:
        scale_factor = 1.0
        
    # Adjust eigenvalues
    lambda_2d = lambda_2d * scale_factor
