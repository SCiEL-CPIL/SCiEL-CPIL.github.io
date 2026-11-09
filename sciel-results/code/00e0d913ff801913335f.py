<<<<<<< SEARCH
    best_score = -np.inf
    best_coeffs = np.zeros(6)
    best_points = _fourier_to_points(best_coeffs)
    
    # Initial evaluation
    init_score = -_objective(best_coeffs)
    if init_score > best_score:
        best_score = init_score
        best_points = _fourier_to_points(best_coeffs)
        
=======
    best_score = -np.inf
    best_coeffs = np.zeros(6)
    best_points, _ = _fourier_to_points(best_coeffs)
    
    # Initial evaluation
    init_score = -_objective(best_coeffs)
    if init_score > best_score:
        best_score = init_score
        best_points, _ = _fourier_to_points(best_coeffs)
        
>>>>>>> REPLACE

