    best_valid_S = calc_S_vec(centers, C0, 0.0)
    best_valid_vars = x0.copy()
    
    def objective(vars):
        nonlocal best_valid_S, best_valid_vars
        Cx, Cy, phi = vars[0], vars[1], vars[2]
        C = np.array([Cx, Cy])
        cents = vars[3:].reshape(N, 2)
        
        S = calc_S_vec(cents, C, phi)
        pen = penalty(cents)
        
        if pen < 1e-6:
            if S < best_valid_S:
                best_valid_S = S
                best_valid_vars = vars.copy()
                
        return S + 10.0 * pen
