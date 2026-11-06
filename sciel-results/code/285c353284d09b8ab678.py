def _fit_angles_via_mps(n, coupling, field, layers):
    # Get exact ground state
    ground_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    # Build H matrix
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n - 1):
        term = I.copy()
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
        
    for i in range(n):
        term = I.copy()
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term

    def objective(angles_flat):
        angles_reshaped = angles_flat.reshape(layers, n)
        state = _simulate_circuit(angles_reshaped, n, layers)
        # Compute energy
        return np.vdot(state, H @ state).real

    # Initial guess
    if coupling != 0:
        base_angle = np.arctan2(field, coupling)
    else:
        base_angle = np.pi / 2
        
    theta0 = np.full((layers, n), base_angle / layers)
    
    from scipy.optimize import minimize
    
    # Try L-BFGS-B
    x0 = theta0.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    best_angles = result.x.reshape(layers, n)
    best_energy = objective(best_angles.flatten())
    
    # Try multiple initializations
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.random.uniform(-np.pi/4, np.pi/4, (layers, n)),
    ]
    
    for init in inits:
        x0 = init.flatten()
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-12})
        e = objective(res.x.flatten())
        if e < best_energy:
            best_energy = e
            best_angles = res.x.reshape(layers, n)
            
    return best_angles
