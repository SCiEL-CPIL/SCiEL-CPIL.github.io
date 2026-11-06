<<<<<<< SEARCH
def _kak_2qubit(target):
    """
    Exact KAK decomposition for 2-qubit gates.
    U = e^{iφ} (A⊗B) exp(i(θ1 σx⊗σx + θ2 σy⊗σy + θ3 σz⊗σz)) (C⊗D)
    
    This uses at most 3 CNOTs.
    """
    n = 4
    U = np.array(target, dtype=complex)
    
    # Check if U is already a product of single-qubit gates
    # If so, no CNOTs needed
    
    # KAK decomposition
    # Step 1: Compute U' = U · (det(U))^{-1/2} to make det(U') = 1
    det_U = np.linalg.det(U)
    if abs(det_U) > 1e-10:
        phase = det_U / abs(det_U)
        U_prime = U / np.sqrt(phase)
    else:
        U_prime = U
    
    # Step 2: Compute the KAK invariants
    # Use the formula: U' = (A⊗B) · exp(i·θ·n·σ) · (C⊗D)
    
    # For SU(4), we can use the fact that any element can be written as
    # (A⊗B) · diag(e^{iθ1}, e^{iθ2}, e^{iθ3}, e^{iθ4}) · (C⊗D)
    # where θ1+θ2+θ3+θ4 = 0
    
    # Let me use a numerical approach: parameterize with 3 CNOTs and optimize
    # But with a better initial guess and more iterations
    
    # Initialize parameters
    params = [0.0] * 24  # 8 u3 gates × 3 params each
    
    def circuit_from_params(p):
        gates = []
        idx = 0
        for layer in range(4):
            gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
            idx += 3
            gates.append(["u3", 1, p[idx], p[idx+1], p[idx+2]])
            idx += 3
            if layer < 3:
                gates.append(["cx", 0, 1])
        return gates
    
    best_params = params[:]
    best_dist = _distance(target, _build_fast(circuit_from_params(best_params), 2))
    
    # Multi-start coordinate descent
    n_starts = 5
    for start in range(n_starts):
        if _time_remaining() < 5:
            break
        
        if start == 0:
            current = [0.0] * 24
        else:
            current = [np.random.uniform(-math.pi, math.pi) for _ in range(24)]
        
        current_dist = _distance(target, _build_fast(circuit_from_params(current), 2))
        
        # Coordinate descent
        for sweep in range(3):
            step = 0.5 / (sweep + 1)
            for i in range(24):
                if _time_remaining() < 2:
                    break
                for delta in [step, -step]:
                    trial = current[:]
                    trial[i] += delta
                    trial[i] = (trial[i] + math.pi) % (2 * math.pi) - math.pi
                    dist = _distance(target, _build_fast(circuit_from_params(trial), 2))
                    if dist < current_dist:
                        current = trial
                        current_dist = dist
        
        if current_dist < best_dist:
            best_params = current
            best_dist = current_dist
    
    return circuit_from_params(best_params)
=======
def _optimize_2qubit(target):
    """
    Optimize a 2-qubit circuit with 3 CNOTs using scipy.
    """
    U_target = np.array(target, dtype=complex)
    
    def cost_func(params):
        if _time_remaining() < 5:
            return 1.0
        gates = []
        idx = 0
        # Layer 0: Single qubit
        gates.append(["u3", 0, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        gates.append(["u3", 1, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        # CNOT 1
        gates.append(["cx", 0, 1])
        # Layer 1: Single qubit
        gates.append(["u3", 0, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        gates.append(["u3", 1, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        # CNOT 2
        gates.append(["cx", 0, 1])
        # Layer 2: Single qubit
        gates.append(["u3", 0, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        gates.append(["u3", 1, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        # CNOT 3
        gates.append(["cx", 0, 1])
        # Layer 3: Single qubit
        gates.append(["u3", 0, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        gates.append(["u3", 1, params[idx], params[idx+1], params[idx+2]])
        idx += 3
        
        U_circuit = _build_fast(gates, 2)
        # Phase-insensitive distance
        dist = 1.0 - abs(np.trace(U_target.conj().T @ U_circuit)) / 4.0
        return dist

    best_params = None
    best_dist = 1.0
    
    # Multiple starts
    n_starts = 10
    for i in range(n_starts):
        if _time_remaining() < 10:
            break
        if i == 0:
            x0 = np.zeros(24)
        else:
            x0 = np.random.uniform(-math.pi, math.pi, 24)
        
        try:
            result = scipy.optimize.minimize(cost_func, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-8})
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x
        except Exception:
            pass

    if best_params is None:
        return []
    return _params_to_gates_2qubit(best_params)
>>>>>>> REPLACE

I need to add helper functions `_params_to_gates_2qubit` and `_optimize_3qubit`.

