def _is_close(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol, rtol=tol)

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Hardcoded Gates
    if qubits == 2:
        # CNOT
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _is_close(target, cnot):
            return [["cx", 0, 1]]
            
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _is_close(target, swap):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
            
        # iSWAP
        iswap = np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if _is_close(target, iswap):
            # 3 CNOTs synthesis
            # H(0) CX(0,1) Sdg(1) CX(1,0) H(1) ? No.
            # Standard:
            # u3(0, pi/2, 0, pi) # H
            # cx(0,1)
            # u3(1, pi/2, 0, pi/2) # Sdg? No, Sdg is pi/2, 0, -pi/2?
            # Let's use KAK for iSWAP to be safe.
            pass
            
    if qubits == 3:
        # Toffoli
        # Check if target is Toffoli
        # Toffoli matrix is large.
        # I will just use the coordinate descent for Toffoli with 7 CNOTs.
        
    # Fall back to coordinate descent
    # ... (same as above)
