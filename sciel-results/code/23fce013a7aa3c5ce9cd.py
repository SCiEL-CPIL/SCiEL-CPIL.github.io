def simulate_circuit_fast(theta, n):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    
    for l in range(len(theta)):
        # Apply Ry
        for i in range(n):
            angle = theta[l, i]
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            # Ry = [[c, -s], [s, c]]
            # We need to apply this to qubit i.
            # Iterate over all basis states
            for idx in range(2**n):
                # Check bit i
                if (idx >> i) & 1 == 0:
                    # |0> -> c|0> + s|1>
                    idx_1 = idx | (1 << i)
                    val_0 = psi[idx]
                    val_1 = psi[idx_1]
                    psi[idx] = c * val_0 + s * val_1
                    psi[idx_1] = -s * val_0 + c * val_1
                # else bit is 1, handled by the pair above?
                # No, we must be careful not to double process.
                # Standard way: iterate over indices where bit i is 0.
        
        # Apply CZ
        if l < len(theta) - 1:
            for i in range(n-1):
                # Flip sign if bit i and i+1 are 1
                mask = 0
                for idx in range(2**n):
                    if ((idx >> i) & 1) and ((idx >> (i+1)) & 1):
                        psi[idx] *= -1
    return psi
