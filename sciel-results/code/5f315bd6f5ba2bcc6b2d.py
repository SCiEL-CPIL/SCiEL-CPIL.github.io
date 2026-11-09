import numpy as np
from scipy.optimize import minimize
import functools

# Cache for optimized parameters
_optimization_cache = {}

def _get_hamiltonian(n, coupling, field):
    """
    Construct the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Identity for Z and X parts
    # Z_i Z_{i+1}
    # X_i
    
    # We can build this using Kronecker products or bit manipulation.
    # Bit manipulation is faster for small n.
    
    # Basis states are integers 0 to 2^n - 1
    # Bit i corresponds to qubit i (LSB is qubit 0)
    
    for state in range(dim):
        # Calculate energy of this basis state for the Z part
        # H_Z = -coupling * sum Z_i Z_{i+1}
        # Z_i eigenvalue is 1 if bit is 0, -1 if bit is 1
        z_energy = 0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            z_energy += z_i * z_ip1
        H_Z_val = -coupling * z_energy
        
        # H_X = -field * sum X_i
        # X_i flips bit i. So <state| X_i |state> is 0.
        # We need to fill the off-diagonal elements.
        
        # Diagonal part
        H[state, state] += H_Z_val
        
        # Off-diagonal part from X terms
        for i in range(n):
            flipped_state = state ^ (1 << i)
            # H[state, flipped_state] += -field
            # Since H is Hermitian, H[flipped_state, state] will be handled when we iterate that state
            # But we can just add it here.
            H[state, flipped_state] += -field

    return H

def _apply_circuit(state, thetas, n):
    """
    Apply the circuit to the initial state |0...0>.
    thetas: array of shape (layers, n)
    """
    # Start with |0...0>
    # State vector
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    
    layers = thetas.shape[0]
    
    # Helper to apply Ry(theta) to qubit i
    def apply_ry(psi, i, theta):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        # Ry = [[c, -s], [s, c]]
        # We need to apply this to the i-th qubit.
        # This is equivalent to applying a 2x2 rotation to the 2^(n-1) blocks.
        # Efficient way: iterate over states where bit i is 0, and mix with state where bit i is 1.
        
        # Create a view or copy? We need to update psi in place or return new.
        # Let's do it in a way that avoids large copies.
        # We can use np.take and np.put, or just loop over blocks.
        # For n=6, dim=64. Looping is fine.
        
        new_psi = np.zeros_like(psi)
        for state in range(2 ** n):
            if (state >> i) & 1 == 0:
                partner = state | (1 << i)
                # |0> -> c|0> + s|1>
                # |1> -> -s|0> + c|1>
                # psi_partner = s * psi_state - c * psi_partner_old ? No.
                # [psi_0']   = [c  -s] [psi_0]
                # [psi_1']     [s   c] [psi_1]
                
                psi_0 = psi[state]
                psi_1 = psi[partner]
                
                new_psi[state] = c * psi_0 - s * psi_1
                new_psi[partner] = s * psi_0 + c * psi_1
            # If bit is 1, we handle it in the partner's iteration? 
            # No, if we only iterate states with bit 0, we cover all pairs.
        return new_psi

    # Helper to apply CZ between i and i+1
    def apply_cz(psi, i, j):
        # CZ = diag(1, 1, 1, -1) in basis 00, 01, 10, 11
        # Flips sign if both bits are 1.
        new_psi = psi.copy()
        mask = 0
        for state in range(2 ** n):
            if ((state >> i) & 1) == 1 and ((state >> j) & 1) == 1:
                new_psi[state] *= -1
        return new_psi

    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            psi = apply_ry(psi, i, thetas[l, i])
        
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                psi = apply_cz(psi, i, i + 1)
                
    return psi

def _energy(thetas, n, coupling, field, H):
    psi = _apply_circuit(np.zeros(2**n), thetas, n)
    # Energy = psi^dagger H psi
    return np.real(np.vdot(psi, H @ psi))

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _optimization_cache:
        return _optimization_cache[key]
        
    H = _get_hamiltonian(n, coupling, field)
    
    # Initial guess
    # If field is small, start near pi/2 for last layer
    # If field is large, start near 0
    init_theta = np.zeros((layers, n))
    if field < 1.0:
        init_theta[-1] = np.pi / 2
    else:
        init_theta[-1] = 0.0
        init_theta[0] = np.pi / 4 # Some entanglement might help
    
    # Flatten parameters
    x0 = init_theta.flatten()
    
    def objective(x):
        thetas = x.reshape((layers, n))
        return _energy(thetas, n, coupling, field, H)
        
    # Optimize
    # Use Nelder-Mead for robustness
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    
    optimal_thetas = res.x.reshape((layers, n))
    
    # Store in cache
    _optimization_cache[key] = optimal_thetas
    
    return optimal_thetas
