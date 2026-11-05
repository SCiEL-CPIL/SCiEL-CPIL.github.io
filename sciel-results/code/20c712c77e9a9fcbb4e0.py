import numpy as np
import scipy.optimize as opt
import time

# Cache for Hamiltonians and U_CZ
_hamiltonian_cache = {}
_ucz_cache = {}
_psi0_cache = {}

def get_hamiltonian(n):
    if n in _hamiltonian_cache:
        return _hamiltonian_cache[n]
    
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        op = I
        for j in range(n):
            if j == i or j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H -= op # coupling = 1
    
    # X terms
    for i in range(n):
        op = I
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H -= op # field = 1 (we scale later? No, field is passed)
        # Wait, the field is a parameter. The Hamiltonian depends on field.
        # So we cannot cache H for different fields easily unless we separate terms.
        
    # Better: Cache ZZ part and X part separately.
    _hamiltonian_cache[n] = H
    return H

# Re-evaluating caching strategy.
# H = -J H_ZZ - h H_X
# We can cache H_ZZ and H_X for each n.

def get_h_terms(n):
    if n in _hamiltonian_cache:
        return _hamiltonian_cache[n]
    
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    H_ZZ = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n - 1):
        op = I
        for j in range(n):
            if j == i or j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H_ZZ += op
        
    H_X = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        op = I
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H_X += op
        
    _hamiltonian_cache[n] = (H_ZZ, H_X)
    return _hamiltonian_cache[n]

def get_ucz(n):
    if n in _ucz_cache:
        return _ucz_cache[n]
        
    I = np.eye(2)
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]])
                   
    U_CZ = np.eye(2**n)
    # Apply CZ on all adjacent pairs
    # The task says: "between layers apply CZ on every adjacent pair in ascending order"
    # This implies a sequence of CZ gates: CZ(0,1), CZ(1,2), ..., CZ(n-2, n-1)
    # These do not commute if they share qubits?
    # CZ(0,1) and CZ(1,2) share qubit 1.
    # So we must apply them sequentially.
    
    # It is easier to build the full U_CZ by applying the gates to the identity matrix.
    U = np.eye(2**n, dtype=complex)
    
    # Helper to apply 2-qubit gate
    def apply_2q_gate(U, gate, q1, q2, n):
        # Permute axes to bring q1, q2 to end
        # This is complex to implement generically and efficiently for small n.
        # Alternative: Construct the full gate matrix.
        pass

    # Given n <= 6, we can just iterate through basis states?
    # Or use the tensor reshaping method on the identity matrix.
    
    # Let's implement a generic apply_gate function.
    def apply_single_qubit_gate(state, gate, qubit, n):
        # state: 1D array
        # gate: 2x2 matrix
        # qubit: index 0..n-1
        # n: num qubits
        
        # Reshape to tensor
        shape = [2] * n
        tensor = state.reshape(shape)
        
        # Apply gate along axis 'qubit'
        # np.tensordot(gate, tensor, axes=([1], [qubit]))
        # But we need to handle the other axes correctly.
        
        # Easier: move axis to front, apply, move back
        # Or just use einsum?
        
        # Let's use the standard trick:
        # Split the state into blocks of size 2^qubit?
        # No, let's stick to the matrix multiplication of the full unitary.
        # It's simpler and fast enough.
        pass

    # Let's build U_CZ by multiplying full unitary matrices.
    # For each pair (i, i+1), construct the full 2^n x 2^n matrix for CZ(i, i+1).
    
    def make_full_gate(gate_2q, q1, q2, n):
        # gate_2q is 4x4
        # q1, q2 are indices
        # Result is 2^n x 2^n
        # We can do this by iterating over all basis states? 64 states.
        # Or using kron products.
        
        # Order of qubits in the matrix representation:
        # Usually qubit 0 is the least significant bit (rightmost in tensor product)
        # or most significant?
        # Standard physics convention: |q_{n-1} ... q_0>
        # np.kron(A, B) acts on (A's space) \otimes (B's space).
        # If we do np.kron(np.kron(I, Z), I), the Z is on the middle qubit?
        # Let's define the mapping clearly.
        # Let qubit 0 be the leftmost in the ket notation |q_0 q_1 ... q_{n-1}>.
        # Then the state vector index is sum q_i 2^{n-1-i}.
        # This corresponds to np.kron(op_0, np.kron(op_1, ...))
        
        # Let's assume qubit 0 is the first in the kron chain (MSB).
        # So state index = sum q_i 2^{n-1-i}.
        
        ops = []
        for i in range(n):
            if i == q1:
                # We need to handle 2-qubit gates carefully with kron
                # It's easier to construct the full matrix by applying the gate to basis states.
                pass

    # Given the complexity of generic gate application, and n is small,
    # let's just precompute the full U_CZ matrix for n=4 and n=6 by simulation.
    # We can simulate the CZ sequence on the basis states.
    
    U = np.eye(2**n, dtype=complex)
    
    # Function to apply CZ(i, i+1) to a state vector
    def apply_cz_seq(state, i, n):
        # state is 2^n vector
        # CZ acts on qubits i and i+1.
        # We can iterate through all basis states.
        # If the bits at i and i+1 are both 1, multiply by -1.
        new_state = state.copy()
        for idx in range(2**n):
            # Extract bits
            # Bit i corresponds to 2^{n-1-i} if q0 is MSB?
            # Let's check the convention.
            # If we use np.kron(op0, op1), the basis is |q0 q1>.
            # Index 0: |00>, 1: |01>, 2: |10>, 3: |11>.
            # So q0 is MSB, q1 is LSB.
            # Bit at position i (0-indexed from left) is (idx >> (n-1-i)) & 1.
            bit_i = (idx >> (n - 1 - i)) & 1
            bit_ip1 = (idx >> (n - 1 - (i + 1))) & 1
            if bit_i == 1 and bit_ip1 == 1:
                new_state[idx] = -state[idx]
            else:
                new_state[idx] = state[idx]
        return new_state

    # Apply CZ(0,1), then CZ(1,2), etc.
    # We apply them to the identity matrix columns.
    U_new = np.zeros_like(U)
    for col in range(2**n):
        e_col = np.zeros(2**n, dtype=complex)
        e_col[col] = 1.0
        # Apply sequence
        for i in range(n - 1):
            e_col = apply_cz_seq(e_col, i, n)
        U_new[:, col] = e_col
        
    _ucz_cache[n] = U_new
    return U_new

def get_psi0(n):
    if n in _psi0_cache:
        return _psi0_cache[n]
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0 # |0...0>
    _psi0_cache[n] = psi0
    return psi0

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Get precomputed terms
    H_ZZ, H_X = get_h_terms(n)
    H = -coupling * H_ZZ - field * H_X
    
    # Find ground state energy
    eigvals = np.linalg.eigvalsh(H)
    E0 = eigvals[0].real
    
    U_CZ = get_ucz(n)
    psi0 = get_psi0(n)
    
    # Precompute Ry gate function
    def make_ry(theta):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]]) # Wait, Ry definition?
        # Standard Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        # This rotates |0> to cos(theta/2)|0> + sin(theta/2)|1>.
        # At theta=pi/2, |0> -> (|0> + |1>)/sqrt(2) = |+>. Correct.
        
    def build_full_ry_product(thetas):
        # thetas is array of length n
        # Returns 2^n x 2^n matrix
        U = np.eye(2)
        for t in thetas:
            Ry = make_ry(t)
            U = np.kron(U, Ry)
        return U

    def cost(theta_flat):
        thetas = theta_flat.reshape(layers, n)
        
        # Build U_Ry1, U_Ry2, U_Ry3
        U_Ry1 = build_full_ry_product(thetas[0])
        U_Ry2 = build_full_ry_product(thetas[1])
        U_Ry3 = build_full_ry_product(thetas[2])
        
        # Total U = U_Ry3 @ U_CZ @ U_Ry2 @ U_CZ @ U_Ry1
        # Note: U_CZ is the sequence of CZ gates.
        U_total = U_Ry3 @ U_CZ @ U_Ry2 @ U_CZ @ U_Ry1
        
        psi = U_total @ psi0
        
        # Energy
        # E = psi^dagger H psi
        # Use real part
        E = np.vdot(psi, H @ psi).real
        
        return E - E0

    # Optimization
    # Try multiple initial guesses
    best_angles = None
    best_cost = np.inf
    
    # Guess 1: All 0
    # Guess 2: All pi/2
    # Guess 3: Random
    
    initial_guesses = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.random.uniform(-np.pi, np.pi, (layers, n))
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(
                cost, 
                guess.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_angles = res.x.reshape(layers, n)
                
        except Exception:
            continue

    if best_angles is None:
        # Fallback to seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    return best_angles
