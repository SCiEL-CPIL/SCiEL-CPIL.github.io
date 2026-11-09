"""
Optimize shallow circuit parameters for Ising ground states using analytic gradient descent.
Strategy:
1. Construct the Hamiltonian H for the open transverse-field Ising chain.
2. Compute the exact ground energy E_0 via exact diagonalization.
3. Implement a state-vector simulator for the specific 3-layer Ry/CZ circuit.
4. Implement analytic backpropagation to compute the gradient of the energy with respect to all parameters.
5. Use the Adam optimizer to minimize the energy gap (E_circuit - E_0).
6. Perform multiple restarts with different initial parameters to avoid local minima,
   managed by a time budget to ensure completion within limits.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        # Z is diag(1, -1). Z_i Z_{i+1} is diagonal.
        # For basis state |b>, the eigenvalue is (-1)^{b_i + b_{i+1}}
        # Actually, Z|0> = |0>, Z|1> = -|1>.
        # So Z_i Z_{i+1} |b> = (-1)^{b_i} (-1)^{b_{i+1}} |b> = (-1)^{b_i + b_{i+1}} |b>.
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # XX terms: -field * sum X_i
    # X flips the bit. X_i |b> = |b ^ (1 << i)>
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            # Note: X is Hermitian, so H[b, b_flipped] is the same term.
            # The loop covers all b, so this correctly fills the matrix.
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _circuit_state(params, n, layers):
    """
    Simulate the circuit and return the final state.
    params: array of shape (layers, n)
    Circuit:
    - Start |0>^n
    - Layer 0: Ry(theta) on all qubits
    - CZ on adjacent pairs (0,1), (1,2), ..., (n-2, n-1)
    - Layer 1: Ry(theta) on all qubits
    - CZ on adjacent pairs
    - Layer 2: Ry(theta) on all qubits
    - No entangler after final layer.
    """
    dim = 1 << n
    # Start state |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Pre-compute Ry matrices for each qubit and layer? 
    # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    # We apply it to specific qubits.
    
    def apply_ry(state, qubit, theta):
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        new_state = np.zeros_like(state)
        mask = 1 << qubit
        for b in range(dim):
            if (b & mask) == 0:
                # |0> component
                b1 = b | mask
                new_state[b] += c * state[b]
                new_state[b1] += s * state[b]
            else:
                # |1> component
                b0 = b & ~mask
                new_state[b0] += -s * state[b]
                new_state[b] += c * state[b]
        return new_state

    def apply_cz(state, q1, q2):
        # CZ is diagonal: -1 if both bits are 1, else 1.
        mask1 = 1 << q1
        mask2 = 1 << q2
        new_state = np.zeros_like(state)
        for b in range(dim):
            if (b & mask1) and (b & mask2):
                new_state[b] = -state[b]
            else:
                new_state[b] = state[b]
        return new_state

    # Apply layers
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            state = apply_ry(state, q, params[l, q])
        
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for q in range(n - 1):
                state = apply_cz(state, q, q + 1)
                
    return state

def _energy_and_grad(params, n, coupling, field, layers):
    """
    Compute energy and gradient using analytic backpropagation.
    """
    dim = 1 << n
    H = _build_hamiltonian(n, coupling, field)
    
    # Forward pass: store states after each gate
    # Gates:
    # For each layer l:
    #   Ry(0), Ry(1), ..., Ry(n-1)
    #   CZ(0,1), CZ(1,2), ..., CZ(n-2, n-1) (if not last layer)
    
    # We need to be careful with the order.
    # Let's store the state after every single gate application.
    # Number of gates:
    # Layer 0: n Ry + (n-1) CZ
    # Layer 1: n Ry + (n-1) CZ
    # Layer 2: n Ry
    # Total gates = 3n + 2(n-1) = 5n - 2.
    # For n=6, 28 gates. Storing 28 states of size 64 is trivial.
    
    states = []
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    states.append(state)
    
    def apply_ry(state, qubit, theta):
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        new_state = np.zeros_like(state)
        mask = 1 << qubit
        for b in range(dim):
            if (b & mask) == 0:
                b1 = b | mask
                new_state[b] += c * state[b]
                new_state[b1] += s * state[b]
            else:
                b0 = b & ~mask
                new_state[b0] += -s * state[b]
                new_state[b] += c * state[b]
        return new_state

    def apply_cz(state, q1, q2):
        mask1 = 1 << q1
        mask2 = 1 << q2
        new_state = np.zeros_like(state)
        for b in range(dim):
            if (b & mask1) and (b & mask2):
                new_state[b] = -state[b]
            else:
                new_state[b] = state[b]
        return new_state

    # Record gate types and parameters for backprop
    # gate_types: list of ('ry', q, theta) or ('cz', q1, q2)
    gates = []
    
    for l in range(layers):
        for q in range(n):
            theta = params[l, q]
            state = apply_ry(state, q, theta)
            states.append(state)
            gates.append(('ry', q, theta))
            
        if l < layers - 1:
            for q in range(n - 1):
                state = apply_cz(state, q, q + 1)
                states.append(state)
                gates.append(('cz', q, q + 1))
                
    final_state = states[-1]
    
    # Energy
    Hpsi = H @ final_state
    energy = np.vdot(final_state, Hpsi).real
    
    # Gradient calculation
    # dE/dtheta = 2 Re( <dpsi/dtheta| H |psi> )
    # We need <dpsi/dtheta| for each parameter.
    # This is equivalent to backpropagating the vector v = H |psi> through the circuit in reverse.
    # Let v_L = H |psi_L>.
    # For a unitary gate U, if |psi_out> = U |psi_in>, then <dpsi_out/dtheta| = <dpsi_in/dtheta| U^dag.
    # So we propagate v backwards: v_in = U^dag v_out.
    # At the gate where the parameter is, we compute the gradient contribution.
    
    v = Hpsi.conj() # This is <psi| H^dag = <psi| H. Wait.
    # We want <dpsi/dtheta| H |psi>.
    # Let A = H |psi>. We want <dpsi/dtheta| A>.
    # In backprop, we usually propagate the "adjoint" of the loss.
    # Loss L = <psi| H |psi>.
    # dL/dtheta = <dpsi/dtheta| H |psi> + <psi| H |dpsi/dtheta>.
    # Since H is Hermitian, the second term is the conjugate of the first.
    # So dL/dtheta = 2 Re( <dpsi/dtheta| H |psi> ).
    
    # Let's define the vector to propagate as v = H |psi>.
    # Then <dpsi/dtheta| v> is the term we need.
    
    v = H @ final_state
    
    grads = np.zeros_like(params)
    
    # Backpropagate
    # We have states[0] ... states[N]
    # gates[0] ... gates[N-1]
    # states[k+1] = Gate_k(states[k])
    
    for k in range(len(gates) - 1, -1, -1):
        gate = gates[k]
        state_in = states[k]
        state_out = states[k+1]
        
        if gate[0] == 'ry':
            _, q, theta = gate
            # Gate U = Ry(theta) on qubit q.
            # We need to update v.
            # v_in = U^dag v_out
            # U = Ry(theta) = [[c, -s], [s, c]] in the subspace of qubit q.
            # U^dag = Ry(-theta) = [[c, s], [-s, c]]
            
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # v_in is initially v (from the next step)
            v_in = np.zeros_like(v)
            mask = 1 << q
            for b in range(dim):
                if (b & mask) == 0:
                    b1 = b | mask
                    # v_out[b] comes from c*v_in[b] - s*v_in[b1] ? No.
                    # v_out = U v_in.
                    # v_out[b] = c v_in[b] - s v_in[b1]
                    # v_out[b1] = s v_in[b] + c v_in[b1]
                    # We have v_out. We want v_in.
                    # v_in = U^dag v_out.
                    # v_in[b] = c v_out[b] + s v_out[b1]
                    # v_in[b1] = -s v_out[b] + c v_out[b1]
                    
                    v_in[b] = c * v[b] + s * v[b1]
                    v_in[b1] = -s * v[b] + c * v[b1]
                else:
                    # Skip, handled in 0 case
                    pass
            
            # Gradient contribution
            # dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # <dpsi/dtheta| = <psi_in| dU/dtheta
            # Term = <psi_in| dU/dtheta | v_out> ? No.
            # We are at gate k. state_out is the state AFTER the gate.
            # v is the adjoint vector arriving at this gate from the right (which is v_out in forward pass).
            # The gradient term is <state_in| (dU/dtheta) |v> ?
            # Let's check dimensions.
            # L = <psi_out| A>.
            # psi_out = U psi_in.
            # dL/dtheta = <dU/dtheta psi_in| A> = <psi_in| (dU/dtheta)^dag | A>.
            # Here A is the vector v (which is H|psi_final> propagated back).
            # So term = <psi_in| (dU/dtheta)^dag | v>.
            # (dU/dtheta)^dag = (dU/dtheta)^T since it's real.
            # dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # (dU/dtheta)^dag = 0.5 * [[-s, c], [-c, -s]]
            
            # Let's compute the inner product.
            # We need to sum over the subspace.
            # Let's just compute the vector w = (dU/dtheta)^dag v.
            # Then term = <state_in| w>.
            
            w = np.zeros_like(v)
            for b in range(dim):
                if (b & mask) == 0:
                    b1 = b | mask
                    # w[b] = -0.5*s * v[b] + 0.5*c * v[b1]
                    # w[b1] = -0.5*c * v[b] - 0.5*s * v[b1]
                    w[b] = -0.5 * s * v[b] + 0.5 * c * v[b1]
                    w[b1] = -0.5 * c * v[b] - 0.5 * s * v[b1]
                
            # Inner product <state_in| w>
            # state_in is complex.
            contrib = np.vdot(state_in, w).real
            grads[0, q] += contrib # Wait, params index.
            # We need to map gate index to param index.
            # The gate is in layer l, qubit q.
            # We need to know l.
            # I didn't store l in the gate tuple. Let's fix that.
            
            # Actually, I can just store the param index directly.
            
            # Let's restart the gate recording with param indices.
            
        else:
            # CZ gate
            # U = CZ. dU/dtheta = 0.
            # No gradient contribution.
            # v_in = U^dag v_out = U v_out (since CZ is Hermitian and unitary).
            # CZ is diagonal, so v_in = v_out.
            # Wait, v_in = v_out for CZ?
            # U = diag(1,1,1,-1). U^dag = U.
            # So v_in = U v_out.
            # But since CZ is diagonal, v_in[b] = U[b,b] v_out[b].
            mask1 = 1 << gate[1]
            mask2 = 1 << gate[2]
            for b in range(dim):
                if (b & mask1) and (b & mask2):
                    v[b] = -v[b]
                # else v[b] stays same
                
            # No gradient update.
            
        # Update v for next step
        if gate[0] == 'ry':
             # We computed v_in above.
             # But my loop structure was messy.
             pass
             
    # This implementation is getting messy. Let's simplify by storing param indices.
    
    # Revised Backprop
    v = H @ final_state
    grads = np.zeros_like(params)
    
    # We need to map gates to param indices.
    # Let's rebuild the gate list with indices.
    param_idx = 0
    gate_info = []
    for l in range(layers):
        for q in range(n):
            gate_info.append(('ry', q, l, q, param_idx))
            param_idx += 1
        if l < layers - 1:
            for q in range(n - 1):
                gate_info.append(('cz', q, q+1, -1, -1))
                
    # Reverse iterate
    for k in range(len(gate_info) - 1, -1, -1):
        g = gate_info[k]
        if g[0] == 'ry':
            _, q, l, q_idx, p_idx = g
            theta = params[l, q_idx]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            v_in = np.zeros_like(v)
            w = np.zeros_like(v)
            mask = 1 << q
            for b in range(dim):
                if (b & mask) == 0:
                    b1 = b | mask
                    # v_in = U^dag v
                    v_in[b] = c * v[b] + s * v[b1]
                    v_in[b1] = -s * v[b] + c * v[b1]
                    
                    # w = (dU/dtheta)^dag v
                    # dU/dtheta = 0.5 [[-s, -c], [c, -s]]
                    # (dU/dtheta)^dag = 0.5 [[-s, c], [-c, -s]]
                    w[b] = -0.5 * s * v[b] + 0.5 * c * v[b1]
                    w[b1] = -0.5 * c * v[b] - 0.5 * s * v[b1]
            
            state_in = states[k]
            contrib = np.vdot(state_in, w).real
            grads[l, q_idx] += contrib
            
            v = v_in
        else:
            # CZ
            _, q1, q2, _, _ = g
            mask1 = 1 << q1
            mask2 = 1 << q2
            # v_in = U^dag v = U v
            for b in range(dim):
                if (b & mask1) and (b & mask2):
                    v[b] = -v[b]
            # No grad update
            
    return energy, grads

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    E0 = _ground_energy(n, coupling, field)
    
    # Adam Optimizer
    def adam_step(params, grads, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return params, m, v, t + 1

    best_params = None
    best_energy = np.inf
    
    # Multiple restarts
    num_restarts = 10
    steps_per_restart = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random init
        params = np.random.uniform(-np.pi, np.pi, (layers, n))
        
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 1
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            energy, grads = _energy_and_grad(params, n, coupling, field, layers)
            
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
                
            params, m, v, t = adam_step(params, grads, m, v, t)
            
    if best_params is None:
        # Fallback to seed-like behavior if no time
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    return best_params
# EVOLVE-BLOCK-END
