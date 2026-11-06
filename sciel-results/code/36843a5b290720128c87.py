"""
Strategy: Analytical Mean-Field Matching.
The circuit ends with a layer of single-qubit rotations (Ry). This means the final state is a product state.
We can analytically determine the optimal product state approximation to the ground state of the Ising model
using a mean-field approach. Then, we can work backwards through the circuit to find the parameters that
produce this product state.
However, because the CZ gates entangle the qubits, we cannot simply work backwards easily without simulation.
A better approach: Since the final state is a product state, we can use the variational principle for product states.
For an open chain, the optimal product state |psi> = |phi_1>...|phi_n> can be found by minimizing <psi|H|psi>.
This is a standard mean-field problem.
Once we have the target product state, we can try to find the circuit parameters.
Actually, the circuit structure is:
|0>^n -> Ry(theta_0) -> CZ -> Ry(theta_1) -> CZ -> Ry(theta_2)
The final state is a product state.
We can use a numerical optimizer but initialize it very well using the mean-field solution for the final layer,
and perhaps fix the first layers to reasonable values or optimize all together with a tight time budget.
Given the small n (4, 6) and fixed layers (3), a local optimizer like BFGS or L-BFGS-B from scipy should converge fast
if initialized well.
Let's implement a fast local optimization using scipy.optimize.minimize, initialized with a mean-field guess.
The mean-field guess for the final layer (layer 2) can be derived by assuming the state before the final layer is
close to the ground state of the Hamiltonian without the final X terms? No, that's not right.
Let's just use a robust optimizer with a good initial guess.
Initial guess: For small fields, the ground state is close to |+>^n (theta=pi/2). For large fields, it's closer to |0>^n or |1>^n.
The crossover is around field ~ coupling.
We can use a simple interpolation or just start with theta = pi/4 for all layers and let the optimizer do the work.
To ensure speed, we will use L-BFGS-B with a small number of iterations and a tight time limit.
"""

import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
            
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
            
    return H

def _get_exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _apply_circuit(theta, n, layers):
    # Start with |0>^n
    # |0> is [1, 0]
    # State vector is a tensor of shape (2,) * n
    # But for n=4,6, we can just use a flat vector of size 2^n
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0...0>
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to qubit i
        # We need to apply single-qubit gates to a multi-qubit state.
        # Efficient way: iterate over basis states? Or use tensor operations?
        # For n=6, dim=64. Iterating is fast enough.
        
        # Ry(theta) = cos(theta/2) I - i sin(theta/2) X
        # c = cos(theta/2), s = sin(theta/2)
        # |0> -> c|0> - i s |1>
        # |1> -> -i s |0> + c|1>
        
        new_state = np.zeros_like(state)
        for i in range(n):
            th = theta[layer, i]
            c = np.cos(th / 2.0)
            s = np.sin(th / 2.0)
            
            # We can apply the gate to all states at once using slicing or loops.
            # Let's do it in-place for this layer's qubit i.
            # Actually, it's safer to compute the new state from the old one.
            # Since Ry is local, we can iterate over states where bit i is 0 or 1.
            
            # Optimization: Pre-compute masks?
            # For small n, a loop is fine.
            
            # Let's create a new state for this layer's operations?
            # No, we can update in place if we are careful, but Ry mixes |0> and |1> for qubit i.
            # So we need to read from old state and write to new state, or use temporary storage.
            
            # Let's just build the new state for this layer.
            # Actually, we can apply gates one by one.
            # Let's stick to a simple loop for clarity and correctness, it's fast for n=6.
            
            # Wait, applying Ry to qubit i affects all states.
            # Let's do:
            # for state in range(dim):
            #    if bit i is 0:
            #       new_state[state] += c * state[state]
            #       new_state[state ^ (1<<i)] += -1j * s * state[state]
            #    else:
            #       new_state[state] += c * state[state]
            #       new_state[state ^ (1<<i)] += -1j * s * state[state]
            
            # This is O(n * dim) per layer. 3 layers * 6 qubits * 64 dim = 1152 ops. Very fast.
            
            # Let's implement this.
            pass # Placeholder, will implement in the actual function below
        
        # The above loop structure is getting messy. Let's implement a helper.
        
        # Re-doing the layer application properly.
        # We'll create a new state vector for the result of this layer's Ry gates.
        # Actually, we can apply Ry to each qubit sequentially.
        # Let's just use a function to apply a single-qubit gate to a specific qubit in the state vector.
        
        def apply_ry(state_vec, qubit, theta):
            dim = len(state_vec)
            new_vec = np.zeros_like(state_vec)
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << qubit
            for state in range(dim):
                if (state & mask) == 0:
                    # |0> -> c|0> - i s |1>
                    new_vec[state] += c * state_vec[state]
                    new_vec[state | mask] += -1j * s * state_vec[state]
                else:
                    # |1> -> -i s |0> + c|1>
                    new_vec[state ^ mask] += -1j * s * state_vec[state]
                    new_vec[state] += c * state_vec[state]
            return new_vec

        # Apply Ry to all qubits
        for i in range(n):
            state = apply_ry(state, i, theta[layer, i])
            
        # Apply CZ if not the last layer
        if layer < layers - 1:
            # CZ: |11> -> -|11>, others unchanged
            # We can do this by multiplying states where both adjacent bits are 1 by -1.
            for i in range(n - 1):
                mask1 = 1 << i
                mask2 = 1 << (i + 1)
                for state in range(dim):
                    if (state & mask1) and (state & mask2):
                        state[state] *= -1.0
                        
    return state

def _compute_energy(state, n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    return np.real(np.vdot(state, H @ state))

def _objective(theta_flat, n, coupling, field, layers, e0):
    theta = theta_flat.reshape((layers, n))
    state = _apply_circuit(theta, n, layers)
    energy = _compute_energy(state, n, coupling, field)
    return (energy - e0) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    e0 = _get_exact_ground_energy(n, coupling, field)
    
    # Initial guess: pi/4 for all
    theta0 = np.full((layers, n), np.pi / 4.0)
    x0 = theta0.flatten()
    
    # Use L-BFGS-B
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, coupling, field, layers, e0),
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100}
        )
        best_theta = res.x.reshape((layers, n))
        best_val = res.fun
    except Exception:
        best_theta = theta0
        best_val = _objective(x0, n, coupling, field, layers, e0)
    
    # Check time
    if time.time() > deadline - 10:
        return best_theta
        
    # If we have time, try a few random restarts to escape local minima
    # But L-BFGS-B is quite good for this smooth landscape.
    # Let's try one more restart with a different initial guess if time permits.
    if time.time() < deadline - 20:
        theta1 = np.full((layers, n), np.pi / 2.0) # |+> state
        x1 = theta1.flatten()
        try:
            res1 = minimize(
                _objective,
                x1,
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100}
            )
            if res1.fun < best_val:
                best_theta = res1.x.reshape((layers, n))
                best_val = res1.fun
        except Exception:
            pass
            
    return best_theta
