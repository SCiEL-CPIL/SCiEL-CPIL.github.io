"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

# Safety margin before the 270s wall-clock kill
_DEADLINE = 255.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def synthesize(problem):
    """
    Synthesize a quantum circuit using a fixed CNOT ladder and coordinate descent.
    """
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Define the CNOT layout: a "ladder" structure
    # For 2 qubits, alternate (0,1) and (1,0)
    # For 3 qubits, a common efficient layout cycles through pairs
    # We use a simple deterministic pattern that is known to be universal enough
    # for the specific targets (CNOT, iSWAP, SWAP, Toffoli, etc.) when combined with U3s.
    
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else: # qubits == 3
        # A standard cycle for 3 qubits to ensure connectivity
        pairs = [(0, 1), (1, 2), (2, 0)]
        
    layout = []
    # Interleave U3s and CNOTs
    # Start with U3s on all qubits
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        # Add U3s on all qubits after each CNOT to allow for arbitrary rotations
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    # Final U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * num_u3
    
    # Map parameter index to gate position and angle index
    param_map = []
    k = 0
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0)) # theta
            param_map.append((i, 1)) # phi
            param_map.append((i, 2)) # lam
            k += 3
        # CNOTs have no parameters

    def get_circuit(params):
        gates = [list(g) for g in layout]
        for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
            gates[gate_idx][2 + angle_idx] = params[p_idx]
        return gates

    def objective(params):
        if _check_deadline():
            return 1.0
        gates = get_circuit(params)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)

    # Initialize parameters to zero
    best_params = [0.0] * n_params
    best_dist = objective(best_params)
    
    rng = random.Random(4242 + qubits + budget)
    
    # Coordinate Descent
    # We perform multiple sweeps, reducing the step size each time
    max_sweeps = 15
    initial_step = math.pi / 2.0
    
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
            
        step = initial_step / (sweep + 1)
        improved = False
        
        for i in range(n_params):
            if _check_deadline():
                break
                
            current_val = best_params[i]
            
            # Try a few candidate values
            # 1. Local step up
            # 2. Local step down
            # 3. Random restart within a larger window if stuck or early
            candidates = [
                current_val + step,
                current_val - step,
                rng.uniform(-math.pi, math.pi)
            ]
            
            for trial in candidates:
                # Normalize angles to [-pi, pi] or [0, 2pi] for consistency? 
                # U3 is periodic. Keeping them in a reasonable range helps numerical stability.
                trial = (trial + math.pi) % (2 * math.pi) - math.pi
                
                new_params = best_params[:]
                new_params[i] = trial
                
                val = objective(new_params)
                
                if val < best_dist:
                    best_params = new_params
                    best_dist = val
                    improved = True
                    break # Move to next parameter immediately if improved
                    
            # If no improvement in a while, we might need a smaller step or random jump
            # The shrinking step handles convergence.
            
        if not improved and sweep > 3:
            # If we haven't improved in a few sweeps, we might be stuck.
            # Try a small random perturbation of the whole vector? 
            # Or just break if step is very small.
            if step < 1e-4:
                break

    return get_circuit(best_params)
# EVOLVE-BLOCK-END
