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


def _build_and_distance(gates, qubits, target):
    """
    Efficiently computes the circuit unitary U and returns the distance to target.
    Uses in-place updates to avoid heavy np.kron operations.
    """
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    
    # Precompute masks
    # qubit 0 is MSB. Bit position for qubit q is qubits - 1 - q.
    masks = [1 << (qubits - 1 - q) for q in range(qubits)]
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            e_lam = np.exp(1j * lam)
            e_phi = np.exp(1j * phi)
            e_phi_lam = np.exp(1j * (phi + lam))
            
            # U_single = [[c, -e_lam*s], [e_phi*s, e_phi_lam*c]]
            u00 = c
            u01 = -e_lam * s
            u10 = e_phi * s
            u11 = e_phi_lam * c
            
            mask = masks[q]
            
            # Iterate over columns where the qubit bit is 0
            # We update pairs (j, j|mask)
            for j in range(dim):
                if (j & mask) == 0:
                    j0 = j
                    j1 = j | mask
                    # Get old columns
                    col0 = U[:, j0]
                    col1 = U[:, j1]
                    # Compute new columns
                    # We must use old col0 and col1 for both new columns
                    new_col0 = u00 * col0 + u01 * col1
                    new_col1 = u10 * col0 + u11 * col1
                    # Assign back
                    U[:, j0] = new_col0
                    U[:, j1] = new_col1
        else:
            _, control, target_q = gate
            mask_c = masks[control]
            mask_t = masks[target_q]
            
            # CNOT: If control bit is 1, flip target bit.
            # This permutes the rows of U.
            # For each row i where control bit is 1, swap row i with row i ^ mask_t.
            # We can do this by iterating i where (i & mask_c) != 0.
            # To avoid double swapping, we only swap if i < i^mask_t? 
            # Actually, if we swap (i, i^mask_t), and later encounter (i^mask_t), we swap back.
            # So we should only perform the swap for a specific subset, e.g., where target bit is 0?
            # If we swap rows i and j, doing it twice is identity.
            # Let's just iterate all i, and if (i & mask_c) != 0, swap row i and row i^mask_t.
            # This will swap every pair twice? No.
            # If i has control=1, we swap i and i^mask_t.
            # i^mask_t will have control=1 (since mask_t is different bit).
            # So when we reach i^mask_t, we will swap it back with i.
            # So we need to ensure we only swap once.
            # Condition: (i & mask_c) != 0 AND (i & mask_t) == 0?
            # If we pick i such that control=1 and target=0, then i^mask_t has control=1 and target=1.
            # We swap them. We never process the "target=1" case as the primary swap initiator if we restrict to target=0.
            for i in range(dim):
                if (i & mask_c) != 0 and (i & mask_t) == 0:
                    j = i ^ mask_t
                    # Swap rows i and j
                    temp = U[i, :].copy()
                    U[i, :] = U[j, :]
                    U[j, :] = temp

    # Calculate distance: 1 - |Tr(target^dagger U)| / dim
    # Tr(target^dagger U) = sum(target^dagger[i, j] * U[j, i])
    # Which is sum(target[i, j]^* * U[j, i])
    # This is the Frobenius inner product <target, U> = sum(target^* elementwise U)
    # np.vdot(target, U) does exactly this: conjugates first arg, multiplies elementwise, sums.
    inner = np.vdot(target, U)
    dist = 1.0 - min(1.0, abs(inner) / dim)
    return dist


def synthesize(problem):
    """
    Synthesize a quantum circuit using a fixed CNOT ladder and coordinate descent.
    Optimized with fast in-place matrix updates.
    """
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Define the CNOT layout
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else: # qubits == 3
        pairs = [(0, 1), (1, 2), (2, 0)]
        
    layout = []
    # Interleave U3s and CNOTs
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * num_u3
    
    # Map parameter index to gate position and angle index
    param_map = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0))
            param_map.append((i, 1))
            param_map.append((i, 2))

    def get_circuit(params):
        gates = [list(g) for g in layout]
        for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
            gates[gate_idx][2 + angle_idx] = params[p_idx]
        return gates

    def objective(params):
        if _check_deadline():
            return 1.0
        gates = get_circuit(params)
        return _build_and_distance(gates, qubits, target)

    # Initialize parameters to zero
    best_params = [0.0] * n_params
    best_dist = objective(best_params)
    
    rng = random.Random(4242 + qubits + budget)
    
    # Coordinate Descent
    # Since _build is faster, we can afford more sweeps and finer steps.
    max_sweeps = 25  # Increased from 15
    
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
            
        # Adaptive step size: start large, shrink
        # Use a schedule that allows exploration then exploitation
        step = math.pi / (1 + sweep)
        
        improved = False
        
        for i in range(n_params):
            if _check_deadline():
                break
                
            current_val = best_params[i]
            
            # Candidates: local steps and a random jump
            # We try 3 local candidates and 1 random
            candidates = [
                current_val + step,
                current_val - step,
                rng.uniform(-math.pi, math.pi)
            ]
            
            # Sort candidates by distance to current? No, just try them.
            # To save time, we can stop early if we find a good improvement?
            # But coordinate descent usually evaluates all to find the best local move.
            # Given the speed, evaluating 3-4 candidates is cheap.
            
            best_trial_val = best_dist
            best_trial_param = current_val
            
            for trial in candidates:
                trial = (trial + math.pi) % (2 * math.pi) - math.pi
                
                new_params = best_params[:]
                new_params[i] = trial
                
                val = objective(new_params)
                
                if val < best_trial_val:
                    best_trial_val = val
                    best_trial_param = trial
            
            if best_trial_val < best_dist:
                best_params[i] = best_trial_param
                best_dist = best_trial_val
                improved = True
            
        if not improved and sweep > 5:
            if step < 1e-3:
                break

    return get_circuit(best_params)
# EVOLVE-BLOCK-END
