import math
import time
import numpy as np
import scipy.linalg

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 230.0  # Safety margin from 270s kill

def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    """
    Fast build of the unitary matrix from a list of gates.
    Qubit 0 is MSB.
    """
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], 
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            
            # Apply single qubit gate on qubit q
            # Reshape matrix to (2, 2, ..., 2)
            # Indices: qubit 0 is the first axis (MSB)
            # We need to apply `single` to axis `q`.
            # np.einsum is fast.
            # Let's use a helper to apply local gate.
            
            # Move axis q to the front? No, just use einsum.
            # matrix has shape (2, 2, ..., 2)
            # We want to multiply by single on axis q.
            
            # Create indices
            # This is tricky with einsum directly for arbitrary q.
            # Alternative: Use reshape and broadcasting.
            
            # Let's stick to a robust, slightly slower but correct method for small N (8 or 16).
            # Reshape to 2D where rows are all other qubits and cols are qubit q.
            
            # Permute axes so that q is the last axis?
            # Original axes: 0, 1, ..., q-1, q, q+1, ..., N-1
            # New axes: 0, 1, ..., q-1, q+1, ..., N-1, q
            axes = list(range(qubits))
            axes.remove(q)
            axes.append(q)
            
            reshaped = matrix.transpose(axes).reshape(-1, 2)
            new_reshaped = reshaped @ single.T # Wait, single acts on the state vector.
            # If matrix is M, and we apply U on qubit q, the new matrix M' satisfies:
            # M' = (I ... U ... I) M
            # In the reshaped form (others, q), M is a block.
            # Actually, M_{i, j} where j includes qubit q index.
            # Let's just use the kron method but optimized? No, kron is slow.
            
            # Let's use the explicit loop for small N.
            # N=8 or 16. 400 gates. 64000 ops. Python loops are slow.
            # Let's use NumPy advanced indexing.
            
            # For u3 on qubit q:
            # The operator is O = I (x) ... U ... I
            # We can construct O efficiently?
            # Or just apply to matrix.
            
            # Let's try the reshape trick again.
            # M is (2, 2, ..., 2)
            # We want to apply U to axis q.
            # M_new[i0, ..., iq, ..., iN] = sum_k M[i0, ..., k, ..., iN] * U[iq, k]
            # This is a matrix multiplication along axis q.
            
            # Move axis q to the end.
            axes_move = list(range(qubits))
            axes_move.remove(q)
            axes_move.append(q)
            M_perm = matrix.transpose(axes_move)
            # Now M_perm has shape (2, 2, ..., 2, 2) where the last axis is q.
            # Reshape to (-1, 2)
            M_flat = M_perm.reshape(-1, 2)
            # Apply U
            # M_flat_new[:, iq] = sum_k M_flat[:, k] * U[iq, k]
            # This is M_flat @ U.T
            M_flat_new = M_flat @ single.T
            # Reshape back
            M_perm_new = M_flat_new.reshape([2] * qubits)
            # Move axis q back to position q
            axes_back = list(range(qubits))
            axes_back.remove(q) # Remove last
            axes_back.insert(q, q) # Insert at q? 
            # Wait, axes_move put q at the end.
            # To reverse, we need to move the last axis back to position q.
            # The current axes order is 0, ..., q-1, q+1, ..., N-1, q.
            # We want 0, ..., q, ..., N-1.
            # So we move the last element to position q.
            axes_restore = list(range(qubits - 1)) + [qubits - 1] # No
            # Let's just use np.transpose with the inverse permutation.
            # Original: 0..N-1. Move q to end: 0..q-1, q+1..N-1, q.
            # Inverse: Move last to q.
            # The inverse permutation of `axes_move` (which is the order of new axes relative to old)
            # `transpose(axes_move)` means new_axis[i] = old_axis[axes_move[i]].
            # We want to go back.
            # Let's just create the inverse permutation list.
            inv_axes = [0] * qubits
            for i, ax in enumerate(axes_move):
                inv_axes[ax] = i
            # Now M_perm_new.transpose(inv_axes) should restore the original order.
            matrix = M_perm_new.transpose(inv_axes)
            
        else:
            _, control, target_q = gate
            # CNOT: control -> target_q
            # Flip target_q if control is 1.
            # This is a permutation of rows and columns?
            # M' = CNOT M CNOT (since CNOT is self-inverse and unitary)
            # Actually, M' = CNOT_op M.
            # CNOT_op swaps |c=1, t=0> with |c=1, t=1> in the basis.
            # Indices:
            # We need to swap rows and columns where control bit is 1.
            
            # Create an index array
            indices = np.arange(dim)
            # Bit for control
            c_bit = (indices >> (qubits - 1 - control)) & 1
            # Bit for target
            t_bit = (indices >> (qubits - 1 - target_q)) & 1
            
            # We want to permute the matrix.
            # P is a permutation matrix such that P M P^T = M'?
            # Actually, the operator is O. M_new = O M.
            # O swaps basis states |...1...0...> and |...1...1...> where control is 1.
            # So we swap the rows corresponding to these states?
            # And since O is symmetric, we swap columns too?
            # No, M_new = O M.
            # Row i of M_new is Row (O i) of M.
            # So we permute rows by the permutation defined by O.
            # And since we are building the unitary matrix of the circuit, we are left-multiplying.
            # So we just need to permute the rows of `matrix` according to the CNOT permutation.
            
            # Permutation:
            # For each index i:
            # If control bit is 1, flip target bit.
            # Else, keep same.
            
            perm = indices.copy()
            mask = (c_bit == 1)
            # Flip target bit for indices where control is 1
            # target_bit_pos = qubits - 1 - target_q
            # perm[mask] = perm[mask] ^ (1 << (qubits - 1 - target_q))
            
            # Wait, indices are the row indices.
            # We want M_new[i, :] = M[perm[i], :]
            # So we need perm such that perm[i] is the original row index that maps to i.
            # CNOT maps |10> to |11> and |11> to |10>.
            # So Row 2 (10) of M_new comes from Row 3 (11) of M?
            # Let's check.
            # O |10> = |11>.
            # <11| O = <10|.
            # M_new = O M.
            # (O M)_{ij} = sum_k O_{ik} M_{kj}.
            # O_{ik} is 1 if k = O(i).
            # So (O M)_{ij} = M_{O(i), j}.
            # So Row i of M_new is Row O(i) of M.
            
            # So we need to permute rows by O(i).
            # O(i) flips target bit if control bit of i is 1.
            
            new_indices = indices.copy()
            # For each i in indices:
            # if control bit of i is 1, flip target bit of i to get O(i).
            # Wait, O acts on the basis vector |i>.
            # So O(|i>) = |j> where j is i with target flipped if control is 1.
            # So we replace row i with row j.
            
            perm = indices.copy()
            # We need to calculate j for each i.
            # j = i ^ (1 << t_pos) if (i >> c_pos) & 1 else i
            
            c_pos = qubits - 1 - control
            t_pos = qubits - 1 - target_q
            
            # Vectorized calculation of perm
            # perm[i] = i ^ (1 << t_pos) if (i >> c_pos) & 1 else i
            
            # Create mask for i where control bit is 1
            mask = (indices >> c_pos) & 1
            perm[mask] = indices[mask] ^ (1 << t_pos)
            
            matrix = matrix[perm, :]
            
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _u3(theta, phi, lam):
    return ["u3", 0, theta, phi, lam]

def _apply_u3(gates, qubit, theta, phi, lam):
    gates.append(["u3", qubit, theta, phi, lam])

def _cx(gates, control, target):
    gates.append(["cx", control, target])

def _decompose_2qubit_kak(U):
    """
    Decompose a 2-qubit unitary U into a circuit with 3 CNOTs.
    """
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # KAK Decomposition
    # Reference: "Synthesis of Two-Qubit Quantum Gates"
    # We use the algorithm from Qiskit or similar.
    
    # 1. Extract global phase
    det_U = np.linalg.det(U)
    alpha = -0.5j * np.log(det_U)
    U_eff = U * np.exp(-alpha)
    
    # 2. Compute the "matching" or KAK parameters.
    # We can use the method of finding A, B, C, D such that
    # U_eff = (A ⊗ B) E (C ⊗ D)
    # where E is the entangling part.
    
    # A robust way is to use the SVD of U_eff in the Liouville space?
    # Or use the explicit formula.
    
    # Let's use the algorithm from "Efficient Quantum Circuits for Arbitrary Two-Qubit Operations"
    # 1. Compute M = U_eff
    # 2. Find A, B, C, D by solving:
    #    A = (1/2) (I + Tr_2(M)) ? No.
    
    # Let's use a known implementation strategy:
    # 1. Compute the local invariants.
    # 2. Determine the KAK angles.
    # 3. Construct the single-qubit gates.
    
    # This is complex. Let's use a numerical optimization as a fallback if KAK fails?
    # No, KAK is standard.
    
    # I will implement the KAK decomposition using the method described in 
    # "Quantum Computation and Quantum Information" (Nielsen & Chuang) or similar.
    
    # Step 1: Compute the "canonical" form parameters.
    # The entangling part is exp(i (b1 XX + b2 YY + b3 ZZ)).
    # The coefficients b1, b2, b3 can be found from the traces:
    # K1 = 0.5 * Tr(U_eff (X⊗X))
    # K2 = 0.5 * Tr(U_eff (Y⊗Y))
    # K3 = 0.5 * Tr(U_eff (Z⊗Z))
    # However, this is only true if A, B, C, D are identity.
    
    # Correct approach:
    # 1. Compute A, B, C, D.
    # 2. Compute the remaining part E = (A^† ⊗ B^†) U_eff (C^† ⊗ D^†).
    # 3. E is in the canonical form.
    
    # To find A, B, C, D:
    # We can use the fact that the singular values of the partial traces are invariant.
    # But a simpler numerical method is to optimize A, B, C, D to minimize the distance to the canonical form.
    # Given time, I will use a simplified KAK implementation.
    
    # Let's use the code from a reliable source or derive it.
    # Actually, for the specific gates in the task (CNOT, iSWAP, SWAP, Toffoli, and 3 generic),
    # the generic ones might be simple.
    
    # I will use a generic 3-CNOT optimizer with a good initial guess (zeros) and Nelder-Mead.
    # This is robust and less prone to analytic errors.
    
    def circuit_from_params(params):
        gates = []
        idx = 0
        # Structure: U3 U3 CX U3 U3 CX U3 U3 CX U3 U3
        # 12 u3 gates, 3 cx gates.
        # Params: 12 * 3 = 36 params.
        
        # Layer 1
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        _cx(gates, 0, 1)
        
        # Layer 2
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        _cx(gates, 0, 1)
        
        # Layer 3
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        _cx(gates, 0, 1)
        
        # Final
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        
        return gates

    # Optimization
    best_params = [0.0] * 36
    best_dist = 1.0
    
    def eval_dist(params):
        gates = circuit_from_params(params)
        U_calc = _build(gates, 2)
        return _distance(U, U_calc)
    
    best_dist = eval_dist(best_params)
    
    # Use Nelder-Mead from scipy if available, else coordinate descent.
    try:
        from scipy.optimize import minimize
        res = minimize(eval_dist, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            best_params = res.x
            best_dist = res.fun
    except:
        # Fallback to coordinate descent
        step = 0.5
        for sweep in range(20):
            if _remaining_time() < 5:
                break
            step = 0.5 / (1 + sweep * 0.1)
            improved = False
            for i in range(36):
                for trial in (best_params[i] + step, best_params[i] - step):
                    new_params = best_params[:]
                    new_params[i] = trial
                    d = eval_dist(new_params)
                    if d < best_dist:
                        best_params = new_params
                        best_dist = d
                        improved = True
            if not improved:
                break

    gates = circuit_from_params(best_params)
    return gates

def _decompose_swap():
    gates = []
    _cx(gates, 0, 1)
    _cx(gates, 1, 0)
    _cx(gates, 0, 1)
    return gates

def _decompose_iswap():
    gates = []
    _apply_u3(gates, 0, math.pi/2, 0, math.pi/2)
    _apply_u3(gates, 1, math.pi/2, 0, math.pi/2)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, -math.pi/2, 0)
    _apply_u3(gates, 1, math.pi/2, math.pi/2, 0)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, -math.pi/2, 0)
    _apply_u3(gates, 1, math.pi/2, math.pi/2, 0)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, 0, math.pi/2)
    _apply_u3(gates, 1, math.pi/2, 0, math.pi/2)
    return gates

def _decompose_toffoli():
    gates = []
    _apply_u3(gates, 2, math.pi/4, 0, 0)
    _cx(gates, 1, 2)
    _apply_u3(gates, 2, -math.pi/4, 0, 0)
    _cx(gates, 0, 2)
    _apply_u3(gates, 2, math.pi/4, 0, 0)
    _cx(gates, 1, 2)
    _apply_u3(gates, 2, -math.pi/4, 0, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, math.pi/2)
    _cx(gates, 0, 1)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, -math.pi/2)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, 0, 0)
    _apply_u3(gates, 0, 0, 0, math.pi/2)
    _cx(gates, 1, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, -math.pi/2)
    _cx(gates, 1, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, math.pi/2)
    return gates

def _identify_gate(target, qubits):
    if qubits == 2:
        cnot = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot):
            return "cnot"
        swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)
        if np.allclose(target, swap):
            return "swap"
        iswap = 1/np.sqrt(2) * np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
        if np.allclose(target, iswap):
            return "iswap"
    elif qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        toffoli[6,6] = 0
        toffoli[7,7] = 0
        if np.allclose(target, toffoli):
            return "toffoli"
    return None

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    gate_type = _identify_gate(target, qubits)
    
    if gate_type == "cnot":
        return [["cx", 0, 1]]
    elif gate_type == "swap":
        return _decompose_swap()
    elif gate_type == "iswap":
        return _decompose_iswap()
    elif gate_type == "toffoli":
        return _decompose_toffoli()
        
    if qubits == 2:
        return _decompose_2qubit_kak(target)
    elif qubits == 3:
        # Check for 2-qubit substructures
        # Pair (0,1)
        if np.allclose(target[0:4, 4:8], 0) and np.allclose(target[4:8, 0:4], 0):
            if np.allclose(target[0:4, 0:4], target[4:8, 4:8]):
                u2 = target[0:4, 0:4]
                return _decompose_2qubit_kak(u2)
                
        # Pair (1,2)
        # Check if U commutes with qubit 0?
        # If U = I ⊗ U2 (on qubits 1,2), then U[0:4, 4:8] != 0 generally.
        # We need to check if U is block diagonal in qubit 0.
        # U = |0><0| ⊗ U_0 + |1><1| ⊗ U_1
        # If U_0 == U_1, then it's I ⊗ U2.
        
        # Check pair (0,2)
        # This requires reordering.
        
        # Fallback: Generic 3-qubit optimizer
        # Use a 7-CNOT structure?
        # For now, return empty or a simple structure if not identified.
        # But the task implies exact synthesis is possible.
        # I will use a generic optimizer with a 9-CNOT budget if available.
        
        def circuit_3q(params):
            gates = []
            idx = 0
            # 3 layers of CX(0,1) and CX(1,2)
            # Total 6 CX?
            # Let's use 7 CX: 0-1, 1-2, 0-1, 1-2, 0-1, 1-2, 0-1
            
            for layer in range(7):
                _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
                _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
                _apply_u3(gates, 2, params[idx+6], params[idx+7], params[idx+8])
                idx += 9
                if layer % 2 == 0:
                    _cx(gates, 0, 1)
                else:
                    _cx(gates, 1, 2)
                    
            _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
            _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
            _apply_u3(gates, 2, params[idx+6], params[idx+7], params[idx+8])
            idx += 9
            return gates

        best_params = [0.0] * (8 * 9) # 8 layers * 9 params
        best_dist = 1.0
        
        def eval_dist_3q(params):
            gates = circuit_3q(params)
            U_calc = _build(gates, 3)
            return _distance(target, U_calc)
            
        best_dist = eval_dist_3q(best_params)
        
        # Optimize
        step = 0.5
        for sweep in range(10): # Limit sweeps for 3-qubit
            if _remaining_time() < 10:
                break
            step = 0.5 / (1 + sweep * 0.1)
            improved = False
            for i in range(72):
                for trial in (best_params[i] + step, best_params[i] - step):
                    new_params = best_params[:]
                    new_params[i] = trial
                    d = eval_dist_3q(new_params)
                    if d < best_dist:
                        best_params = new_params
                        best_dist = d
                        improved = True
            if not improved:
                break
                
        return circuit_3q(best_params)

# EVOLVE-BLOCK-END
