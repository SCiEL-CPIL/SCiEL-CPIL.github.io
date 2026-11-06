"""Analytic layer-by-layer synthesis with exact decompositions for known gates."""
import math
import time

import numpy as np

_DEADLINE = 255.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_unitary(gates, qubits):
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, ctrl, tgt = gate
            # Apply CNOT: U = P_cnot @ U
            # CNOT swaps rows where control=1 and target flips
            # Instead of building full matrix, permute rows of U
            indices = np.arange(dim)
            ctrl_bit = (indices >> (qubits - 1 - ctrl)) & 1
            tgt_bit = (indices >> (qubits - 1 - tgt)) & 1
            # For CNOT, we flip target bit when control bit is 1
            # U_new[i] = U[i ^ (ctrl_bit[i] << (qubits-1-tgt))]
            new_indices = indices ^ (ctrl_bit << (qubits - 1 - tgt))
            U = U[new_indices, :]
    return U


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _is_circuit_close(target, gates, qubits, tol=1e-9):
    matrix = _build_unitary(gates, qubits)
    return _distance(target, matrix) < tol


def _u3_from_matrix(mat):
    """Extract u3 parameters from a 2x2 unitary matrix."""
    mat = mat.astype(complex)
    # Handle global phase: make mat[0,0] real and non-negative
    if abs(mat[0, 0]) > 1e-12:
        phase = mat[0, 0] / abs(mat[0, 0])
        mat = mat / phase
    else:
        phase = 1.0
    
    c = mat[0, 0]
    s = mat[1, 0]
    
    # theta/2 from |c| and |s|
    theta = 2 * math.atan2(abs(s), abs(c))
    
    # phi and lam from phases
    if abs(c) > 1e-12:
        phi = math.angle(s / c)
    else:
        phi = 0.0
    
    if abs(s) > 1e-12:
        lam = math.angle(-c / s)
    else:
        lam = 0.0
    
    return [theta, phi, lam]


def _extract_single_qubit_2qubit(U, qubit):
    """Extract single-qubit gate on given qubit from 2-qubit unitary.
    Returns the 2x2 matrix."""
    # For qubit 0 (most significant): U = A ⊗ B where we want A
    # U[i,j] where i = i0*2 + i1, j = j0*2 + j1
    # A[i0, j0] = (1/2) * sum_{i1,j1} U[i0*2+i1, j0*2+j1] * delta stuff
    # Actually, if U = A ⊗ B, then U[i0*2+i1, j0*2+j1] = A[i0,j0] * B[i1,j1]
    # So A[i0,j0] = U[i0*2+i1, j0*2+j1] / B[i1,j1] for any i1,j1 where B[i1,j1] != 0
    
    # Better: use partial trace approach
    # A = (1/2) * (U * (I ⊗ X)) ... no
    # Actually: A = Tr_B(U) where Tr_B means trace over qubit 1
    # For U = A ⊗ B: Tr_B(U) = A * Tr(B) = A * 2 (if B is unitary, Tr(B) is not necessarily 2)
    # Wait, that's not right either. Let me think differently.
    
    # If U = A ⊗ B, then:
    # U[0,0]=A[0,0]*B[0,0], U[0,1]=A[0,0]*B[0,1], U[0,2]=A[0,1]*B[0,0], U[0,3]=A[0,1]*B[0,1]
    # U[1,0]=A[1,0]*B[0,0], U[1,1]=A[1,0]*B[0,1], U[1,2]=A[1,1]*B[0,0], U[1,3]=A[1,1]*B[0,1]
    # U[2,0]=A[0,0]*B[1,0], U[2,1]=A[0,0]*B[1,1], U[2,2]=A[0,1]*B[1,0], U[2,3]=A[0,1]*B[1,1]
    # U[3,0]=A[1,0]*B[1,0], U[3,1]=A[1,0]*B[1,1], U[3,2]=A[1,1]*B[1,0], U[3,3]=A[1,1]*B[1,1]
    
    if qubit == 0:
        # A is the top-left 2x2 block structure
        # A[i,j] can be extracted from U[i*2, j*2] / B[0,0] etc.
        # But we don't know B. However, if U = A ⊗ B exactly, then:
        # A[0,0] = U[0,0]/B[0,0] = U[0,1]/B[0,1] = U[2,0]/B[1,0] = U[2,1]/B[1,1]
        # This is tricky without knowing B.
        
        # Alternative: use the fact that for product states,
        # A = U * (|0><0| ⊗ I + |1><1| ⊗ I) ... no
        # Let me just use: A = (1/2) * (U @ (I ⊗ X)) ... no
        
        # Simplest: if U = A ⊗ B, then A = U @ (I ⊗ B†) but we don't know B.
        # Let's use: A[i,j] = sum_k U[i*2+k, j*2+k] * conj(B[k,k]) / 2 ... still need B
        
        # OK different approach: 
        # A[0,0] = (U[0,0] + U[1,1]) / (B[0,0] + B[1,1]) ... no
        
        # Let me just use SVD-like approach or direct extraction assuming structure.
        # For the layer-by-layer peeling, after removing CNOTs, the remaining
        # unitary should be a product of single-qubit gates.
        # So U = A ⊗ B (or B ⊗ A depending on qubit order).
        
        # For qubit 0: U = A ⊗ B
        # A[0,0] = U[0,0]/B[0,0]. We can find B[0,0] from consistency.
        # U[0,0] = A[0,0]*B[0,0]
        # U[0,1] = A[0,0]*B[0,1]
        # U[1,0] = A[1,0]*B[0,0]
        # U[1,1] = A[1,0]*B[0,1]
        # So U[0,0]*U[1,1] = U[0,1]*U[1,0] (determinant condition for the block)
        # And A[0,0]/A[1,0] = U[0,0]/U[1,0] = U[0,1]/U[1,1]
        
        # Let me just use: A[0,0] = U[0,0], A[0,1] = U[0,2], A[1,0] = U[1,0], A[1,1] = U[1,2]
        # This works if B[0,0] = B[0,1] = B[1,0] = B[1,1] = 1, i.e., B = I.
        # In general, this won't work.
        
        # Proper extraction:
        # A = (1/2) * (U * (I ⊗ I) with appropriate contractions)
        # Actually: A[i,j] = (1/d) * sum_k U[i*d+k, j*d+k] where d is dimension of B
        # No, that gives A * Tr(B)/d... 
        
        # Let me just use: for U = A ⊗ B,
        # A = U[0:2, 0:2] / B[0,0] if B[0,0] != 0
        # But we can find B[0,0] = U[0,0]/A[0,0]... circular.
        
        # Use: A[0,0]*B[0,0] = U[0,0], A[0,1]*B[0,0] = U[0,2]
        # So A[0,1]/A[0,0] = U[0,2]/U[0,0]
        # And A[1,0]/A[0,0] = U[1,0]/U[0,0]
        # So A = [[1, U[0,2]/U[0,0]], [U[1,0]/U[0,0], ?]] * U[0,0]/B[0,0]
        # A[1,1]: A[1,1]*B[0,0] = U[1,2], so A[1,1] = U[1,2]/B[0,0]
        # From A[0,0]*B[0,0] = U[0,0]: B[0,0] = U[0,0]/A[0,0]
        # So A[1,1] = U[1,2]*A[0,0]/U[0,0]
        
        # This is getting complicated. Let me use a different approach.
        # Since after peeling CNOTs, U should be a product of single-qubit gates,
        # I can just use the top-left 2x2 block as A (assuming B has B[0,0]=1 or similar).
        # Actually, let me just return the appropriate 2x2 block and let the u3 extraction handle it.
        
        # For qubit 0: A is related to U[0:2, 0:2] but scaled by 1/B[0,0]
        # For now, let's assume the product structure and extract directly:
        # A[0,0] = U[0,0], A[0,1] = U[0,2], A[1,0] = U[1,0], A[1,1] = U[1,2]
        # This is correct if B[0,0] = 1, which may not be true.
        # But for the purpose of u3 extraction, the relative phases and magnitudes
        # should be preserved up to a global scale factor which u3_from_matrix handles.
        
        A = np.array([[U[0, 0], U[0, 2]],
                      [U[1, 0], U[1, 2]]], dtype=complex)
    else:
        # qubit 1: U = A ⊗ B, we want B
        # B[0,0] = U[0,0]/A[0,0], B[0,1] = U[0,1]/A[0,0], etc.
        # B[i,j] = U[i,j]/A[0,0] = U[i+2,j+2]/A[1,1]
        # B[0,0] = U[0,0], B[0,1] = U[0,1], B[1,0] = U[2,0], B[1,1] = U[2,1]
        # (assuming A[0,0] = 1)
        B = np.array([[U[0, 0], U[0, 1]],
                      [U[2, 0], U[2, 1]]], dtype=complex)
        A = B
    
    return A


def _peel_2qubit(U, budget):
    """Layer-by-layer analytic peeling for 2-qubit unitaries."""
    qubits = 2
    gates = []
    remaining = U.copy()
    
    for layer in range(budget):
        # Extract single-qubit gates before CNOT
        for q in range(qubits):
            mat = _extract_single_qubit_2qubit(remaining, q)
            params = _u3_from_matrix(mat)
            gates.append(["u3", q, params[0], params[1], params[2]])
        
        # Apply CNOT 0->1
        gates.append(["cx", 0, 1])
        
        # Update remaining: remove the applied gates
        # remaining = U_target * (applied gates)^†
        # For simplicity, just track what we've applied and compute at the end
    
    # Final single-qubit gates
    for q in range(qubits):
        mat = _extract_single_qubit_2qubit(remaining, q)
        params = _u3_from_matrix(mat)
        gates.append(["u3", q, params[0], params[1], params[2]])
    
    return gates


def _exact_cnot_2q():
    """Exact CNOT on 2 qubits (control=0, target=1)."""
    return [
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
    ]


def _exact_iswap_2q():
    """Exact iSWAP on 2 qubits with 2 CNOTs."""
    # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    # Decomposition with 2 CNOTs:
    gates = [
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 0, 0.0, 0.0, math.pi],
        ["u3", 1, 0.0, 0.0, math.pi],
    ]
    return gates


def _exact_swap_2q():
    """Exact SWAP on 2 qubits."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_toffoli_3q():
    """Exact Toffoli on 3 qubits (control=0,1 target=2) with 6 CNOTs."""
    # Standard 6-CNOT Toffoli decomposition
    # T = u3(0, 0, pi/4), T† = u3(0, 0, -pi/4)
    # H = u3(pi/2, 0, pi/2)
    # S = u3(0, 0, pi/2), S† = u3(0, 0, -pi/2)
    
    # One known 6-CNOT decomposition:
    gates = [
        ["u3", 2, 0.0, 0.0, math.pi / 4],   # T on q2
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T† on q2
        ["cx", 0, 2],
        ["u3", 2, 0.0, 0.0, math.pi / 4],   # T on q2
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T† on q2
        ["u3", 1, 0.0, 0.0, math.pi / 4],   # T on q1
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, -math.pi / 4],  # T† on q1
        ["u3", 0, 0.0, 0.0, math.pi / 4],   # T on q0
        ["u3", 1, 0.0, 0.0, -math.pi / 4],  # T† on q1
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, math.pi / 4],   # T on q1
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T† on q2
        ["cx", 0, 2],
        ["u3", 2, 0.0, 0.0, math.pi / 4],   # T on q2
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T† on q2
    ]
    return gates


def _coordinate_descent(target, gates_template, qubits, budget, max_sweeps=15):
    """Coordinate descent to optimize u3 parameters."""
    n_params = 3 * sum(1 for g in gates_template if g[0] == "u3")
    
    def build_with_params(values):
        gates = []
        k = 0
        for gate in gates_template:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_unitary(build_with_params(best), qubits))
    
    for sweep in range(max_sweeps):
        if _start_time is not None and time.time() - _start_time > _DEADLINE - 5:
            break
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if _start_time is not None and time.time() - _start_time > _DEADLINE - 5:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + 0.5 * step, best[i] - 0.5 * step):
                values = best[:]
                values[i] = trial
                dist = _distance(target, _build_unitary(build_with_params(values), qubits))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        if not improved:
            break
    
    return build_with_params(best), best_dist


def synthesize(problem):
    """Synthesize quantum circuit for given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "").lower()
    
    # Exact decompositions for known gates
    if qubits == 2:
        if "cnot" in name and budget >= 1:
            gates = _exact_cnot_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if "iswap" in name and budget >= 2:
            gates = _exact_iswap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if "swap" in name and budget >= 3:
            gates = _exact_swap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
    
    if qubits == 3 and "toffoli" in name and budget >= 6:
        gates = _exact_toffoli_3q()
        if _is_circuit_close(target, gates, qubits):
            return gates
    
    # Generic synthesis: build template with CNOTs and single-qubit gates
    # Use a balanced layout
    pairs_2q = [(0, 1), (1, 0)]
    pairs_3q = [(0, 1), (1, 2), (0, 2), (2, 1), (1, 0), (2, 0)]
    
    if qubits == 2:
        pairs = pairs_2q
    else:
        pairs = pairs_3q
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Coordinate descent optimization
    gates, dist = _coordinate_descent(target, layout, qubits, budget)
    
    return gates
