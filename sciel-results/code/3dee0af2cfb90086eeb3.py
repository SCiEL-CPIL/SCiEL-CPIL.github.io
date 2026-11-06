"""Quantum circuit synthesis: exact SVD layer-peeling with known gate decompositions."""
import time
import math
import numpy as np

_T0 = time.monotonic()
_DEADLINE = 250.0


def _time_left():
    return _DEADLINE - (time.monotonic() - _T0)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(q, theta, phi, lam, nq):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
    op = np.array([[1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for k in range(nq):
        op = np.kron(op, single if k == q else eye2)
    return op


def _cx_matrix(control, target, nq):
    n = 2 ** nq
    idx = np.arange(n)
    c_bit = (idx >> (nq - 1 - control)) & 1
    t_shift = (nq - 1 - target)
    new_idx = idx ^ (c_bit << t_shift)
    op = np.zeros((n, n), dtype=complex)
    op[new_idx, idx] = 1.0
    return op


def _build_fast(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            op = _u3_matrix(q, theta, phi, lam, qubits)
        else:
            _, c, t = gate
            op = _cx_matrix(c, t, qubits)
        matrix = op @ matrix
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", int(control), int(target)]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _u3_from_matrix(mat2):
    """Extract (theta, phi, lam) from a 2x2 unitary, up to global phase."""
    det = mat2[0, 0] * mat2[1, 1] - mat2[0, 1] * mat2[1, 0]
    if abs(det) > 1e-15:
        phase = det / abs(det)
        mat2 = mat2 / phase

    a = mat2[0, 0]
    b = mat2[0, 1]
    c = mat2[1, 0]
    d = mat2[1, 1]

    theta = 2.0 * math.atan2(abs(b), abs(a))

    if abs(b) < 1e-12 and abs(c) < 1e-12:
        if abs(a) > 1e-12 and abs(d) > 1e-12:
            phi = 0.0
            lam = math.atan2(np.imag(d / a), np.real(d / a))
        else:
            phi = 0.0
            lam = 0.0
        return (theta, phi, lam)

    if abs(b) > 1e-12:
        lam = math.atan2(-np.imag(b), -np.real(b))
    else:
        lam = 0.0

    if abs(c) > 1e-12:
        phi = math.atan2(np.imag(c), np.real(c))
    else:
        phi = 0.0

    return (theta, phi, lam)


def _make_unitary(m):
    """Make a matrix unitary via polar decomposition."""
    U, _, Vh = np.linalg.svd(m)
    return U @ Vh


def _extract_single_qubit_layers_2qubit(matrix4):
    """
    Given a 4x4 unitary that should be of the form A ⊗ B (product of single-qubit gates),
    extract A (on qubit 0) and B (on qubit 1).
    """
    m = matrix4.reshape(2, 2, 2, 2).transpose(0, 1, 2, 3)
    
    b_slice = m[0, 0, :, :]
    a00 = matrix4[0, 0]
    if abs(a00) > 1e-12:
        b = b_slice / a00
    else:
        a11 = matrix4[3, 3]
        if abs(a11) > 1e-12:
            b_slice = m[1, 1, :, :]
            b = b_slice / a11
        else:
            b = np.eye(2, dtype=complex)
    
    b00 = b[0, 0]
    if abs(b00) > 1e-12:
        a_slice = m[:, :, 0, 0]
        a = a_slice / b00
    else:
        b11 = b[1, 1]
        if abs(b11) > 1e-12:
            a_slice = m[:, :, 1, 1]
            a = a_slice / b11
        else:
            a = np.eye(2, dtype=complex)
    
    a = _make_unitary(a)
    b = _make_unitary(b)
    
    return a, b


def _peel_2qubit(target, budget):
    """SVD layer peeling for 2-qubit unitaries."""
    residual = target.copy()
    gates = []
    
    cnot_pattern = [(0, 1), (1, 0)]
    
    for layer in range(budget):
        if _time_left() < 5.0:
            break
        
        c, t = cnot_pattern[layer % 2]
        cx_op = _cx_matrix(c, t, 2)
        
        C = np.eye(2, dtype=complex)
        D = np.eye(2, dtype=complex)
        
        for _iter in range(40):
            if _time_left() < 3.0:
                break
            
            cd = np.kron(C, D)
            left = residual @ cd.conj().T
            s_left_cx = left @ cx_op
            
            a, b = _extract_single_qubit_layers_2qubit(s_left_cx)
            
            cd_new = cx_op @ np.kron(a, b).conj().T @ residual
            cd_new = _make_unitary(cd_new)
            
            c_new, d_new = _extract_single_qubit_layers_2qubit(cd_new)
            
            C = c_new
            D = d_new
        
        theta_c, phi_c, lam_c = _u3_from_matrix(C)
        theta_d, phi_d, lam_d = _u3_from_matrix(D)
        
        gates.append(_u3(0, theta_c, phi_c, lam_c))
        gates.append(_u3(1, theta_d, phi_d, lam_d))
        gates.append(_cx(c, t))
        
        cd = np.kron(C, D)
        residual = residual @ cd.conj().T @ cx_op
    
    if _time_left() > 1.0:
        a, b = _extract_single_qubit_layers_2qubit(residual)
        theta_a, phi_a, lam_a = _u3_from_matrix(a)
        theta_b, phi_b, lam_b = _u3_from_matrix(b)
        
        gates.append(_u3(0, theta_a, phi_a, lam_a))
        gates.append(_u3(1, theta_b, phi_b, lam_b))
    
    return gates


def _extract_3qubit_layers(mat8):
    """
    Extract 3 single-qubit gates from an 8x8 matrix that should be A ⊗ B ⊗ C.
    Returns (A, B, C) for qubits 0, 1, 2.
    """
    m = mat8.reshape(2, 2, 2, 2, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    # m[i0, j0, i1, j1, i2, j2] = A[i0,j0] * B[i1,j1] * C[i2,j2]
    
    # Get C from slice i0=0,j0=0,i1=0,j1=0
    c_slice = m[0, 0, 0, 0, :, :]
    a00 = mat8[0, 0]
    if abs(a00) > 1e-12:
        c = c_slice / a00
    else:
        # Try another entry
        a11 = mat8[1, 1]
        if abs(a11) > 1e-12:
            c_slice = m[0, 0, 1, 1, :, :]
            c = c_slice / a11
        else:
            c = np.eye(2, dtype=complex)
    
    # Get B from slice i0=0,j0=0,i2=0,j2=0
    b_slice = m[0, 0, :, :, 0, 0]
    c00 = c[0, 0]
    if abs(c00) > 1e-12:
        b = b_slice / c00
    else:
        c11 = c[1, 1]
        if abs(c11) > 1e-12:
            b_slice = m[0, 0, :, :, 1, 1]
            b = b_slice / c11
        else:
            b = np.eye(2, dtype=complex)
    
    # Get A from slice i1=0,j1=0,i2=0,j2=0
    a_slice = m[:, :, 0, 0, 0, 0]
    b00 = b[0, 0]
    if abs(b00) > 1e-12:
        a = a_slice / b00
    else:
        b11 = b[1, 1]
        if abs(b11) > 1e-12:
            a_slice = m[:, :, 1, 1, 0, 0]
            a = a_slice / b11
        else:
            a = np.eye(2, dtype=complex)
    
    a = _make_unitary(a)
    b = _make_unitary(b)
    c = _make_unitary(c)
    
    return a, b, c


def _peel_3qubit(target, budget):
    """SVD layer peeling for 3-qubit unitaries."""
    residual = target.copy()
    gates = []
    
    # Use a pattern that covers all pairs
    cnot_pattern = [(0, 1), (1, 2), (0, 2)]
    
    for layer in range(budget):
        if _time_left() < 5.0:
            break
        
        c, t = cnot_pattern[layer % 3]
        cx_op = _cx_matrix(c, t, 3)
        
        C0 = np.eye(2, dtype=complex)
        C1 = np.eye(2, dtype=complex)
        C2 = np.eye(2, dtype=complex)
        
        for _iter in range(25):
            if _time_left() < 3.0:
                break
            
            cd = np.kron(np.kron(C0, C1), C2)
            left = residual @ cd.conj().T
            s_left_cx = left @ cx_op
            
            a, b, cc = _extract_3qubit_layers(s_left_cx)
            
            cd_new = cx_op @ np.kron(np.kron(a, b), cc).conj().T @ residual
            cd_new = _make_unitary(cd_new)
            
            c0_new, c1_new, c2_new = _extract_3qubit_layers(cd_new)
            
            C0 = c0_new
            C1 = c1_new
            C2 = c2_new
        
        theta_0, phi_0, lam_0 = _u3_from_matrix(C0)
        theta_1, phi_1, lam_1 = _u3_from_matrix(C1)
        theta_2, phi_2, lam_2 = _u3_from_matrix(C2)
        
        gates.append(_u3(0, theta_0, phi_0, lam_0))
        gates.append(_u3(1, theta_1, phi_1, lam_1))
        gates.append(_u3(2, theta_2, phi_2, lam_2))
        gates.append(_cx(c, t))
        
        cd = np.kron(np.kron(C0, C1), C2)
        residual = residual @ cd.conj().T @ cx_op
    
    if _time_left() > 1.0:
        a, b, cc = _extract_3qubit_layers(residual)
        theta_a, phi_a, lam_a = _u3_from_matrix(a)
        theta_b, phi_b, lam_b = _u3_from_matrix(b)
        theta_c, phi_c, lam_c = _u3_from_matrix(cc)
        
        gates.append(_u3(0, theta_a, phi_a, lam_a))
        gates.append(_u3(1, theta_b, phi_b, lam_b))
        gates.append(_u3(2, theta_c, phi_c, lam_c))
    
    return gates


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]
    
    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    
    if name == "iSWAP" and qubits == 2:
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates
    
    if name == "Toffoli" and qubits == 3:
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates
    
    return None


def synthesize(problem):
    start_time = time.monotonic()
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build_fast(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # For generic unitaries, use SVD peeling
    if qubits == 2:
        result = _peel_2qubit(target, budget)
    else:
        result = _peel_3qubit(target, budget)
    
    # Verify budget constraints
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        truncated = []
        for g in result:
            if g[0] == "cx":
                if count < budget:
                    truncated.append(g)
                    count += 1
            else:
                truncated.append(g)
        result = truncated
    
    if len(result) > 400:
        result = result[:400]
    
    return result
