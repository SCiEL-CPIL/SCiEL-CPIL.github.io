"""Quantum circuit synthesis via exact algebraic decomposition with optimization fallback."""
import math
import time
import numpy as np
import random

_DEADLINE = time.time() + 260.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    """Build the unitary matrix from a gate list."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, c, t = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            U = U[idx ^ (bit << (qubits - 1 - t)), :]
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim."""
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / dim)


def _h(q):
    return ['u3', q, math.pi / 2, 0, math.pi]


def _s(q):
    return ['u3', q, 0, 0, math.pi / 2]


def _sdag(q):
    return ['u3', q, 0, 0, -math.pi / 2]


def _t(q):
    return ['u3', q, math.pi / 4, 0, 0]


def _tdag(q):
    return ['u3', q, -math.pi / 4, 0, 0]


def _cnot(c, t):
    return ['cx', c, t]


def _decomp_cnot_01():
    return [_cnot(0, 1)]


def _decomp_cnot_10():
    return [_cnot(1, 0)]


def _decomp_swap():
    return [_cnot(0, 1), _cnot(1, 0), _cnot(0, 1)]


def _decomp_iswap():
    return [_h(0), _h(1), _cnot(0, 1), _sdag(0), _s(1), _cnot(0, 1), _h(0), _h(1)]


def _decomp_toffoli(c1, c2, t):
    gates = []
    gates.append(_h(t))
    gates.append(_cnot(c1, t))
    gates.append(_tdag(t))
    gates.append(_t(c2))
    gates.append(_cnot(c2, t))
    gates.append(_t(t))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, t))
    gates.append(_t(c2))
    gates.append(_tdag(t))
    gates.append(_h(t))
    gates.append(_cnot(c1, c2))
    gates.append(_t(c1))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, c2))
    return gates


def _match_known(target, qubits):
    """Try to match the target against known gate decompositions."""
    if qubits == 2:
        candidates = [
            (_decomp_cnot_01(), 1),
            (_decomp_cnot_10(), 1),
            (_decomp_swap(), 3),
            (_decomp_iswap(), 2),
        ]
        for gates, cnot_count in candidates:
            U = _build(gates, qubits)
            if _distance(target, U) < 1e-10:
                return gates
    elif qubits == 3:
        for c1 in range(3):
            for c2 in range(3):
                if c2 == c1:
                    continue
                t = 3 - c1 - c2
                gates = _decomp_toffoli(c1, c2, t)
                U = _build(gates, qubits)
                if _distance(target, U) < 1e-10:
                    return gates
    return None


def _h_matrix():
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _u3_from_matrix(M, q):
    """Decompose a 2x2 unitary into u3(theta, phi, lam)."""
    a = M[0, 0]
    b_term = -M[0, 1]
    
    theta = 2 * math.acos(min(1.0, abs(a)))
    
    if abs(b_term) < 1e-10:
        lam = 0
        phi = 0
    else:
        lam = math.atan2(-b_term.imag, b_term.real)
        b = abs(b_term)
        if abs(b) < 1e-10:
            phi = 0
        else:
            phi_term = M[1, 0]
            phi = math.atan2(phi_term.imag, phi_term.real)
    
    return [['u3', q, theta, phi, lam]]


def _is_product_of_two(U):
    """Check if a 4x4 unitary is a product of two 2x2 unitaries."""
    if abs(U[0, 0]) < 1e-10:
        return False
    
    # Try to decompose U = A ⊗ B
    # Gauge: set A[0,0] = 1
    A00 = 1
    A01 = U[0, 2] / U[0, 0]
    A10 = U[2, 0] / U[0, 0]
    A11 = U[2, 2] / U[0, 0]
    A = np.array([[A00, A01], [A10, A11]], dtype=complex)
    
    B00 = U[0, 0]
    B01 = U[0, 1]
    B10 = U[1, 0]
    B11 = U[1, 1]
    B = np.array([[B00, B01], [B10, B11]], dtype=complex)
    
    # Check if A ⊗ B ≈ U
    recon = np.kron(A, B)
    if np.max(np.abs(recon - U)) < 1e-8:
        return True
    
    # Try other gauge: set B[0,0] = 1
    A00 = U[0, 0]
    A01 = U[0, 2]
    A10 = U[2, 0]
    A11 = U[2, 2]
    A = np.array([[A00, A01], [A10, A11]], dtype=complex)
    
    B00 = 1
    B01 = U[0, 1] / A00 if abs(A00) > 1e-10 else 0
    B10 = U[1, 0] / A00 if abs(A00) > 1e-10 else 0
    B11 = U[1, 1] / A00 if abs(A00) > 1e-10 else 0
    B = np.array([[B00, B01], [B10, B11]], dtype=complex)
    
    recon = np.kron(A, B)
    if np.max(np.abs(recon - U)) < 1e-8:
        return True
    
    return False


def _decompose_product(U):
    """Decompose U = A ⊗ B. Returns (A, B)."""
    if abs(U[0, 0]) < 1e-10:
        # Try gauge: B[0,0] = 1
        A00 = U[0, 0]
        A01 = U[0, 2]
        A10 = U[2, 0]
        A11 = U[2, 2]
        A = np.array([[A00, A01], [A10, A11]], dtype=complex)
        
        B00 = 1
        B01 = U[0, 1] / A00 if abs(A00) > 1e-10 else 0
        B10 = U[1, 0] / A00 if abs(A00) > 1e-10 else 0
        B11 = U[1, 1] / A00 if abs(A00) > 1e-10 else 0
        B = np.array([[B00, B01], [B10, B11]], dtype=complex)
        return A, B
    
    # Gauge: A[0,0] = 1
    A00 = 1
    A01 = U[0, 2] / U[0, 0]
    A10 = U[2, 0] / U[0, 0]
    A11 = U[2, 2] / U[0, 0]
    A = np.array([[A00, A01], [A10, A11]], dtype=complex)
    
    B00 = U[0, 0]
    B01 = U[0, 1]
    B10 = U[1, 0]
    B11 = U[1, 1]
    B = np.array([[B00, B01], [B10, B11]], dtype=complex)
    return A, B


def _cnot_matrix(c, t, qubits):
    """Build CNOT matrix for control c, target t on qubits."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    idx = np.arange(n)
    bit = (idx >> (qubits - 1 - c)) & 1
    perm = idx ^ (bit << (qubits - 1 - t))
    U = U[perm, :]
    return U


def _two_qubit_synthesis(target, budget):
    """
    Synthesize a 2-qubit unitary using algebraic decomposition with up to 3 CNOTs.
    """
    n = 4
    
    # Try 1 CNOT: U = L · CX · R where L, R are products of single-qubit gates
    for c, t in [(0, 1), (1, 0)]:
        CX = _cnot_matrix(c, t, 2)
        
        # Try R = I: L = U · CX
        L = target @ CX
        if _is_product_of_two(L):
            A, B = _decompose_product(L)
            gates = []
            gates.extend(_u3_from_matrix(A, 0))
            gates.extend(_u3_from_matrix(B, 1))
            gates.append(_cnot(c, t))
            return gates
        
        # Try various R values
        for phi in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]:
            R0 = np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)
            R1 = np.eye(2, dtype=complex)
            R = np.kron(R0, R1)
            L = target @ CX @ R.conj().T
            if _is_product_of_two(L):
                A, B = _decompose_product(L)
                gates = []
                gates.extend(_u3_from_matrix(A, 0))
                gates.extend(_u3_from_matrix(B, 1))
                gates.append(_cnot(c, t))
                gates.extend(_u3_from_matrix(R0, 0))
                return gates
    
    # Try 2 CNOTs: U = L1 · CX · D · CX · L2
    CX = _cnot_matrix(0, 1, 2)
    
    # Grid search over diagonal phases
    num_samples = 12
    angles = np.linspace(-math.pi, math.pi, num_samples, endpoint=False)
    
    best_dist = 1.0
    best_gates = []
    
    for a1 in angles:
        for a2 in angles:
            for a3 in angles:
                if time.time() > _DEADLINE - 30:
                    break
                for a4 in angles:
                    d = np.array([np.exp(1j*a1), np.exp(1j*a2), np.exp(1j*a3), np.exp(1j*a4)], dtype=complex)
                    D = np.diag(d)
                    
                    # U = L1 · CX · D · CX · L2
                    # Try L1 = I: L2 = CX · D^{-1} · CX · U
                    L1 = np.eye(4, dtype=complex)
                    L2 = CX @ np.diag(1/d) @ CX @ target
                    if _is_product_of_two(L2):
                        C, Dmat = _decompose_product(L2)
                        gates = []
                        gates.extend(_u3_from_matrix(C, 0))
                        gates.extend(_u3_from_matrix(Dmat, 1))
                        gates.append(_cnot(0, 1))
                        # Implement diag(d) as single-qubit gates
                        # diag(d) = (diag(d0,d1)) ⊗ (diag(d2/d0, d3/d1)) if d0 != 0
                        if abs(d[0]) > 1e-10:
                            d0_phase = math.atan2(d[0].imag, d[0].real)
                            d1_phase = math.atan2(d[1].imag, d[1].real)
                            d2_over_d0 = d[2] / d[0]
                            d3_over_d1 = d[3] / d[1]
                            d2_phase = math.atan2(d2_over_d0.imag, d2_over_d0.real)
                            d3_phase = math.atan2(d3_over_d1.imag, d3_over_d1.real)
                            # q0 gets phase d0, q1 gets phase d2/d0
                            gates.append(['u3', 0, 0, 0, d0_phase])
                            gates.append(['u3', 1, 0, 0, d2_phase])
                        else:
                            # Fallback: approximate with u3 gates
                            gates.append(['u3', 0, 0, 0, a1])
                            gates.append(['u3', 1, 0, 0, a3])
                        gates.append(_cnot(0, 1))
                        
                        U_check = _build(gates, 2)
                        dist = _distance(target, U_check)
                        if dist < best_dist:
                            best_dist = dist
                            best_gates = gates
                        if dist < 1e-10:
                            return gates
    
    if best_gates and best_dist < 1e-8:
        return best_gates
    
    # Fallback: 3 CNOTs with optimization
    if budget >= 3:
        return _optimize_two_qubit(target, 3)
    
    # Last resort: return identity
    return []


def _optimize_two_qubit(target, num_cnots):
    """Optimize a 2-qubit circuit with given number of CNOTs."""
    n = 4
    qubits = 2
    
    cnot_positions = [(0, 1)] * num_cnots
    num_params = (num_cnots + 1) * qubits * 3
    
    params = np.zeros(num_params)
    
    def objective(p):
        gates = []
        idx = 0
        for i in range(num_cnots):
            for q in range(qubits):
                theta = p[idx]
                phi = p[idx+1]
                lam = p[idx+2]
                gates.append(['u3', q, theta, phi, lam])
                idx += 3
            gates.append(_cnot(0, 1))
        for q in range(qubits):
            theta = p[idx]
            phi = p[idx+1]
            lam = p[idx+2]
            gates.append(['u3', q, theta, phi, lam])
            idx += 3
        
        U = _build(gates, qubits)
        return _distance(target, U)
    
    best_dist = objective(params)
    best_params = params.copy()
    
    step = 0.5
    no_improve = 0
    
    for sweep in range(200):
        if time.time() > _DEADLINE - 30:
            break
        
        improved = False
        for i in range(num_params):
            if time.time() > _DEADLINE - 30:
                break
            
            base = params[i]
            for sign in (1, -1):
                params[i] = base + sign * step
                dist = objective(params)
                if dist < best_dist:
                    best_dist = dist
                    best_params = params.copy()
                    improved = True
                else:
                    params[i] = base
        
        if improved:
            no_improve = 0
            step *= 0.8
        else:
            no_improve += 1
            if no_improve > 5:
                break
            step *= 0.5
    
    gates = []
    idx = 0
    for i in range(num_cnots):
        for q in range(qubits):
            theta = best_params[idx]
            phi = best_params[idx+1]
            lam = best_params[idx+2]
            gates.append(['u3', q, theta, phi, lam])
            idx += 3
        gates.append(_cnot(0, 1))
    for q in range(qubits):
        theta = best_params[idx]
        phi = best_params[idx+1]
        lam = best_params[idx+2]
        gates.append(['u3', q, theta, phi, lam])
        idx += 3
    
    return gates


def _three_qubit_synthesis(target, budget):
    """
    Synthesize a 3-qubit unitary using CNOTs and single-qubit gates.
    """
    n = 8
    qubits = 3
    
    pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    num_cnots = min(budget, len(pairs))
    cnot_positions = pairs[:num_cnots]
    
    num_params = (num_cnots + 1) * qubits * 3
    
    params = np.zeros(num_params)
    
    def objective(p):
        gates = []
        idx = 0
        for i in range(num_cnots):
            for q in range(qubits):
                theta = p[idx]
                phi = p[idx+1]
                lam = p[idx+2]
                gates.append(['u3', q, theta, phi, lam])
                idx += 3
            c, t = cnot_positions[i]
            gates.append(_cnot(c, t))
        for q in range(qubits):
            theta = p[idx]
            phi = p[idx+1]
            lam = p[idx+2]
            gates.append(['u3', q, theta, phi, lam])
            idx += 3
        
        U = _build(gates, qubits)
        return _distance(target, U)
    
    best_dist = objective(params)
    best_params = params.copy()
    
    step = 0.5
    no_improve = 0
    
    max_sweeps = 100
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
            break
        
        improved = False
        for i in range(num_params):
            if time.time() > _DEADLINE - 20:
                break
            
            base = params[i]
            for sign in (1, -1):
                params[i] = base + sign * step
                dist = objective(params)
                if dist < best_dist:
                    best_dist = dist
                    best_params = params.copy()
                    improved = True
                else:
                    params[i] = base
        
        if improved:
            no_improve = 0
            step *= 0.8
        else:
            no_improve += 1
            if no_improve > 3:
                break
            step *= 0.5
    
    gates = []
    idx = 0
    for i in range(num_cnots):
        for q in range(qubits):
            theta = best_params[idx]
            phi = best_params[idx+1]
            lam = best_params[idx+2]
            gates.append(['u3', q, theta, phi, lam])
            idx += 3
        c, t = cnot_positions[i]
        gates.append(_cnot(c, t))
    for q in range(qubits):
        theta = best_params[idx]
        phi = best_params[idx+1]
        lam = best_params[idx+2]
        gates.append(['u3', q, theta, phi, lam])
        idx += 3
    
    return gates


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try known gate matching first
    known = _match_known(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known
    
    # Generic synthesis
    if qubits == 2:
        return _two_qubit_synthesis(target, budget)
    else:
        return _three_qubit_synthesis(target, budget)
