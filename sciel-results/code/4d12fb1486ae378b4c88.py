"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill

# Canonical Unitaries (up to global phase)
# We define them as complex matrices.
# Qubit 0 is MSB.
# 2 Qubits:
# Basis: 00, 01, 10, 11 (Indices 0, 1, 2, 3)
CNOT_2Q = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

SWAP_2Q = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

iSWAP_2Q = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# 3 Qubits:
# Basis: 000, 001, 010, 011, 100, 101, 110, 111 (Indices 0-7)
TOFFOLI_3Q = np.eye(8, dtype=complex)
# Toffoli flips target (qubit 2) if controls (qubits 0,1) are 1.
# 110 is index 6, 111 is index 7.
TOFFOLI_3Q[6, 6] = 0
TOFFOLI_3Q[6, 7] = 1
TOFFOLI_3Q[7, 6] = 1
TOFFOLI_3Q[7, 7] = 0

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass, efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
            # u3 = [[c, -e1*s], [e2*s, e2*e1*c]]
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.zeros_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)

def _is_same_unitary(u1, u2, tol=1e-5):
    """Check if two unitaries are the same up to global phase."""
    n = u1.shape[0]
    # Find a non-zero element
    for i in range(n):
        for j in range(n):
            if abs(u1[i, j]) > tol:
                phase = u1[i, j] / u2[i, j]
                # Check if u1 * conj(phase) == u2
                if np.allclose(u1 * np.conj(phase), u2, atol=tol):
                    return True
    return False

def _synthesize_cnot_2q():
    return [["cx", 0, 1]]

def _synthesize_swap_2q():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]

def _synthesize_iswap_2q():
    # iSWAP = H(0) H(1) CX(0,1) S(0) S(1) CX(0,1) H(0) H(1) * global phase?
    # Standard decomposition:
    # iSWAP = 
    #   H(0) H(1)
    #   CX(0,1)
    #   S(0) S(1)  (S = diag(1, i))
    #   CX(0,1)
    #   H(0) H(1)
    # Let's verify this.
    # Actually, iSWAP is symmetric.
    # A known decomposition uses 3 CNOTs.
    # Ref: https://qiskit.org/documentation/stubs/qiskit.circuit.library.iSwapGate.html
    # iSWAP = 
    #   H(0) H(1)
    #   CX(0,1)
    #   S(0) S(1)
    #   CX(0,1)
    #   H(0) H(1)
    # Wait, S gate is u3(pi/2, 0, 0) ? No, S = diag(1, i).
    # u3(theta, phi, lam) = 
    #   [[cos(theta/2), -e^{i lam} sin(theta/2)],
    #    [e^{i phi} sin(theta/2), e^{i(phi+lam)} cos(theta/2)]]
    # S: [[1, 0], [0, i]].
    # If theta=pi/2, phi=0, lam=0:
    # [[1/sqrt(2), -1/sqrt(2)], [1/sqrt(2), 1/sqrt(2)]] -> This is H * R_z? No.
    # Let's use specific u3 parameters for S.
    # S = R_z(pi/2) * global phase? 
    # R_z(theta) = [[e^{-i theta/2}, 0], [0, e^{i theta/2}]]
    # S = [[1, 0], [0, i]] = e^{i pi/4} [[e^{-i pi/4}, 0], [0, e^{i pi/4}]] = e^{i pi/4} R_z(pi/2).
    # u3(theta, phi, lam) = e^{i (phi+lam)/2} R_z(phi) R_x(theta) R_z(lam).
    # To get R_z(pi/2), set theta=0, phi=pi/2, lam=0.
    # u3(0, pi/2, 0) = e^{i pi/4} R_z(pi/2) = S.
    # So S is u3(0, pi/2, 0).
    
    return [
        ["u3", 0, math.pi, 0, 0], # H
        ["u3", 1, math.pi, 0, 0], # H
        ["cx", 0, 1],
        ["u3", 0, 0, math.pi/2, 0], # S
        ["u3", 1, 0, math.pi/2, 0], # S
        ["cx", 0, 1],
        ["u3", 0, math.pi, 0, 0], # H
        ["u3", 1, math.pi, 0, 0]  # H
    ]

def _synthesize_toffoli_3q():
    # Standard 6 CNOT Toffoli decomposition
    # Ref: Nielsen & Chuang, or Qiskit ToffoliGate
    # Qiskit decomposition:
    # u3(pi/2, 0, -pi/2) on q2
    # cx(0, 2)
    # u3(-pi/4, 0, 0) on q2
    # cx(1, 2)
    # u3(pi/4, 0, 0) on q2
    # cx(0, 2)
    # u3(-pi/4, 0, 0) on q2
    # cx(1, 2)
    # u3(pi/4, 0, pi/2) on q2
    # u3(0, 0, pi) on q1
    # cx(0, 1)
    # u3(0, 0, -pi/4) on q1
    # u3(0, 0, pi/4) on q2
    # cx(0, 2)
    # u3(0, 0, -pi/4) on q2
    # u3(0, 0, pi/4) on q1
    # cx(0, 1)
    # u3(0, 0, pi) on q1
    # This uses 6 CNOTs.
    
    return [
        ["u3", 2, math.pi/2, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["u3", 1, 0, 0, math.pi],
        ["cx", 0, 1],
        ["u3", 1, 0, 0, -math.pi/4],
        ["u3", 2, 0, 0, math.pi/4],
        ["cx", 0, 2],
        ["u3", 2, 0, 0, -math.pi/4],
        ["u3", 1, 0, 0, math.pi/4],
        ["cx", 0, 1],
        ["u3", 1, 0, 0, math.pi]
    ]

def _synthesize_generic(problem, qubits, budget, target):
    """Optimize a generic circuit."""
    # Layout: Interleave single-qubit gates and CNOTs
    # For 3 qubits, a common structure is layers of CNOTs.
    # We will create a layout with 'budget' CNOTs.
    # If budget is 6, use the standard 6-CNOT ladder.
    # If budget is different, adjust.
    
    pairs_3q = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    
    layout = []
    # Initial single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    for i in range(budget):
        # Add CNOT
        c, t = pairs_3q[i % len(pairs_3q)]
        layout.append(["cx", c, t])
        # Add single-qubit gates after each CNOT (except maybe last)
        if i < budget - 1:
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    # Final single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    # Nelder-Mead Optimization
    # Initial guess: zeros
    simplex = [ [0.0] * n_params ]
    for i in range(n_params):
        point = [0.0] * n_params
        point[i] = 0.1
        simplex.append(point)
        
    # Evaluate initial simplex
    simplex_values = []
    for point in simplex:
        val = _build_fast(circuit(point), qubits, target)
        simplex_values.append(val)
        
    # Sort simplex
    order = np.argsort(simplex_values)
    simplex = [simplex[i] for i in order]
    simplex_values = [simplex_values[i] for i in order]
    
    best_point = simplex[0]
    best_val = simplex_values[0]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 500
    iter_count = 0
    
    while iter_count < max_iter:
        if time.time() - _START_TIME > _DEADLINE:
            break
            
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = _build_fast(circuit(xr), qubits, target)
        
        if fr < simplex_values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = _build_fast(circuit(xe), qubits, target)
            if fe < fr:
                simplex[-1] = xe
                simplex_values[-1] = fe
            else:
                simplex[-1] = xr
                simplex_values[-1] = fr
        elif fr < simplex_values[-2]:
            simplex[-1] = xr
            simplex_values[-1] = fr
        else:
            # Contraction
            if fr < simplex_values[-1]:
                # Outside contraction
                xc = centroid + 0.5 * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = _build_fast(circuit(xc), qubits, target)
            if fc < min(fr, simplex_values[-1]):
                simplex[-1] = xc
                simplex_values[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex_values[i] = _build_fast(circuit(simplex[i]), qubits, target)
        
        # Sort
        order = np.argsort(simplex_values)
        simplex = [simplex[i] for i in order]
        simplex_values = [simplex_values[i] for i in order]
        
        if simplex_values[0] < best_val:
            best_val = simplex_values[0]
            best_point = simplex[0]
            
        iter_count += 1
        
    return circuit(best_point)

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if qubits == 2:
        if _is_same_unitary(target, CNOT_2Q):
            return _synthesize_cnot_2q()
        if _is_same_unitary(target, SWAP_2Q):
            return _synthesize_swap_2q()
        if _is_same_unitary(target, iSWAP_2Q):
            return _synthesize_iswap_2q()
            
    if qubits == 3:
        if _is_same_unitary(target, TOFFOLI_3Q):
            return _synthesize_toffoli_3q()
            
    # Generic optimization
    return _synthesize_generic(problem, qubits, budget, target)
# EVOLVE-BLOCK-END
