"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _apply_u3(matrix, q, theta, phi, lam, qubits):
    """Apply u3 gate to a specific qubit in-place on the matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
    # u3 = [[c, -e1*s], [e2*s, e2*e1*c]]
    idx = np.arange(2 ** qubits)
    bit_pos = qubits - 1 - q
    bit = (idx >> bit_pos) & 1
    # For each pair of states differing only in qubit q
    for base in range(2 ** qubits):
        if base & (1 << bit_pos):
            continue
        i0 = base
        i1 = base | (1 << bit_pos)
        m00, m01 = matrix[i0], matrix[i1]
        matrix[i0] = c * m00 - e1 * s * m01
        matrix[i1] = e2 * s * m00 + e2 * e1 * c * m01


def _apply_cx(matrix, control, target, qubits):
    """Apply CNOT gate in-place on the matrix."""
    idx = np.arange(2 ** qubits)
    c_bit = (idx >> (qubits - 1 - control)) & 1
    t_bit = (idx >> (qubits - 1 - target)) & 1
    # Swap rows where control=1
    for i in range(2 ** qubits):
        if c_bit[i] == 1:
            j = i ^ (1 << (qubits - 1 - target))
            matrix[i], matrix[j] = matrix[j].copy(), matrix[i].copy()


def _build_fast(gates, qubits):
    """Build unitary using in-place operations for speed."""
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            _apply_u3(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            _apply_cx(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _nelder_mead(f, x0, max_iter=500, tol=1e-12, timeout=None):
    """Nelder-Mead optimizer with timeout support."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] = x0[i] + 0.1 * (abs(x0[i]) + 1.0)
        simplex.append(point)
    
    fvals = [f(p) for p in simplex]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 2.0
    
    for it in range(max_iter):
        if timeout is not None and time.time() > timeout:
            break
        
        # Sort simplex
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    order = sorted(range(n + 1), key=lambda i: fvals[i])
    return simplex[order[0]], fvals[order[0]]


def _known_decomposition(name, qubits, budget):
    """Return known exact decompositions for standard gates."""
    gates = []
    
    if name == "CNOT" and qubits == 2:
        # CNOT is just a CNOT
        gates = [["cx", 0, 1]]
        return gates
    
    if name == "iSWAP" and qubits == 2:
        # iSWAP: 0->0, 1->i*2, 2->i*1, 3->3
        # Decomposition: H on both, CNOT, H on both, then phase corrections
        # iSWAP = (I⊗H) CNOT (I⊗H) * diag(1,i,i,1) up to global phase
        # Actually: iSWAP = exp(i*pi/4 * (X⊗X + Y⊗Y)/2)
        # Using: H1 CNOT H1 then phase gates
        # Standard decomposition:
        gates = [
            ["u3", 0, math.pi/2, 0, math.pi/2],  # S on q0
            ["u3", 1, math.pi/2, 0, math.pi/2],  # S on q1
            ["u3", 0, math.pi/2, 0, 0],          # H-like
            ["u3", 1, math.pi/2, 0, 0],          # H-like
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, 0, 0],
            ["u3", 1, math.pi/2, 0, 0],
            ["u3", 0, math.pi/2, 0, -math.pi/2], # S† on q0
            ["u3", 1, math.pi/2, 0, -math.pi/2], # S† on q1
        ]
        return gates
    
    if name == "SWAP" and qubits == 2:
        # SWAP = 3 CNOTs
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        return gates
    
    if name == "Toffoli" and qubits == 3:
        # Toffoli: |110> -> |111>, |111> -> |110>
        # Standard decomposition uses 6 CNOTs
        # From Nielsen & Chuang
        gates = [
            ["u3", 2, math.pi/2, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, math.pi/4, 0, 0],
            ["cx", 0, 2],
            ["u3", 2, -math.pi/4, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, math.pi/4, 0, 0],
            ["cx", 0, 2],
            ["u3", 2, -math.pi/4, 0, 0],
            ["u3", 1, math.pi/2, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, math.pi/2, 0, 0],
        ]
        # This uses 6 CNOTs, within budget of 6
        return gates
    
    return None


def synthesize(problem):
    """Synthesize quantum circuit with known decompositions and optimization for generic gates."""
    name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known decomposition first
    known = _known_decomposition(name, qubits, budget)
    if known is not None:
        # Verify it's correct
        if len(known) <= 400:
            cnot_count = sum(1 for g in known if g[0] == "cx")
            if cnot_count <= budget:
                dist = _distance(target, _build_fast(known, qubits))
                if dist < 1e-10:
                    return known
    
    # For generic unitaries, use optimization
    # Build a layout with the given CNOT budget
    if qubits == 2:
        pairs = [(0, 1)]
    else:
        pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    
    # Create layout: layers of CNOTs with u3 gates around them
    layout = []
    for layer in range(budget):
        # Add u3 gates for all qubits before CNOT
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        # Add CNOT
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    # Final u3 gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Count parameters
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Initial values
    best = [0.0] * n_params
    
    def circuit_from_values(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(values):
        gates = circuit_from_values(values)
        if len(gates) > 400:
            return 1.0
        cnot_count = sum(1 for g in gates if g[0] == "cx")
        if cnot_count > budget:
            return 1.0
        matrix = _build_fast(gates, qubits)
        return _distance(target, matrix)
    
    # Nelder-Mead optimization
    best_values, best_dist = _nelder_mead(objective, best, max_iter=300, tol=1e-14, timeout=_DEADLINE)
    
    return circuit_from_values(best_values)
# EVOLVE-BLOCK-END
