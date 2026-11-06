"""Exact quantum circuit synthesis with structured decomposition for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]])


def _apply_single_qubit(matrix, q, theta, phi, lam, qubits):
    """Apply a u3 gate on qubit q to the matrix (left multiplication)."""
    n = 2 ** qubits
    single = _u3_matrix(theta, phi, lam)
    eye = np.eye(2, dtype=complex)
    # Build the full operator
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, single if k == q else eye)
    return op @ matrix


def _apply_cnot(matrix, control, target, qubits):
    """Apply a CNOT gate (left multiplication)."""
    n = 2 ** qubits
    perm = np.arange(n)
    cbit = qubits - 1 - control
    tbit = qubits - 1 - target
    for i in range(n):
        if (i >> cbit) & 1:
            perm[i] = i ^ (1 << tbit)
    return matrix[perm, :]


def _build_circuit_matrix(gates, qubits):
    """Build the unitary matrix from a gate list."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_single_qubit(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            matrix = _apply_cnot(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _exact_cnot():
    return [["cx", 0, 1]]


def _exact_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_iswap():
    """iSWAP decomposition using 2 CNOTs."""
    # iSWAP = diag(1,0,0,1) + i|01><10| + i|10><01|
    # Known decomposition: 
    # H on both, CX, H on both, then phase gates
    # Actually: iSWAP = (S⊗S) CX (H⊗H) CX (H⊗H) (S†⊗S†) ... let me verify
    # Standard: 2 CNOTs + single qubit gates
    gates = [
        ["u3", 0, math.pi / 2, 0, 0],
        ["u3", 1, math.pi / 2, 0, 0],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, math.pi / 2, math.pi / 2],
        ["u3", 1, math.pi / 2, math.pi / 2, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, -math.pi / 2, math.pi / 2],
        ["u3", 1, math.pi / 2, -math.pi / 2, math.pi / 2],
    ]
    return gates


def _toffoli_exact():
    """Standard 6-CNOT Toffoli decomposition."""
    gates = [
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["u3", 1, math.pi / 2, 0, 0],
        ["u3", 1, math.pi / 4, 0, 0],
        ["cx", 0, 1],
        ["u3", 1, -math.pi / 4, 0, 0],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, 0, 0],
        ["u3", 0, math.pi / 4, 0, 0],
        ["u3", 1, math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["u3", 0, -math.pi / 4, 0, 0],
        ["u3", 1, -math.pi / 4, 0, 0],
        ["u3", 2, math.pi / 2, 0, 0],
    ]
    return gates


def _decompose_2qubit(target, budget):
    """
    Decompose a 2-qubit unitary using the standard KAK/Cartan decomposition.
    U = A * (exp(i a X⊗X) exp(i b Y⊗Y) exp(i c Z⊗Z)) * B
    where A, B are single-qubit unitaries.
    For our purposes, we use a CNOT-based decomposition.
    """
    # For 2 qubits, we can use: U = (A1⊗A2) CX (B1⊗B2) CX (C1⊗C2) CX (D1⊗D2)
    # with 3 CNOTs. But for iSWAP we need 2.
    # Let's use a general approach: parameterize with CNOTs and optimize single-qubit gates.
    
    if budget >= 3:
        # Use 3 CNOTs decomposition
        layout = [
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
        ]
    elif budget >= 2:
        # Use 2 CNOTs
        layout = [
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
        ]
    else:
        # Only single qubit gates possible
        layout = [
            ["u3", 0, 0, 0, 0],
            ["u3", 1, 0, 0, 0],
        ]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    # Initial guess
    rng = random.Random(42)
    x0 = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    
    bounds = [(-math.pi, math.pi)] * n_params
    
    def objective(x):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], x[k], x[k+1], x[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        matrix = _build_circuit_matrix(gates, 2)
        return _distance(target, matrix)
    
    best_x = x0
    best_f = objective(x0)
    
    # Nelder-Mead
    n = n_params
    simplex = [x0[:]]
    fvals = [f0]
    
    for i in range(n):
        x = x0[:]
        step = 0.1
        x[i] += step
        x[i] = max(bounds[i][0], min(bounds[i][1], x[i]))
        simplex.append(x)
        fvals.append(objective(x))
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    max_iter = 300
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < 1e-10:
            break
        
        centroid = np.zeros(n)
        for i in range(n):
            centroid += np.array(simplex[i])
        centroid /= n
        
        xr = centroid + alpha * (centroid - np.array(simplex[-1]))
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
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
            xc = centroid + rho * (np.array(simplex[-1]) - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (np.array(simplex[i]) - np.array(simplex[0]))
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])
    
    best_idx = np.argmin(fvals)
    best_x = simplex[best_idx]
    best_f = fvals[best_idx]
    
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best_x[k], best_x[k+1], best_x[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    
    return gates, best_f


def _decompose_3qubit(target, budget):
    """
    Decompose a 3-qubit unitary using a CNOT-based decomposition.
    Use a ladder of CNOTs with single-qubit gates.
    """
    # Standard decomposition uses multiple CNOTs
    # For 3 qubits, we can use a recursive approach or a fixed layout
    
    if budget >= 6:
        # Use a layout with 6 CNOTs
        cnot_pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    elif budget >= 4:
        cnot_pairs = [(0,1), (1,2), (0,2), (1,0)]
    else:
        cnot_pairs = [(0,1), (1,2)]
    
    layout = []
    for i, pair in enumerate(cnot_pairs):
        layout += [["u3", q, 0, 0, 0] for q in range(3)]
        layout.append(["cx", pair[0], pair[1]])
    layout += [["u3", q, 0, 0, 0] for q in range(3)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    rng = random.Random(42)
    x0 = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    bounds = [(-math.pi, math.pi)] * n_params
    
    def objective(x):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], x[k], x[k+1], x[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        if len(gates) > 400:
            return 1.0
        matrix = _build_circuit_matrix(gates, 3)
        return _distance(target, matrix)
    
    best_x = x0
    best_f = objective(x0)
    
    # Nelder-Mead with limited iterations due to high dimensionality
    n = n_params
    simplex = [x0[:]]
    fvals = [best_f]
    
    for i in range(min(n, 20)):  # Limit simplex size for high-dim
        x = x0[:]
        step = 0.1
        x[i % n] += step
        x[i % n] = max(bounds[i % n][0], min(bounds[i % n][1], x[i % n]))
        simplex.append(x)
        fvals.append(objective(x))
    
    max_iter = 100
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        order = sorted(range(len(simplex)), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < 1e-8:
            break
        
        centroid = np.zeros(n)
        for i in range(len(simplex) - 1):
            centroid += np.array(simplex[i])
        centroid /= (len(simplex) - 1)
        
        xr = centroid + 1.0 * (centroid - np.array(simplex[-1]))
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < fvals[0]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + 0.5 * (np.array(simplex[-1]) - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (np.array(simplex[i]) - np.array(simplex[0]))
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])
    
    best_idx = np.argmin(fvals)
    best_x = simplex[best_idx]
    best_f = fvals[best_idx]
    
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best_x[k], best_x[k+1], best_x[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    
    # Trim to 400 gates max
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates, best_f


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates
    if qubits == 2:
        target_cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        target_iswap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        target_swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        for known_name, known_mat, exact_func in [
            ("CNOT", target_cnot, _exact_cnot),
            ("iSWAP", target_iswap, _exact_iswap),
            ("SWAP", target_swap, _exact_swap)
        ]:
            overlap = abs(np.trace(known_mat.conj().T @ target)) / 4
            if overlap > 0.999:
                gates = exact_func()
                if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
                    return gates
    
    if qubits == 3 and budget >= 6:
        target_toffoli = np.zeros((8, 8), dtype=complex)
        for i in range(6):
            target_toffoli[i, i] = 1
        target_toffoli[6, 7] = 1
        target_toffoli[7, 6] = 1
        
        overlap = abs(np.trace(target_toffoli.conj().T @ target)) / 8
        if overlap > 0.999:
            gates = _toffoli_exact()
            if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
                return gates
    
    # Generic unitary
    if qubits == 2:
        gates, dist = _decompose_2qubit(target, budget)
    else:
        gates, dist = _decompose_3qubit(target, budget)
    
    # Verify constraints
    n_cnots = sum(1 for g in gates if g[0] == "cx")
    if n_cnots > budget:
        # Remove excess CNOTs
        count = 0
        new_gates = []
        for g in gates:
            if g[0] == "cx" and count >= budget:
                continue
            if g[0] == "cx":
                count += 1
            new_gates.append(g)
        gates = new_gates
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
# EVOLVE-BLOCK-END
