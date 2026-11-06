"""Rewritten program for quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build full operator efficiently
            op = np.eye(n, dtype=complex)
            # Apply single-qubit gate on qubit q (0-indexed, 0 is MSB)
            for i in range(n):
                bit = (i >> (qubits - 1 - q)) & 1
                if bit == 0:
                    op[i, i] = single[0, 0]
                    op[i, i ^ (1 << (qubits - 1 - q))] = single[0, 1]
                else:
                    op[i, i] = single[1, 1]
                    op[i, i ^ (1 << (qubits - 1 - q))] = single[1, 0]
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            perm = np.arange(n)
            for i in range(n):
                cbit = (i >> (qubits - 1 - control)) & 1
                if cbit:
                    perm[i] = i ^ (1 << (qubits - 1 - target))
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass, more efficient."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e0 = np.exp(1j * lam)
            e1 = np.exp(1j * phi)
            e2 = np.exp(1j * (phi + lam))
            single = np.array([[c, -e0 * s], [e1 * s, e2 * c]], dtype=complex)
            # Apply to matrix: for each basis state, update
            new_matrix = np.zeros_like(matrix)
            shift = 1 << (qubits - 1 - q)
            for i in range(n):
                bit = (i >> (qubits - 1 - q)) & 1
                if bit == 0:
                    new_matrix[i, :] = single[0, 0] * matrix[i, :] + single[0, 1] * matrix[i | shift, :]
                    new_matrix[i | shift, :] = single[1, 0] * matrix[i, :] + single[1, 1] * matrix[i | shift, :]
                # else: i already has bit set, handled above
            matrix = new_matrix
        else:
            _, control, target_q = gate
            perm = np.arange(n)
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _make_layout(qubits, budget):
    """Create a layout with CNOTs and u3 placeholders."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout.append(("u3", 0))
        layout.append(("u3", 1))
        if qubits == 3:
            layout.append(("u3", 2))
        layout.append(("cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
    for q in range(qubits):
        layout.append(("u3", q))
    return layout


def _circuit_from_params(layout, params, qubits):
    """Build gate list from layout and parameter values."""
    gates = []
    k = 0
    for item in layout:
        if item[0] == "u3":
            q = item[1]
            gates.append(["u3", q, params[k], params[k + 1], params[k + 2]])
            k += 3
        else:
            gates.append(["cx", item[1], item[2]])
    return gates


def _nelder_mead(f, x0, max_iter=300, tol=1e-8, deadline=None):
    """Nelder-Mead optimizer with deadline guard."""
    n = len(x0)
    alpha, gamma, rho = 1.0, 2.0, 0.5
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] = x0[i] + 0.5 * (abs(x0[i]) + 1.0)
        simplex.append(point)
    
    fvals = [f(p) for p in simplex]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid excluding worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        worst = simplex[-1]
        refl = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        frefl = f(refl)
        
        if fvals[0] <= frefl < fvals[-2]:
            simplex[-1] = refl
            fvals[-1] = frefl
            continue
        
        # Expansion
        if frefl < fvals[0]:
            exp = [centroid[j] + gamma * (refl[j] - centroid[j]) for j in range(n)]
            fexp = f(exp)
            if fexp < frefl:
                simplex[-1] = exp
                fvals[-1] = fexp
            else:
                simplex[-1] = refl
                fvals[-1] = frefl
            continue
        
        # Contraction
        if frefl < fvals[-1]:
            contr = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            fcontr = f(contr)
            if fcontr < fvals[-1]:
                simplex[-1] = contr
                fvals[-1] = fcontr
            else:
                # Shrink
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [best[j] + rho * (simplex[i][j] - best[j]) for j in range(n)]
                    fvals[i] = f(simplex[i])
        else:
            # Shrink
            best = simplex[0]
            for i in range(1, n + 1):
                simplex[i] = [best[j] + rho * (simplex[i][j] - best[j]) for j in range(n)]
                fvals[i] = f(simplex[i])
    
    order = sorted(range(n + 1), key=lambda i: fvals[i])
    return simplex[order[0]], fvals[order[0]]


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    start_time = time.time()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Create layout
    layout = _make_layout(qubits, budget)
    n_params = 3 * sum(1 for item in layout if item[0] == "u3")
    
    # Initial parameters: small random values
    import random
    rng = random.Random(42 + qubits + budget)
    x0 = [rng.uniform(-0.1, 0.1) for _ in range(n_params)]
    
    # Objective function
    def objective(params):
        gates = _circuit_from_params(layout, params, qubits)
        return _build_fast(gates, qubits, target)
    
    # Run Nelder-Mead
    best_params, best_val = _nelder_mead(objective, x0, max_iter=200, tol=1e-9, deadline=_DEADLINE)
    
    # Try a few more restarts if time allows
    for restart in range(5):
        if time.time() > _DEADLINE - 30:
            break
        x0 = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        p, v = _nelder_mead(objective, x0, max_iter=100, tol=1e-9, deadline=_DEADLINE)
        if v < best_val:
            best_params, best_val = p, v
    
    gates = _circuit_from_params(layout, best_params, qubits)
    return gates
# EVOLVE-BLOCK-END
