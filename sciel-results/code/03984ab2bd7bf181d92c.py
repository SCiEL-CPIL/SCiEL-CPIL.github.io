"""Quantum circuit synthesis with exact known-gate circuits and fast optimization fallback."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # Safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            U = operator @ U
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            U = U[index ^ (bit << (qubits - 1 - target)), :]
    return U


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _check_close(target, gates, qubits, tol=1e-10):
    if _time_left() < 0.5:
        return False
    U = _build_fast(gates, qubits)
    return _distance(target, U) < tol


def _known_circuit_cnot_2q():
    return [["cx", 0, 1]]


def _known_circuit_swap_2q():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _known_circuit_iswap_2q():
    # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    # Decomposition: (H⊗H) · CNOT(0,1) · (S†⊗S) · CNOT(0,1) · (H⊗H)
    # But let's use a verified 2-CNOT decomposition:
    # iSWAP can be done with:
    # u3(0, pi/2, 0, pi/2), u3(1, pi/2, 0, pi/2), cx(0,1),
    # u3(0, pi/2, pi, -pi/2), u3(1, pi/2, pi, -pi/2), cx(0,1)
    # Let me verify this gives iSWAP...
    # Actually, let me use the standard decomposition:
    # iSWAP = (1/√2) [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # A known 2-CNOT circuit:
    gates = [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, math.pi, -math.pi/2],
        ["u3", 1, math.pi/2, math.pi, -math.pi/2],
        ["cx", 0, 1],
    ]
    return gates


def _known_circuit_toffoli_3q():
    # Toffoli (CCX) with controls 0,1 and target 2
    # Known 7-CNOT decomposition:
    gates = [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, math.pi, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, -math.pi/4, 0, math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, -math.pi/4, 0, math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, math.pi, -math.pi/2],
    ]
    return gates


def _fredkin_circuit():
    # Fredkin (CSWAP) with control 0, data 1,2
    # Known decomposition with 3 CNOTs
    gates = [
        ["cx", 1, 2],
        ["cx", 0, 2],
        ["cx", 1, 2],
    ]
    return gates


def _ccz_circuit():
    # CCZ with controls 0,1 and target 2
    gates = [
        ["u3", 2, 0, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, 0, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -math.pi/2],
    ]
    return gates


def _optimize_generic(target, qubits, budget):
    """
    Optimize a parameterized circuit to match the target unitary.
    Uses a template with alternating u3 gates and CNOTs.
    """
    if _time_left() < 1.0:
        return []
    
    n = 2 ** qubits
    target = np.asarray(target, dtype=complex)
    
    # Determine number of u3 gates per CNOT layer
    # For 2 qubits: 2 u3 before each CNOT, 2 after
    # For 3 qubits: 3 u3 before each CNOT, 3 after
    
    # Build a template: for each CNOT, add u3 gates on all qubits before and after
    # Total CNOTs = budget
    # Total u3 = 2 * budget * qubits (before and after each CNOT)
    
    # But we need to be careful about the total gate count (max 400)
    max_gates = 400
    # Each CNOT uses 1 gate slot, each u3 uses 1 gate slot
    # So: budget (CNOTs) + 2*budget*qubits (u3s) <= 400
    # For qubits=2: budget + 4*budget = 5*budget <= 400 -> budget <= 80
    # For qubits=3: budget + 6*budget = 7*budget <= 400 -> budget <= 57
    
    # Let's use a simpler template:
    # Layer structure: [u3s, CNOT, u3s, CNOT, u3s, ...]
    # Alternate CNOT patterns for 3 qubits
    
    if qubits == 2:
        # Template: u3(0), u3(1), cx(0,1), u3(0), u3(1), cx(1,0), ...
        # Use budget CNOTs, alternating between cx(0,1) and cx(1,0)
        template = []
        for i in range(budget):
            # Add u3 gates on both qubits
            template.append(["u3", 0, 0, 0, 0])
            template.append(["u3", 1, 0, 0, 0])
            # Add CNOT
            if i % 2 == 0:
                template.append(["cx", 0, 1])
            else:
                template.append(["cx", 1, 0])
        # Add final u3s
        template.append(["u3", 0, 0, 0, 0])
        template.append(["u3", 1, 0, 0, 0])
        
        n_params = 2 * (budget + 1) * 3  # 2*(budget+1) u3 gates, 3 params each
        n_cnots = budget
        
    elif qubits == 3:
        # Template: u3s on all 3 qubits, then CNOT
        # Cycle through CNOT patterns: (0,1), (0,2), (1,2)
        template = []
        cnot_patterns = [(0, 1), (0, 2), (1, 2)]
        for i in range(budget):
            # Add u3 gates on all 3 qubits
            for q in range(3):
                template.append(["u3", q, 0, 0, 0])
            # Add CNOT
            c, t = cnot_patterns[i % 3]
            template.append(["cx", c, t])
        # Add final u3s
        for q in range(3):
            template.append(["u3", q, 0, 0, 0])
        
        n_params = 3 * (budget + 1) * 3  # 3*(budget+1) u3 gates, 3 params each
        n_cnots = budget
    else:
        return []
    
    # Check total gate count
    if len(template) > 400:
        # Reduce budget
        while len(template) > 400 and budget > 0:
            budget -= 1
            # Rebuild template
            template = []
            if qubits == 2:
                for i in range(budget):
                    template.append(["u3", 0, 0, 0, 0])
                    template.append(["u3", 1, 0, 0, 0])
                    if i % 2 == 0:
                        template.append(["cx", 0, 1])
                    else:
                        template.append(["cx", 1, 0])
                template.append(["u3", 0, 0, 0, 0])
                template.append(["u3", 1, 0, 0, 0])
                n_params = 2 * (budget + 1) * 3
            else:
                cnot_patterns = [(0, 1), (0, 2), (1, 2)]
                for i in range(budget):
                    for q in range(3):
                        template.append(["u3", q, 0, 0, 0])
                    c, t = cnot_patterns[i % 3]
                    template.append(["cx", c, t])
                for q in range(3):
                    template.append(["u3", q, 0, 0, 0])
                n_params = 3 * (budget + 1) * 3
    
    if budget == 0:
        return []
    
    # Parameter vector: each u3 gate has 3 parameters (theta, phi, lam)
    # We'll optimize these parameters
    
    def _build_from_params(params):
        """Build unitary from parameter vector."""
        gates = []
        p_idx = 0
        for gate in template:
            if gate[0] == "u3":
                q = gate[1]
                theta = params[p_idx]
                phi = params[p_idx + 1]
                lam = params[p_idx + 2]
                p_idx += 3
                gates.append(["u3", q, theta, phi, lam])
            else:
                gates.append(gate)
        return _build_fast(gates, qubits)
    
    def _objective(params):
        """Objective function: distance to target."""
        U = _build_from_params(params)
        return _distance(target, U)
    
    # Multi-start Nelder-Mead optimization
    best_params = None
    best_dist = 1.0
    
    n_starts = 5
    per_start_time = max(0.5, (_time_left() - 0.5) / n_starts)
    
    for start in range(n_starts):
        if _time_left() < 1.0:
            break
        
        # Initialize parameters
        if start == 0:
            # Start from zero
            params = np.zeros(n_params)
        else:
            # Random initialization
            rng = np.random.default_rng(seed=42 + start)
            params = rng.uniform(-math.pi, math.pi, size=n_params)
        
        # Nelder-Mead optimization
        from scipy.optimize import minimize
        
        max_iter = int(min(200, max(20, per_start_time / 0.05)))
        
        try:
            result = minimize(
                _objective,
                params,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True,
                    'maxfev': int(min(500, max(50, per_start_time / 0.02)))
                }
            )
            
            dist = result.fun
            if dist < best_dist:
                best_dist = dist
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # If we found a good solution, build the gate list
    if best_params is not None and best_dist < 0.1:
        gates = []
        p_idx = 0
        for gate in template:
            if gate[0] == "u3":
                q = gate[1]
                theta = best_params[p_idx]
                phi = best_params[p_idx + 1]
                lam = best_params[p_idx + 2]
                p_idx += 3
                gates.append(["u3", q, float(theta), float(phi), float(lam)])
            else:
                gates.append(gate)
        return gates
    
    return []


def synthesize(problem):
    """
    Synthesize circuit for the given target unitary.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try known circuits first
    if qubits == 2:
        # CNOT gate
        if budget >= 1:
            gates = _known_circuit_cnot_2q()
            if _check_close(target, gates, qubits):
                return gates
        
        # SWAP gate
        if budget >= 3:
            gates = _known_circuit_swap_2q()
            if _check_close(target, gates, qubits):
                return gates
        
        # iSWAP gate
        if budget >= 2:
            gates = _known_circuit_iswap_2q()
            if _check_close(target, gates, qubits):
                return gates
    
    if qubits == 3:
        # Toffoli gate
        if budget >= 7:
            gates = _known_circuit_toffoli_3q()
            if _check_close(target, gates, qubits):
                return gates
        
        # Fredkin gate
        if budget >= 3:
            gates = _fredkin_circuit()
            if _check_close(target, gates, qubits):
                return gates
        
        # CCZ gate
        if budget >= 3:
            gates = _ccz_circuit()
            if _check_close(target, gates, qubits):
                return gates
    
    # For generic targets, use optimization
    if budget > 0 and _time_left() > 2.0:
        gates = _optimize_generic(target, qubits, budget)
        if gates:
            # Verify the solution
            if _check_close(target, gates, qubits, tol=1e-6):
                return gates
            # Even if not exact, return if it's reasonably close
            U = _build_fast(gates, qubits)
            d = _distance(target, U)
            if d < 0.5:
                return gates
    
    # Fallback: return identity
    return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
# EVOLVE-BLOCK-END
