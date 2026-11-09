"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Deadline guard: 270s wall-clock kill, keep 30s safety margin
_DEADLINE = time.time() + 240.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the full unitary from a gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    """Return the 2x2 u3 matrix for given parameters."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _apply_u3_to_subblock(matrix, qubit, theta, phi, lam, qubits):
    """Apply u3 on a specific qubit to the full matrix."""
    single = _u3_matrix(theta, phi, lam)
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator @ matrix


def _cnot_matrix(control, target, qubits):
    """Return the full CNOT matrix for given control and target."""
    dim = 2 ** qubits
    result = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> (qubits - 1 - control)) & 1:
            j = i ^ (1 << (qubits - 1 - target))
            result[i, j] = 1.0
    return result


def _extract_u3_from_subblock(matrix, qubit, qubits):
    """
    Extract optimal u3 parameters for a qubit by looking at the reduced state.
    We use the Schmidt decomposition approach: trace out other qubits,
    then fit a u3 to the reduced 2x2 block.
    """
    dim = 2 ** qubits
    half = dim // 2
    
    # Reshape matrix to isolate the target qubit
    # The qubit is at position `qubit` (0 = MSB)
    # We want to find U such that U @ (reduced) is as close to identity-like as possible
    
    # Extract the submatrix corresponding to this qubit
    # For qubit q, the matrix can be seen as a 2x2 block matrix where each block
    # is a (dim/2) x (dim/2) matrix
    
    # Reshape: (2, half, 2, half) where first 2 is the target qubit row, second is col
    reshaped = matrix.reshape(2, half, 2, half)
    
    # Contract over all other qubits to get a 2x2 effective matrix
    # This is the partial trace over all other qubits
    # effective[i,j] = sum_k reshaped[i, k, j, k]
    effective = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            effective[i, j] = np.sum(reshaped[i, :, j, :].diagonal())
    
    # Now we want to find u3 params (theta, phi, lam) such that
    # u3 @ effective is as close to a scalar multiple of identity as possible
    # Actually, we want the optimal left-multiplier that makes the result closest to identity
    
    # Use SVD: effective = U @ S @ Vh
    # The optimal u3 to make effective as close to scalar*I is related to U
    U, S, Vh = np.linalg.svd(effective)
    
    # The phase and rotation can be extracted from U
    # u3(theta, phi, lam) should be close to U up to a global phase
    # Let's fit: minimize |u3 - U|
    
    # Global phase: make U[0,0] real and positive
    phase = np.angle(U[0, 0]) if abs(U[0, 0]) > 1e-10 else 0.0
    U_rotated = U * np.exp(-1j * phase)
    
    # Extract theta, phi, lam from U_rotated
    # u3 = [[cos(t/2), -exp(i*l)*sin(t/2)], [exp(i*p)*sin(t/2), exp(i*(p+l))*cos(t/2)]]
    
    a = U_rotated[0, 0]
    b = U_rotated[0, 1]
    c = U_rotated[1, 0]
    d = U_rotated[1, 1]
    
    # cos(t/2) = |a|, sin(t/2) = |b| = |c|
    # Let's be careful with edge cases
    cos_t_half = abs(a)
    sin_t_half = abs(b)
    
    # Clamp for numerical stability
    cos_t_half = max(0, min(1, cos_t_half))
    sin_t_half = max(0, min(1, sin_t_half))
    
    theta = 2 * math.atan2(sin_t_half, cos_t_half)
    
    if sin_t_half < 1e-10:
        # theta ~ 0, u3 ~ diagonal
        phi = np.angle(c) if abs(c) > 1e-10 else 0.0
        lam = 0.0
    else:
        # exp(i*phi) = c / sin(t/2) approximately
        phi = np.angle(c) - np.angle(sin_t_half) if abs(c) > 1e-10 else 0.0
        # exp(i*lam) = -b / sin(t/2)
        lam = np.angle(-b) - np.angle(sin_t_half) if abs(b) > 1e-10 else 0.0
    
    # Refine with a few iterations of local optimization
    best = (theta, phi, lam)
    best_err = float('inf')
    
    for _ in range(5):
        for dt in (-0.1, 0.1):
            for dp in (-0.1, 0.1):
                for dl in (-0.1, 0.1):
                    if dt == 0 and dp == 0 and dl == 0:
                        continue
                    t_new = best[0] + dt
                    p_new = best[1] + dp
                    l_new = best[2] + dl
                    u3_try = _u3_matrix(t_new, p_new, l_new)
                    err = np.sum(np.abs(u3_try - U_rotated) ** 2)
                    if err < best_err:
                        best_err = err
                        best = (t_new, p_new, l_new)
    
    return best


def _cnot_ladder_2qubit(num_cnots):
    """Generate a CNOT ladder pattern for 2 qubits."""
    pairs = [(0, 1), (1, 0)]
    ladder = []
    for i in range(num_cnots):
        ladder.append(pairs[i % 2])
    return ladder


def _cnot_ladder_3qubit(num_cnots):
    """Generate a CNOT ladder pattern for 3 qubits."""
    # Standard pattern: cycle through all directed pairs
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    ladder = []
    for i in range(num_cnots):
        ladder.append(pairs[i % len(pairs)])
    return ladder


def _decompose_with_ladder(target, qubits, budget, ladder):
    """
    Decompose target using a fixed CNOT ladder, optimizing u3 parameters.
    """
    # Build the full gate layout: u3 on all qubits before each CNOT, and u3 after
    # Layout: [u3 on all qubits] [cx] [u3 on all qubits] [cx] ... [u3 on all qubits]
    
    num_cx = budget
    layout = []
    
    for i in range(num_cx):
        # u3 on all qubits before this cx
        for q in range(qubits):
            layout.append(('u3', q, 0.0, 0.0, 0.0))
        # cx
        layout.append(('cx', ladder[i][0], ladder[i][1]))
    
    # u3 on all qubits at the end
    for q in range(qubits):
        layout.append(('u3', q, 0.0, 0.0, 0.0))
    
    # Count u3 parameters
    num_u3 = sum(1 for g in layout if g[0] == 'u3')
    num_params = 3 * num_u3
    
    # Initialize parameters to zero
    params = [0.0] * num_params
    
    def build_circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == 'u3':
                gates.append(['u3', gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(['cx', gate[1], gate[2]])
        return gates
    
    def evaluate(params):
        circuit = build_circuit(params)
        matrix = _build(circuit, qubits)
        return _distance(target, matrix), circuit
    
    best_params = params[:]
    best_dist, best_circuit = evaluate(best_params)
    
    # Iterative optimization: coordinate descent with random restarts
    rng = np.random.RandomState(42 + qubits)
    
    max_iter = 50
    step_sizes = [math.pi / (2 * (i + 1)) for i in range(8)]
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        improved = False
        for i in range(num_params):
            if time.time() > _DEADLINE - 30:
                    break
            
            # Try several steps
            for step in step_sizes:
                for sign in (-1, 1):
                    trial = params[:]
                    trial[i] = params[i] + sign * step
                    # Wrap angles
                    trial[i] = (trial[i] + math.pi) % (2 * math.pi) - math.pi
                    
                    dist, _ = evaluate(trial)
                    if dist < best_dist - 1e-12:
                        params = trial
                        best_dist = dist
                        best_params = trial[:]
                        best_circuit = build_circuit(trial)
                        improved = True
                        break
                if improved:
                    break
            
            # Also try random jump
            if not improved and rng.rand() < 0.3:
                trial = params[:]
                trial[i] = rng.uniform(-math.pi, math.pi)
                dist, _ = evaluate(trial)
                if dist < best_dist - 1e-12:
                    params = trial
                    best_dist = dist
                    best_params = trial[:]
                    best_circuit = build_circuit(trial)
                    improved = True
        
        if not improved:
            # Try a larger random perturbation
            if rng.rand() < 0.5:
                trial = best_params[:]
                for i in range(min(num_params, 10)):
                    idx = rng.randint(0, num_params - 1)
                    trial[idx] += rng.uniform(-0.5, 0.5)
                dist, _ = evaluate(trial)
                if dist < best_dist - 1e-12:
                    params = trial
                    best_dist = dist
                    best_params = trial[:]
                    best_circuit = build_circuit(trial)
    
    return best_circuit


def _known_gates():
    """Return known exact circuits for standard gates."""
    known = {}
    
    # CNOT (2 qubits)
    known["CNOT"] = {
        'qubits': 2,
        'gates': [['cx', 0, 1]],
        'cnot_count': 1
    }
    
    # iSWAP (2 qubits): |01> -> i|10>, |10> -> i|01>
    # iSWAP = exp(i*pi/4 * (XX + YY))
    # Circuit: H on both, then iSWAP via CNOT-based circuit
    # iSWAP = (I ⊗ H) CNOT (I ⊗ S) CNOT (I ⊗ H) ... actually let me use a known decomposition
    # iSWAP = exp(i*pi/4 (XX+YY)) 
    # A standard circuit: 
    #   u3(0, pi/2, 0) on q0, u3(0, pi/2, 0) on q1  -- these are H gates (up to phase)
    #   Actually let me just use the standard:
    #   iSWAP = (H⊗H) CNOT (Z^{1/2}⊗I) CNOT (H⊗H) with phases
    # Let me use a simpler known form:
    # iSWAP matrix: [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # Using CNOTs: 
    #   u3(pi/2, 0, 0) on q0  -- H
    #   u3(pi/2, 0, 0) on q1  -- H
    #   cx(0,1)
    #   u3(pi/2, 0, 0) on q1  -- S gate (phase)
    #   cx(0,1)
    #   u3(pi/2, 0, 0) on q0  -- H
    #   u3(pi/2, 0, 0) on q1  -- H
    #   u3(pi, 0, 0) on q0     -- Z
    #   u3(pi, 0, 0) on q1     -- Z
    # This is getting complex. Let me just mark it for decomposition.
    
    # SWAP (2 qubits): 3 CNOTs
    known["SWAP"] = {
        'qubits': 2,
        'gates': [
            ['cx', 0, 1],
            ['cx', 1, 0],
            ['cx', 0, 1]
        ],
        'cnot_count': 3
    }
    
    # Toffoli (3 qubits): standard 6-CNOT decomposition
    known["Toffoli"] = {
        'qubits': 3,
        'gates': [
            ['u3', 2, math.pi/2, 0, 0],  # H on target
            ['cx', 1, 2],
            ['u3', 2, 0, 0, math.pi/4],  # T^dag on target
            ['cx', 0, 2],
            ['u3', 2, 0, 0, -math.pi/4],  # T on target
            ['cx', 1, 2],
            ['u3', 2, 0, 0, math.pi/4],  # T^dag on target
            ['u3', 0, 0, 0, -math.pi/4],  # T on control1
            ['cx', 0, 1],
            ['u3', 1, 0, 0, math.pi/4],  # T^dag on control2
            ['u3', 2, 0, 0, -math.pi/4],  # T on target
            ['cx', 0, 2],
            ['u3', 0, 0, 0, math.pi/4],  # T^dag on control1
            ['u3', 2, math.pi/2, 0, 0],  # H on target
        ],
        'cnot_count': 6
    }
    
    return known


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check deadline
    if time.time() > _DEADLINE - 30:
        return []
    
    # Try known gates first
    known = _known_gates()
    if name in known:
        kg = known[name]
        if kg['qubits'] == qubits and kg['cnot_count'] <= budget:
            # Verify the known circuit is correct
            matrix = _build(kg['gates'], qubits)
            dist = _distance(target, matrix)
            if dist < 1e-6:
                return kg['gates']
    
    # Use decomposition
    if qubits == 2:
        ladder = _cnot_ladder_2qubit(budget)
    else:
        ladder = _cnot_ladder_3qubit(budget)
    
    gates = _decompose_with_ladder(target, qubits, budget, ladder)
    
    # Verify CNOT count
    n_cnot = sum(1 for g in gates if g[0] == "cx")
    if n_cnot > budget:
        # Truncate to budget
        count = 0
        truncated = []
        for g in gates:
            if g[0] == "cx" and count >= budget:
                continue
            if g[0] == "cx":
                count += 1
            truncated.append(g)
        gates = truncated
    
    return gates
# EVOLVE-BLOCK-END
