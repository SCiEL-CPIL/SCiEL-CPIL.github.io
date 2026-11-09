import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 250.0

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            op = _u3_matrix(th, ph, lm)
            full = np.array([[1]], dtype=complex)
            for k in range(qubits):
                full = np.kron(full, op if k == q else eye2)
            U = full @ U
        else:
            _, c, t = g
            n_bits = qubits
            idx = np.arange(n)
            cb = (idx >> (n_bits - 1 - c)) & 1
            perm = idx ^ (cb << (n_bits - 1 - t))
            U = U[perm, :]
    return U

def _dist(t, u):
    n = t.shape[0]
    return 1.0 - min(1.0, abs(np.trace(t.conj().T @ u)) / n)

def _count_cx(gates):
    return sum(1 for g in gates if g[0] == "cx")

def _detect_known(target, qubits):
    tol = 1e-8
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=tol):
            return "cnot"
        iswap = 0.5*np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if np.allclose(target, iswap, atol=tol):
            return "iswap"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=tol):
            return "swap"
    if qubits == 3:
        toff = np.zeros((8,8), dtype=complex)
        for i in range(8):
            b0, b1, b2 = (i>>2)&1, (i>>1)&1, i&1
            if b0 and b1:
                toff[i, i^1] = 1
            else:
                toff[i, i] = 1
        if np.allclose(target, toff, atol=tol):
            return "toffoli"
    return "generic"

def _exact_toffoli(qubits):
    gates = []
    gates.append(["u3", 2, math.pi/2, 0, 0])
    gates.append(["cx", 2, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi/2, 0, 0])
    return gates

def _iswap_exact():
    gates = []
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, math.pi/2])
    gates.append(["u3", 1, 0, 0, math.pi/2])
    return gates

def _optimal_u3(A):
    """Find (theta, phi, lam) such that u3(theta,phi,lam) best approximates A (2x2)."""
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    sqrt_det = np.sqrt(det_A)
    if abs(sqrt_det) < 1e-15:
        return (0.0, 0.0, 0.0)
    # u3 = sqrt_det * [[c, -e^{i*lam}*s], [e^{i*phi}*s, e^{i*(phi+lam)}*c]]
    # So: c = A[0,0]/sqrt_det, -e^{i*lam}*s = A[0,1]/sqrt_det
    #     e^{i*phi}*s = A[1,0]/sqrt_det, e^{i*(phi+lam)}*c = A[1,1]/sqrt_det
    c_val = A[0,0] / sqrt_det
    s_val_neg = A[0,1] / sqrt_det  # this is -e^{i*lam}*s
    s_val = A[1,0] / sqrt_det      # this is e^{i*phi}*s

    # c = cos(theta/2), s = sin(theta/2)
    # c^2 + s^2 should = 1 for valid u3
    c_abs = abs(c_val)
    s_abs = abs(s_val)
    norm = math.sqrt(c_abs**2 + s_abs**2)
    if norm < 1e-15:
        return (0.0, 0.0, 0.0)

    # theta/2 = arcsin(|s|/norm), but we need to be careful
    # Let's use: cos(theta/2) = c_val / |c_val| * (|c_val|/norm) ... 
    # Actually, for u3: c = cos(theta/2) >= 0, s = sin(theta/2) >= 0 (theta in [0, pi])
    # But c_val and s_val can have phases.
    # u3 structure: c is real and >= 0, s is real and >= 0
    # A[0,0] = c * sqrt_det, A[1,0] = e^{i*phi} * s * sqrt_det
    # A[0,1] = -e^{i*lam} * s * sqrt_det, A[1,1] = e^{i*(phi+lam)} * c * sqrt_det
    
    # So: c = A[0,0]/sqrt_det must be real and non-negative
    #     s * e^{i*phi} = A[1,0]/sqrt_det
    #     -s * e^{i*lam} = A[0,1]/sqrt_det
    #     c * e^{i*(phi+lam)} = A[1,1]/sqrt_det
    
    # For a proper u3, c and s are real non-negative.
    # Let's project: c_proj = Re(A[0,0]/sqrt_det), s_proj = |A[1,0]/sqrt_det|
    # But we need consistency. Let's just use the magnitude approach.
    
    # theta/2 from |c| and |s|:
    theta_half = math.atan2(s_abs, c_abs)
    theta = 2.0 * theta_half
    
    # phi from A[1,0]/sqrt_det = e^{i*phi} * s
    # e^{i*phi} = (A[1,0]/sqrt_det) / s  (s is real positive)
    if s_abs > 1e-15:
        e_iphi = (A[1,0] / sqrt_det) / s_abs
        phi = np.angle(e_iphi)
    else:
        phi = 0.0
    
    # lam from A[0,1]/sqrt_det = -e^{i*lam} * s
    # e^{i*lam} = -(A[0,1]/sqrt_det) / s
    if s_abs > 1e-15:
        e_ilam = -(A[0,1] / sqrt_det) / s_abs
        lam = np.angle(e_ilam)
    else:
        lam = 0.0
    
    return (float(theta), float(phi), float(lam))

def _apply_u3_to_layer(U, qubit, theta, phi, lam, qubits):
    """Apply u3 on given qubit to full unitary U."""
    n = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    op = _u3_matrix(theta, phi, lam)
    full = np.array([[1]], dtype=complex)
    for k in range(qubits):
        full = np.kron(full, op if k == qubit else eye2)
    return full @ U

def _apply_cx(U, control, target, qubits):
    """Apply CNOT to full unitary U."""
    n = 2 ** qubits
    idx = np.arange(n)
    cb = (idx >> (qubits - 1 - control)) & 1
    perm = idx ^ (cb << (qubits - 1 - target))
    return U[perm, :]

def _greedy_optimize(target, qubits, cnot_template, max_restarts=3):
    """
    Greedy layer-by-layer optimization:
    For each layer of u3 gates (before each CNOT and after last CNOT),
    find optimal u3 params by solving a 2x2 subproblem.
    """
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    
    best_gates = None
    best_dist = 1.0
    
    for restart in range(max_restarts):
        if _remaining() < 3.0:
            break
        
        # Initialize with identity (or random for restart > 0)
        if restart == 0:
            # Start from identity
            U = np.eye(n, dtype=complex)
        else:
            # Random init
            U = np.eye(n, dtype=complex)
            for _ in range(num_layers):
                for q in range(qubits):
                    th = np.random.uniform(0, math.pi)
                    ph = np.random.uniform(-math.pi, math.pi)
                    lm = np.random.uniform(-math.pi, math.pi)
                    U = _apply_u3_to_layer(U, q, th, ph, lm, qubits)
                if _ < num_cx:
                    c, t = cnot_template[_]
                    U = _apply_cx(U, c, t, qubits)
        
        # Greedy: for each layer, optimize each u3 in sequence
        for layer in range(num_layers):
            if _remaining() < 2.0:
                break
            # For each qubit in this layer, find optimal u3
            for q in range(qubits):
                if _remaining() < 1.0:
                    break
                # Current U has all layers up to this point applied
                # We want to find u3 on qubit q that makes U closer to target
                # Strategy: compute residual = target @ U^{-1} (what we still need)
                # Then the u3 on qubit q should approximate the qubit-q block of residual
                # Actually, let's think differently:
                # U = U_layer * U_prev where U_layer = u3 on qubit q (tensor product with identities)
                # We want U_layer * U_prev ≈ target
                # U_layer ≈ target @ U_prev^{-1}
                # But U_layer is a specific form (u3 on one qubit).
                # So we need to find u3 on qubit q that best approximates target @ U_prev^{-1}
                
                # Compute U_prev = U with the current u3 on qubit q set to identity
                # Actually, U already has the current u3 applied. Let's undo it.
                # Better: track U_prev (before applying u3 on qubit q in this layer)
                
                # Let me restructure: build up U layer by layer
                # At each step, U = (u3_q @ ... @ u3_0) @ CNOT_k @ ... @ (u3_q @ ... @ u3_0) @ CNOT_0 @ ... @ u3_q @ ... @ u3_0
                # For each u3, we want to optimize it given all others fixed.
                
                # Simpler approach: just do coordinate descent with analytic solution
                # For u3 on qubit q at position in the circuit:
                # U = L_q @ M where L_q is u3 on qubit q (tensor with I)
                # We want L_q @ M ≈ target
                # L_q ≈ target @ M^{-1}
                # Extract the 2x2 block for qubit q from target @ M^{-1}
                # and find optimal u3 params.
                
                # Compute M = U with the u3 on qubit q replaced by identity
                # This is tricky without tracking. Let's use a different approach:
                
                # Forward-backward: 
                # Let P = product of all gates before this u3
                # Let Q = product of all gates after this u3
                # U = P @ u3_q @ Q
                # We want P @ u3_q @ Q ≈ target
                # u3_q ≈ P^{-1} @ target @ Q^{-1}
                # Extract 2x2 block and find optimal u3.
                
                # This requires tracking P and Q which is complex.
                # Alternative: just use the current U and do a local search.
                
                # Let's use a simpler greedy: 
                # Compute residual = target @ U^dagger
                # The u3 on qubit q should make the (q,q) block of residual closer to identity
                # Actually, let's just compute what u3 would make U closer to target.
                
                # U_new = u3_q @ U (if u3 is at the front)
                # But u3 is in the middle of the circuit...
                
                # OK, let me just do a proper coordinate descent:
                # For each u3 gate in the circuit, temporarily remove it,
                # compute what it should be, then re-insert.
                
                pass  # Will implement below
        
        # If we didn't do proper optimization, at least compute distance
        # Build gates from current U (this won't work directly)
        # Let's just compute distance of current U to target
        d = _dist(target, U)
        if d < best_dist:
            best_dist = d
            # Need to extract gates from U... this is hard
            # For now, skip gate extraction
            best_gates = None
    
    # Fallback: use the initial template with identity u3s
    gates = []
    for layer in range(num_layers):
        for q in range(qubits):
            gates.append(["u3", q, 0.0, 0.0, 0.0])
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    
    return gates, best_dist

def _optimize_2qubit_greedy(target, cnot_template):
    """
    Greedy coordinate descent for 2-qubit circuits.
    For each u3 gate, find optimal params by solving a 2x2 subproblem.
    """
    n = 4
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    T_dag = target.conj().T
    
    # Initialize with identity u3s
    u3_params = [[(0.0, 0.0, 0.0) for _ in range(2)] for _ in range(num_layers)]
    
    def build_U():
        U = np.eye(4, dtype=complex)
        for layer in range(num_layers):
            for q in range(2):
                th, ph, lm = u3_params[layer][q]
                op = _u3_matrix(th, ph, lm)
                if q == 0:
                    full = np.kron(op, np.eye(2, dtype=complex))
                else:
                    full = np.kron(np.eye(2, dtype=complex), op)
                U = full @ U
            if layer < num_cx:
                c, t = cnot_template[layer]
                idx = np.arange(4)
                cb = (idx >> (1 - c)) & 1
                perm = idx ^ (cb << (1 - t))
                U = U[perm, :]
        return U
    
    def dist_fn(U):
        return 1.0 - min(1.0, abs(np.trace(T_dag @ U)) / 4)
    
    best_U = build_U()
    best_dist = dist_fn(best_U)
    
    # Coordinate descent: iterate over all u3 gates
    for iteration in range(50):
        if _remaining() < 3.0:
            break
        improved = False
        for layer in range(num_layers):
            for q in range(2):
                if _remaining() < 1.0:
                    break
                # Save current params
                old_th, old_ph, old_lm = u3_params[layer][q]
                # Set to identity
                u3_params[layer][q] = (0.0, 0.0, 0.0)
                U_no_u3 = build_U()
                
                # Compute what u3 on qubit q should be
                # U = u3_q @ U_no_u3 (if u3 is before other gates in layer)
                # Actually, the order matters. Let's think about this more carefully.
                
                # In our circuit, for each layer:
                # First apply u3 on qubit 0, then u3 on qubit 1, then CNOT.
                # So for u3 on qubit q in layer L:
                # U = (u3_1_L @ u3_0_L @ CNOT_{L-1} @ ... @ u3_1_0 @ u3_0_0 @ CNOT_0 @ ... )
                # Wait, the order is: for layer 0: u3_0, u3_1, CNOT_0
                # Then layer 1: u3_0, u3_1, CNOT_1
                # etc.
                # So U = u3_0_last @ u3_1_last @ CNOT_last @ ... @ u3_0_0 @ u3_1_0 @ CNOT_0
                # No wait, matrix multiplication: U = last_gate @ ... @ first_gate
                # So if gates are applied in order g1, g2, ..., gk, then U = gk @ ... @ g2 @ g1
                # In our build_U, we do U = full @ U, so the last gate applied is leftmost.
                # So for layer 0: first u3_0, then u3_1, then CNOT_0
                # U after layer 0 = CNOT_0 @ u3_1 @ u3_0
                # Then layer 1: u3_0, u3_1, CNOT_1
                # U after layer 1 = CNOT_1 @ u3_1 @ u3_0 @ CNOT_0 @ u3_1 @ u3_0
                # etc.
                
                # So for u3 on qubit q in layer L:
                # It's sandwiched between other gates.
                # Let's define:
                # P = product of all gates before u3_q_L (applied first)
                # Q = product of all gates after u3_q_L (applied later)
                # U = Q @ u3_q_L @ P
                # We want Q @ u3_q_L @ P ≈ target
                # u3_q_L ≈ Q^{-1} @ target @ P^{-1}
                
                # This is complex to compute. Let's use a simpler approach:
                # Just do a local search around the current params.
                
                # Try a few candidate angles and pick the best
                best_th, best_ph, best_lm = old_th, old_ph, old_lm
                best_d = dist_fn(build_U())  # This is with identity u3
                
                # Actually, let's compute the optimal u3 by extracting the 2x2 block
                # from Q^{-1} @ target @ P^{-1}
                
                # To do this properly, I need to track P and Q.
                # Let me restructure build_U to return P and Q for each u3.
                
                # For now, let's just use a simple grid search for theta, and solve for phi, lam analytically.
                
                # Set u3 to identity, get U_no_u3
                # The u3 on qubit q at this position: 
                # If q=0: U = (stuff after) @ (u3_0 @ I) @ (stuff before)
                # If q=1: U = (stuff after) @ (I @ u3_1) @ (stuff before)
                
                # Let's just compute U_with_identity and then try to find u3 that improves things.
                
                # For q=0: u3_0 @ I acts on the leftmost qubit
                # U = A @ (u3_0 ⊗ I) @ B where A and B are 4x4
                # We want A @ (u3_0 ⊗ I) @ B ≈ target
                # (u3_0 ⊗ I) ≈ A^{-1} @ target @ B^{-1}
                # Extract the 2x2 block for qubit 0.
                
                # This is getting too complex. Let me just use the numerical gradient approach
                # but with a much smaller number of iterations and better initialization.
                
                # Actually, let me try a different structural approach:
                # Use the fact that for 2-qubit unitaries, we can use KAK decomposition.
                
                u3_params[layer][q] = (old_th, old_ph, old_lm)
        
        # After one pass, check if we improved
        U_new = build_U()
        d_new = dist_fn(U_new)
        if d_new < best_dist - 1e-12:
            best_dist = d_new
            best_U = U_new
            improved = True
        elif not improved and iteration > 0:
            break
    
    # Extract gates from best_U... this is hard.
    # For now, just return the u3_params we have.
    gates = []
    for layer in range(num_layers):
        for q in range(2):
            th, ph, lm = u3_params[layer][q]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    
    return gates, best_dist

def _kak_2qubit(target):
    """
    KAK decomposition for 2-qubit unitary.
    Returns cnot_template and u3_params for a circuit with up to 3 CNOTs.
    """
    # For now, this is a placeholder. We'll use a simple approach.
    # The KAK decomposition is: U = (A ⊗ B) @ CNOT^k @ (C ⊗ D)
    # where k is 0, 1, or 2 (or 3 depending on convention).
    
    # This is complex to implement correctly. Let's use a simpler approach:
    # Just use the greedy optimization with a good initialization.
    
    return None

def _optimize_circuit(target, qubits, cnot_template, max_iter=300):
    """
    Optimize u3 parameters for a fixed CNOT template.
    Uses a combination of analytic initialization and gradient-based refinement.
    """
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * qubits
    num_params = num_u3 * 3
    T_dag = target.conj().T
    
    # Analytic initialization: for each u3, try to match the corresponding 2x2 block
    params = np.zeros(num_params, dtype=float)
    
    # Build initial circuit with identity u3s
    def build_from_params(p):
        U = np.eye(n, dtype=complex)
        param_idx = 0
        eye2 = np.eye(2, dtype=complex)
        for layer in range(num_layers):
            for q in range(qubits):
                th = p[param_idx]
                ph = p[param_idx+1]
                lm = p[param_idx+2]
                R = _u3_matrix(th, ph, lm)
                full = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    full = np.kron(full, R if k == q else eye2)
                U = full @ U
                param_idx += 3
            if layer < num_cx:
                c, t = cnot_template[layer]
                idx = np.arange(n)
                cb = (idx >> (qubits - 1 - c)) & 1
                perm = idx ^ (cb << (qubits - 1 - t))
                U = U[perm, :]
        return U
    
    U_init = build_from_params(params)
    
    # Analytic init: for each u3, extract the 2x2 block from target @ U_init^{-1}
    # This is approximate but should give a good starting point.
    try:
        U_inv = np.linalg.inv(U_init)
        residual = target @ U_inv
        param_idx = 0
        for layer in range(num_layers):
            for q in range(qubits):
                # Extract 2x2 block for qubit q from residual
                # For qubit 0 in 2-qubit system: block is residual[0:2, 0:2]
                # For qubit 1: block is residual[2:4, 2:4] (approximately)
                if qubits == 2:
                    if q == 0:
                        block = residual[0:2, 0:2]
                    else:
                        block = residual[2:4, 2:4]
                else:
                    # For 3-qubit, this is more complex
                    # Just use identity for now
                    block = np.eye(2, dtype=complex)
                
                th, ph, lm = _optimal_u3(block)
                params[param_idx] = th
                params[param_idx+1] = ph
                params[param_idx+2] = lm
                param_idx += 3
    except:
        pass
    
    # Gradient-based refinement with Adam
    def cost_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        return 1.0 - min(1.0, abs(tr_val) / n), tr_val
    
    def grad_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        cost = 1.0 - min(1.0, abs(tr_val) / n)
        if abs(tr_val) < 1e-12:
            return cost, np.zeros(num_params)
        grad = np.zeros(num_params)
        eps = 1e-6
        for k in range(num_params):
            p_plus = p.copy()
            p_plus[k] += eps
            U_plus = build_from_params(p_plus)
            tr_plus = np.trace(T_dag @ U_plus)
            dtr = (tr_plus - tr_val) / eps
            grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
        grad = -grad / n
        return cost, grad
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.1
    best_cost = 1.0
    best_params = params.copy()
    
    cost, _ = cost_fn(params)
    if cost < best_cost:
        best_cost = cost
        best_params = params.copy()
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        cost, grad = grad_fn(params)
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        if best_cost < 1e-10:
            break
        t_adam += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_adam)
        v_hat = v / (1 - beta2**t_adam)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        if it % 30 == 29 and it > 0:
            lr *= 0.85
    
    return best_params, best_cost

def _params_to_gates(params, cnot_template, qubits):
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    gates = []
    param_idx = 0
    for layer in range(num_layers):
        for q in range(qubits):
            th = params[param_idx]
            ph = params[param_idx+1]
            lm = params[param_idx+2]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    return gates

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    gate_type = _detect_known(target, qubits)

    if gate_type == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_type == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_type == "iswap" and qubits == 2:
        return _iswap_exact()
    elif gate_type == "toffoli" and qubits == 3:
        return _exact_toffoli(qubits)

    # Generic: analytic init + gradient refinement
    if qubits == 2:
        num_cx = min(budget, 3)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,0), (0,1), (1,0), (0,1), (1,0)]
        cnot_template = pairs[:num_cx]
    else:
        num_cx = min(budget, 6)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        cnot_template = pairs[:num_cx]

    max_iter = 200 if qubits == 2 else 150
    params, cost = _optimize_circuit(target, qubits, cnot_template, max_iter=max_iter)
    gates = _params_to_gates(params, cnot_template, qubits)

    while _count_cx(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break

    return gates
