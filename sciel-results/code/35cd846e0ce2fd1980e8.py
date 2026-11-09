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

def _u3_from_matrix(A):
    """
    Given a 2x2 unitary matrix A, find (theta, phi, lam) such that
    u3(theta, phi, lam) = A up to global phase.
    If A is not exactly a u3 (e.g. has a global phase), we factor it out.
    """
    # u3(theta, phi, lam) = [[c, -e^{i lam} s], [e^{i phi} s, e^{i(phi+lam)} c]]
    # where c = cos(theta/2), s = sin(theta/2)
    
    # First, check if A is unitary (up to numerical precision)
    # A^dagger A should be ~ I
    Adag = A.conj().T
    prod = Adag @ A
    I = np.eye(2, dtype=complex)
    if not np.allclose(prod, I, atol=1e-6):
        # Not unitary, return identity
        return (0.0, 0.0, 0.0)
    
    # The determinant of u3 is e^{i(phi+lam)} * c^2 - (-e^{i lam} s)(e^{i phi} s)
    # = e^{i(phi+lam)} c^2 + e^{i(phi+lam)} s^2 = e^{i(phi+lam)}
    # So det(u3) = e^{i(phi+lam)}
    
    det_A = A[0,0] * A[1,1] - A[0,1] * A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    # We want to find theta, phi, lam such that:
    # A[0,0] = c * e^{i alpha}  (some global phase e^{i alpha})
    # Actually, let's factor out the global phase.
    # Let A = e^{i alpha} * B where B has det(B) = 1 (SL(2,C) part)
    # Then B = A / e^{i alpha}
    # And B should be in the form of u3 up to a phase that makes det = 1.
    
    # For u3, det = e^{i(phi+lam)}. Let's call this phase P.
    # So u3 = P * M where M has det(M) = 1.
    # M = [[c e^{-i phi/2} e^{-i lam/2}, -e^{i lam/2} s e^{-i phi/2}],
    #      [e^{i phi/2} s e^{-i lam/2}, c e^{i phi/2} e^{i lam/2}]]
    # This is getting complicated. Let's use a simpler approach.
    
    # We know:
    # A[0,0] = c * e^{i alpha}
    # A[1,1] = c * e^{i(alpha + phi + lam)}
    # A[1,0] = s * e^{i(alpha + phi)}
    # A[0,1] = -s * e^{i(alpha + lam)}
    
    # Let's estimate the global phase alpha.
    # |A[0,0]| = c, |A[1,0]| = s
    # c^2 + s^2 = 1
    c_abs = abs(A[0,0])
    s_abs = abs(A[1,0])
    norm = math.sqrt(c_abs**2 + s_abs**2)
    if norm < 1e-15:
        return (0.0, 0.0, 0.0)
    
    # Normalize to ensure c^2 + s^2 = 1
    c_val = c_abs / norm
    s_val = s_abs / norm
    
    theta = 2.0 * math.atan2(s_val, c_val)
    
    # Now find phi and lam.
    # A[1,0] = s * e^{i(alpha + phi)}
    # A[0,1] = -s * e^{i(alpha + lam)}
    
    # e^{i(alpha + phi)} = A[1,0] / s
    # e^{i(alpha + lam)} = -A[0,1] / s
    
    # We don't need alpha explicitly if we can find phi and lam relative to each other.
    # Let's assume the global phase is such that A[0,0] is real and positive.
    # alpha = -angle(A[0,0])
    # Then A[0,0] / e^{i alpha} = c (real positive)
    
    # Let's define B = A * e^{-i alpha} where alpha = -angle(A[0,0])
    # Then B[0,0] = c (real positive)
    # B[1,0] = s * e^{i phi}
    # B[0,1] = -s * e^{i lam}
    # B[1,1] = c * e^{i(phi + lam)}
    
    alpha = -np.angle(A[0,0])
    B = A * np.exp(-1j * alpha)
    
    # Now B[0,0] should be real positive (c)
    # B[1,0] = s * e^{i phi}
    # B[0,1] = -s * e^{i lam}
    
    if s_val > 1e-15:
        # e^{i phi} = B[1,0] / s_val
        e_iphi = B[1,0] / s_val
        phi = np.angle(e_iphi)
        
        # e^{i lam} = -B[0,1] / s_val
        e_ilam = -B[0,1] / s_val
        lam = np.angle(e_ilam)
    else:
        phi = 0.0
        lam = 0.0
    
    return (float(theta), float(phi), float(lam))

def _compute_circuit(params, cnot_template, qubits):
    """Compute the circuit unitary from params and fixed CNOT template."""
    n = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    U = np.eye(n, dtype=complex)
    param_idx = 0
    for layer in range(num_layers):
        for q in range(qubits):
            th = params[param_idx]
            ph = params[param_idx+1]
            lm = params[param_idx+2]
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

def _optimize_circuit(target, qubits, cnot_template, max_iter=300):
    """
    Optimize u3 parameters for a fixed CNOT template using Adam gradient descent.
    Uses numerical gradients (forward differences).
    """
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * qubits
    num_params = num_u3 * 3
    T_dag = target.conj().T

    np.random.seed(42)
    params = np.zeros(num_params, dtype=float)
    for i in range(num_u3):
        params[3*i] = np.random.uniform(0, math.pi)
        params[3*i+1] = np.random.uniform(-math.pi, math.pi)
        params[3*i+2] = np.random.uniform(-math.pi, math.pi)

    def cost_fn(p):
        U = _compute_circuit(p, cnot_template, qubits)
        tr_val = np.trace(T_dag @ U)
        return 1.0 - min(1.0, abs(tr_val) / n), tr_val

    def grad_fn(p):
        U = _compute_circuit(p, cnot_template, qubits)
        tr_val = np.trace(T_dag @ U)
        cost = 1.0 - min(1.0, abs(tr_val) / n)
        if abs(tr_val) < 1e-12:
            return cost, np.zeros(num_params)
        grad = np.zeros(num_params)
        eps = 1e-5
        for k in range(num_params):
            p_plus = p.copy()
            p_plus[k] += eps
            U_plus = _compute_circuit(p_plus, cnot_template, qubits)
            tr_plus = np.trace(T_dag @ U_plus)
            dtr = (tr_plus - tr_val) / eps
            # d|tr|/dp = Re(conj(tr) * dtr) / |tr|
            grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
        # We minimize cost = 1 - |tr|/n, so dcost/dp = -grad/n
        grad = -grad / n
        return cost, grad

    # Adam
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.05
    best_cost = 1.0
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
        if it % 50 == 49 and it > 0:
            lr *= 0.8

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
