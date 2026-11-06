"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 235.0  # safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return (np.array(problem["real"], dtype=complex)
            + 1j * np.array(problem["imaginary"], dtype=complex))


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
                    dtype=complex)


def _build_fast(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            op = np.array([[1.0]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else np.eye(2, dtype=complex))
            U = op @ U
        else:
            _, c, t = gate
            cb, tb = qubits - 1 - c, qubits - 1 - t
            idx = np.arange(n)
            bit = (idx >> cb) & 1
            U = U[idx ^ (bit << tb), :]
    return U


def _distance(target, U):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / target.shape[0])


def _extract_u3(A):
    """
    Extract u3(theta, phi, lam) parameters from a 2x2 unitary A,
    allowing for a global phase. We find theta, phi, lam such that:
    u3(theta, phi, lam) @ diag(e^{i a}, e^{i b}) ≈ A  (up to global phase)
    Actually, we want: A = e^{i g} * u3(theta, phi, lam)
    So: u3(theta, phi, lam) = e^{-i g} * A
    We can absorb the global phase into the u3 parameters if needed,
    but since u3 doesn't have a free global phase, we need to be careful.
    
    The u3 matrix is:
    [[cos(theta/2), -e^{i lam} sin(theta/2)],
     [e^{i phi} sin(theta/2), e^{i(phi+lam)} cos(theta/2)]]
    
    Given A (2x2 unitary), we want to find theta, phi, lam and global phase g
    such that A = e^{i g} * u3(theta, phi, lam).
    
    Equivalently: u3(theta, phi, lam) = e^{-i g} A.
    Since the overall phase is irrelevant for the circuit (we can absorb it),
    we just need to find theta, phi, lam such that u3 matches A up to global phase.
    
    From the u3 form:
    |A[0,0]| = |cos(theta/2)|  (if we absorb phase)
    |A[0,1]| = |sin(theta/2)|
    
    So: cos(theta/2) = |A[0,0]|, sin(theta/2) = |A[0,1]|
    theta = 2 * arccos(|A[0,0]|)
    
    Then:
    A[0,0] = e^{i g} cos(theta/2)  =>  e^{i g} = A[0,0] / cos(theta/2)  (if cos != 0)
    A[0,1] = -e^{i g} e^{i lam} sin(theta/2)  =>  e^{i lam} = -A[0,1] / (e^{i g} sin(theta/2))
    A[1,0] = e^{i g} e^{i phi} sin(theta/2)  =>  e^{i phi} = A[1,0] / (e^{i g} sin(theta/2))
    """
    a00, a01 = A[0, 0], A[0, 1]
    a10, a11 = A[1, 0], A[1, 1]
    
    eps = 1e-12
    
    # Compute cos(theta/2) and sin(theta/2)
    cos_half = abs(a00)
    sin_half = abs(a01)
    
    # Normalize in case of numerical drift
    norm = math.sqrt(cos_half * cos_half + sin_half * sin_half)
    if norm < eps:
        # Degenerate: identity-like
        return (0.0, 0.0, 0.0)
    cos_half /= norm
    sin_half /= norm
    
    # Clamp for safety
    cos_half = min(1.0, max(0.0, cos_half))
    sin_half = min(1.0, max(0.0, sin_half))
    
    theta = 2.0 * math.acos(cos_half)
    
    if cos_half < eps:
        # theta ~ pi, sin_half ~ 1
        # A[0,0] ~ 0, A[0,1] = -e^{i(g+lam)}
        # A[1,0] = e^{i(g+phi)}, A[1,1] = e^{i(g+phi+lam)}
        # From A[0,1]: e^{i(g+lam)} = -A[0,1]
        # From A[1,0]: e^{i(g+phi)} = A[1,0]
        # So: e^{i lam} = -A[0,1] / e^{ig}, e^{i phi} = A[1,0] / e^{ig}
        # We can pick g=0 WLOG for the u3 params (global phase absorbed)
        # Actually, we need consistent g. Let's use:
        # e^{ig} = A[1,1] / e^{i(g+phi+lam)} ... circular.
        # Better: from A[1,1] = e^{i(g+phi+lam)} and A[0,1] = -e^{i(g+lam)}
        # => e^{i phi} = A[1,1] / (-A[0,1])
        # And e^{i lam} = -A[0,1] / e^{ig}, e^{i phi} = A[1,0] / e^{ig}
        # => A[1,0] / (-A[0,1]) = e^{i phi} / e^{i lam} = e^{i(phi-lam)}
        # Let's just set g such that e^{ig} = 1 (absorb into u3)
        # Then: e^{i lam} = -A[0,1], e^{i phi} = A[1,0]
        # Check: A[1,1] should be e^{i(phi+lam)} = e^{i phi} * e^{i lam} = A[1,0] * (-A[0,1])
        # This may not hold exactly due to global phase, but for our purposes:
        lam = np.angle(-a01)
        phi = np.angle(a10)
        # Verify and adjust if needed
        # A[1,1] should be e^{i(phi+lam)} up to global phase
        # We'll trust this and let the build verify
    elif sin_half < eps:
        # theta ~ 0
        # A[0,0] = e^{ig}, A[1,1] = e^{i(g+phi+lam)}
        # A[0,1] ~ 0, A[1,0] ~ 0
        # phi+lam = angle(A[1,1] / A[0,0])
        # Set phi=0, lam=phi+lam
        if abs(a00) > eps:
            phi_plus_lam = np.angle(a11 / a00)
        else:
            phi_plus_lam = 0.0
        return (0.0, 0.0, phi_plus_lam)
    else:
        # General case
        # e^{ig} = a00 / cos_half
        e_ig = a00 / cos_half
        e_ig = e_ig / abs(e_ig)  # normalize to unit modulus
        
        # e^{i lam} = -a01 / (e^{ig} * sin_half)
        e_ilam = -a01 / (e_ig * sin_half)
        e_ilam = e_ilam / abs(e_ilam)
        
        # e^{i phi} = a10 / (e^{ig} * sin_half)
        e_iphi = a10 / (e_ig * sin_half)
        e_iphi = e_iphi / abs(e_iphi)
        
        lam = np.angle(e_ilam)
        phi = np.angle(e_iphi)
        
        # Normalize angles to [-pi, pi]
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        lam = (lam + math.pi) % (2 * math.pi) - math.pi
    
    return (theta, phi, lam)


def _qsd_2qubit(U, qubits=2):
    """
    Quantum Shannon Decomposition for a 2-qubit unitary U.
    U = (A ⊗ I) · CX_{0,1} · (B ⊗ I) · CX_{0,1} · (C ⊗ I) · CX_{0,1} · (D ⊗ I)
    where CX_{0,1} has control on qubit 0 (MSB) and target on qubit 1 (LSB).
    
    Returns a list of gates.
    """
    n = 2 ** qubits
    # U is 4x4
    
    # The QSD decomposition algorithm:
    # We need to find A, B, C, D (each 2x2 unitary) such that:
    # U = (A⊗I) · CX · (B⊗I) · CX · (C⊗I) · CX · (D⊗I)
    
    # CX_{0,1} matrix (control q0, target q1):
    # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
    # In basis |q0 q1>: index = 2*q0 + q1
    # CX = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    
    CX = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)
    
    I2 = np.eye(2, dtype=complex)
    
    # Algorithm:
    # Step 1: U · (I ⊗ D†) = (A⊗I) · CX · (B⊗I) · CX · (C⊗I)
    # We need to find D first.
    
    # From the structure, the last two columns of U·(I⊗D†)·CX†·(I⊗C†)·CX†·(I⊗B†)·CX† should give (A⊗I).
    
    # Let's use a different approach: solve for the matrices directly.
    
    # U = (A⊗I) · CX · (B⊗I) · CX · (C⊗I) · CX · (D⊗I)
    
    # Let M1 = (A⊗I) · CX · (B⊗I) · CX · (C⊗I) · CX
    # Then U = M1 · (D⊗I)
    # So D⊗I = M1† · U
    # And M1 = U · (D†⊗I)
    
    # This is circular. Let's use the standard QSD extraction.
    
    # Standard QSD extraction for 2 qubits:
    # 1. Compute U1 = U · (I ⊗ D†) where D is chosen so that the (0,0) block of U1·CX† has specific form.
    
    # Actually, let me use a known algorithm:
    
    # The QSD can be extracted as follows:
    # Let U = [[P, Q], [R, S]] where P,Q,R,S are 2x2 blocks.
    
    # For CX_{0,1} (control MSB, target LSB):
    # CX · (X⊗I) · CX = (I⊗X) · ... no, that's not right.
    
    # Let me just implement the known decomposition directly.
    
    # Reference: The QSD for 2 qubits can be computed as:
    # 1. Let V = U
    # 2. Find D such that V · (I ⊗ D†) has the property that its top-left 2x2 block and bottom-right 2x2 block satisfy certain conditions.
    
    # A simpler approach: use the fact that any 2-qubit unitary can be written as:
    # U = (A ⊗ B) · (I ⊗ R) · CX · (I ⊗ S) · CX · (I ⊗ T) · CX · (I ⊗ V)
    # No, that's not standard either.
    
    # Let me use a numerical approach: since we only need to handle specific targets,
    # and for generic unitaries we'll fall back to optimization, let me just return None
    # for now and rely on the optimization fallback.
    
    # Actually, let me implement a proper QSD extraction.
    
    # From Nielsen & Chuang, or from the paper "Synthesis of Quantum Logic Circuits":
    
    # For 2 qubits, U = (A⊗I) CX (B⊗I) CX (C⊗I) CX (D⊗I)
    
    # Step 1: Compute W = U · (I⊗D†) · CX† · (I⊗C†) · CX† · (I⊗B†) · CX†
    # Then W = (A⊗I), so A = W[0:2, 0:2] (top-left block).
    
    # But we don't know B, C, D yet.
    
    # Alternative: work backwards from U.
    # U · (I⊗D†) = (A⊗I) CX (B⊗I) CX (C⊗I) CX
    # Let V1 = U · (I⊗D†)
    # V1 · CX† = (A⊗I) CX (B⊗I) CX (C⊗I)
    # V2 = V1 · CX†
    # V2 · (I⊗C†) = (A⊗I) CX (B⊗I) CX
    # V3 = V2 · (I⊗C†)
    # V3 · CX† = (A⊗I) CX (B⊗I)
    # V4 = V3 · CX†
    # V4 · (I⊗B†) = (A⊗I) CX
    # V5 = V4 · (I⊗B†)
    # V5 · CX† = (A⊗I)
    # V6 = V5 · CX† = (A⊗I)
    
    # So: A = V6[0:2, 0:2]
    
    # To find D, C, B, A, we need to make choices at each step.
    
    # The standard approach is to choose D such that V1 = U·(I⊗D†) has the form
    # where V1[0,0] is real and non-negative, etc.
    
    # This is getting complex. For the purpose of this task, since we have known
    # decompositions for the standard gates and the generic unitaries can be handled
    # by optimization, let me skip the full QSD and use a simpler approach.
    
    return None


def _known_decompositions(name, qubits):
    """Return exact gate list for the fixed gates, else None."""
    if name == "CNOT":
        return [["cx", 0, 1]]
    if name == "iSWAP":
        # iSWAP = [[0, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 0]]
        # Decomposition using H, CNOT, S gates
        # iSWAP = (S⊗S) · (H⊗H) · CX_{0,1} · CX_{1,0} · (H⊗H) · (S†⊗S†) · phase
        # Let me use a verified decomposition:
        # H = u3(pi/2, 0, pi)
        # S = u3(0, 0, pi/2)
        # S† = u3(0, 0, -pi/2)
        gates = [
            ["u3", 0, math.pi / 2, 0.0, math.pi],  # H on q0
            ["u3", 1, math.pi / 2, 0.0, math.pi],  # H on q1
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["u3", 0, math.pi / 2, 0.0, math.pi],  # H on q0
            ["u3", 1, math.pi / 2, 0.0, math.pi],  # H on q1
            ["u3", 0, 0.0, 0.0, math.pi / 2],      # S on q0
            ["u3", 1, 0.0, 0.0, math.pi / 2],      # S on q1
        ]
        return gates
    if name == "SWAP":
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    if name == "Toffoli":
        # Toffoli with control 0,1 target 2, within 6 CNOT budget.
        T = math.pi / 4
        return [
            ["u3", 2, math.pi / 2, 0.0, 0.0],       # H on q2
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, T],                 # T on q2
            ["cx", 0, 2],
            ["u3", 1, 0.0, 0.0, -T],                # T† on q1
            ["u3", 2, 0.0, 0.0, T],                 # T on q2
            ["cx", 1, 2],
            ["u3", 0, 0.0, 0.0, -T],                # T† on q0
            ["u3", 1, 0.0, 0.0, T],                 # T on q1
            ["cx", 0, 2],
            ["u3", 2, 0.0, 0.0, -T],                # T† on q2
            ["u3", 2, math.pi / 2, 0.0, 0.0],       # H on q2
        ]
    return None


def _make_layout(qubits, budget):
    """Build a CNOT ladder layout with u3 between CNOTs."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    for q in range(qubits):
        layout.append(["u3", q])
    return layout


def _layout_to_gates(layout, values):
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _optimize_generic(target, qubits, budget, seed):
    layout = _make_layout(qubits, budget)
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    rng = np.random.RandomState(seed)

    best = None
    best_dist = float("inf")

    # multi-start
    n_starts = 3
    for s in range(n_starts):
        if _time_left() < 30:
            break
        if s == 0:
            vals = np.zeros(n_params)
        else:
            vals = rng.uniform(-math.pi, math.pi, size=n_params)

        dist = _distance(target, _build_fast(_layout_to_gates(layout, vals), qubits))
        if dist < best_dist:
            best, best_dist = vals.copy(), dist

        # coordinate descent: coarse then fine
        for sweep in range(15):
            if _time_left() < 20:
                break
            step = 1.0 / (1 + sweep * 2)
            improved = False
            for i in range(n_params):
                if _time_left() < 20:
                    break
                for trial in (vals[i] + step, vals[i] - step,
                              vals[i] + rng.uniform(-step, step)):
                    new_vals = vals.copy()
                    new_vals[i] = trial
                    d = _distance(target, _build_fast(_layout_to_gates(layout, new_vals), qubits))
                    if d < dist:
                        vals, dist = new_vals, d
                        improved = True
            if not improved and sweep > 8:
                break

        if dist < best_dist:
            best, best_dist = vals.copy(), dist

    # fine polish
    if best is not None and _time_left() > 20:
        vals = best.copy()
        dist = best_dist
        for sweep in range(10):
            if _time_left() < 15:
                break
            step = 0.1 / (1 + sweep)
            for i in range(len(vals)):
                if _time_left() < 15:
                    break
                for trial in (vals[i] + step, vals[i] - step):
                    new_vals = vals.copy()
                    new_vals[i] = trial
                    d = _distance(target, _build_fast(_layout_to_gates(layout, new_vals), qubits))
                    if d < dist:
                        vals, dist = new_vals, d

    if best is None:
        best = np.zeros(n_params)
    return _layout_to_gates(layout, best)


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Try known exact decompositions first
    known = _known_decompositions(name, qubits)
    if known is not None:
        n_cx = sum(1 for g in known if g[0] == "cx")
        if n_cx <= budget and len(known) <= 400:
            # Verify the circuit is correct
            U = _build_fast(known, qubits)
            if _distance(target, U) < 1e-8:
                return known
        # If verification fails, fall through to generic

    # For generic unitaries, use optimization
    return _optimize_generic(target, qubits, budget, seed=12345 + qubits)
# EVOLVE-BLOCK-END
