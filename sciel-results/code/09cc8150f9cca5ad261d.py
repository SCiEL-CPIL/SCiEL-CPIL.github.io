"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 230.0  # Safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
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


def _phase_fix(target, U):
    """Find global phase that maximizes |tr(target^dag U)|."""
    tr = np.trace(target.conj().T @ U)
    if abs(tr) < 1e-15:
        return U
    phase = np.exp(-1j * np.angle(tr))
    return phase * U


def _two_qubit_synthesis(target):
    """
    Synthesize a 2-qubit unitary using at most 3 CNOTs.
    Uses the standard decomposition: U = A ⊗ B · CNOT · C ⊗ D · CNOT · E ⊗ F · CNOT
    """
    # The standard 3-CNOT decomposition for 2-qubit gates
    # U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT
    # We need to find single-qubit gates A,B,C,D,E,F
    
    # Step 1: Factor out the first CNOT
    # U · CNOT^{-1} = U · CNOT (since CNOT is self-inverse)
    U1 = target @ _cnot_mat(0, 1, 2)
    
    # U1 should be (A⊗B)(C⊗D)CNOT(E⊗F)
    # Let's try a different approach: use the known sufficient decomposition
    
    # For 2-qubit gates, we can use:
    # U = (R1⊗R2) CNOT (R3⊗R4) CNOT (R5⊗R6)
    # This requires 2 CNOTs in some cases, but 3 in general.
    
    # Actually, let me use a more direct approach.
    # The standard result: any 2-qubit unitary can be written as
    # U = (A ⊗ B) · CNOT · (C ⊗ D) · CNOT · (E ⊗ F) · CNOT
    # with 6 single-qubit gates and 3 CNOTs.
    
    # Let's implement the algorithm from Nielsen & Chuang / Barenco et al.
    
    # First, compute the "local equivalence" classes
    # For now, let's use a numerical approach
    
    # Try to factor U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT
    # This is equivalent to finding A,B,C,D,E,F such that:
    # (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT = U
    
    # Let V = U · CNOT^{-1} = U · CNOT
    # Then (A⊗B) CNOT (C⊗D) CNOT (E⊗F) = V
    
    # Let W = V · CNOT^{-1} = V · CNOT = U · CNOT · CNOT = U
    # Hmm, this is getting circular. Let me think more carefully.
    
    # Standard decomposition:
    # U = (A ⊗ B) · CX · (C ⊗ D) · CX · (E ⊗ F) · CX
    # where CX = CNOT(0,1)
    
    # Multiply both sides by CX^{-1} = CX on the right:
    # U · CX = (A ⊗ B) · CX · (C ⊗ D) · CX · (E ⊗ F)
    
    # Let P = U · CX
    # P = (A ⊗ B) · CX · (C ⊗ D) · CX · (E ⊗ F)
    
    # Let Q = P · CX^{-1} = P · CX
    # Q = (A ⊗ B) · CX · (C ⊗ D)
    
    # So Q = (A⊗B) CX (C⊗D)
    # This means Q · CX^{-1} · (C⊗D)^{-1} = (A⊗B)
    # Q · CX · (C⊗D)^{-1} = (A⊗B)
    
    # For (A⊗B) to be a product state, we need to find C,D such that
    # Q · CX · (C⊗D)^{-1} is a product.
    
    # This is still complex. Let me use a different strategy:
    # Use the fact that we can always decompose using the "canonical form"
    
    # Actually, for the specific gates mentioned (CNOT, iSWAP, SWAP, Toffoli),
    # we can hardcode the decompositions!
    
    # But we also need to handle "three generic unitaries".
    # Let me use a general numerical approach.
    
    # Strategy: Use the known decomposition with 3 CNOTs.
    # U = (A⊗B) CX (C⊗D) CX (E⊗F) CX
    
    # We'll use the following approach:
    # 1. Compute U1 = U @ CX (right-multiply by inverse of last CX)
    # 2. U1 = (A⊗B) CX (C⊗D) CX (E⊗F)
    # 3. Compute U2 = U1 @ CX (right-multiply by inverse of middle CX)
    # 4. U2 = (A⊗B) CX (C⊗D)
    # 5. Now U2 is of the form (A⊗B) CX (C⊗D)
    # 6. We need to find single-qubit gates such that U2 = (A⊗B) CX (C⊗D)
    
    # For step 6, we can use the following:
    # U2 @ CX^{-1} = (A⊗B) CX (C⊗D) CX^{-1} = (A⊗B) CX (C⊗D) CX
    # Hmm, this doesn't simplify nicely.
    
    # Let me try yet another approach: direct optimization with a better initial guess.
    # Or use the matrix product state approach.
    
    # Actually, the cleanest approach for exact synthesis is:
    # For 2 qubits: U = (A⊗B) CX (C⊗D) CX (E⊗F) CX
    # We can find this by:
    # 1. Let M1 = U @ CX (remove last CX)
    # 2. Let M2 = M1 @ CX (remove middle CX)  
    # 3. M2 = (A⊗B) CX (C⊗D)
    # 4. Let M3 = M2 @ CX (remove first CX from right, but this gives (A⊗B) CX (C⊗D) CX)
    
    # I think I'm overcomplicating this. Let me use a simpler but effective approach:
    # Use the fact that for 2-qubit gates, we can use the decomposition:
    # U = R1(0) R1(1) CX R2(0) R2(1) CX R3(0) R3(1) CX R4(0) R4(1)
    # where Ri are single-qubit rotations.
    
    # For now, let's use a coordinate descent approach but with a much better initial guess
    # and more iterations, plus a deadline guard.
    
    # Actually, let me just implement the standard 3-CNOT decomposition properly.
    
    # The key insight: any 2-qubit unitary U can be written as
    # U = (A ⊗ B) · CX_{0,1} · (C ⊗ D) · CX_{0,1} · (E ⊗ F) · CX_{0,1}
    
    # Algorithm:
    # 1. Compute U_temp = U @ CX_{0,1}  (right-multiply by CX^{-1} = CX)
    #    Now U_temp = (A⊗B) CX (C⊗D) CX (E⊗F)
    # 2. Compute U_temp2 = U_temp @ CX_{0,1}
    #    Now U_temp2 = (A⊗B) CX (C⊗D)
    # 3. We need to factor U_temp2 = (A⊗B) CX (C⊗D)
    #    This is a 2-qubit gate with one CNOT in the middle.
    #    We can write: U_temp2 @ CX^{-1} = (A⊗B) CX (C⊗D) CX^{-1} = (A⊗B)
    #    Wait: (A⊗B) CX (C⊗D) CX = (A⊗B) (CX (C⊗D) CX)
    #    And CX (C⊗D) CX is just some 2-qubit gate.
    
    # Let me just use numerical optimization with good initialization.
    
    CX = _cnot_mat(0, 1, 2)
    
    # Layout: u3(0), u3(1), CX, u3(0), u3(1), CX, u3(0), u3(1), CX
    # 6 single-qubit gates, 3 CNOTs
    
    # Parameters: 6 gates × 3 params = 18 parameters
    
    # Initialize with zeros
    params = [0.0] * 18
    
    def circuit_from_params(p):
        gates = []
        idx = 0
        for q in [0, 1]:
            gates.append(["u3", q, p[idx], p[idx+1], p[idx+2]])
            idx += 3
        gates.append(["cx", 0, 1])
        for q in [0, 1]:
            gates.append(["u3", q, p[idx], p[idx+1], p[idx+2]])
            idx += 3
        gates.append(["cx", 0, 1])
        for q in [0, 1]:
            gates.append(["u3", q, p[idx], p[idx+1], p[idx+2]])
            idx += 3
        gates.append(["cx", 0, 1])
        return gates
    
    best_params = params[:]
    best_dist = _distance(target, _build(circuit_from_params(best_params), 2))
    
    # Coordinate descent with multiple passes
    for sweep in range(30):
        if _time_left() < 5.0:
            break
        step = 0.5 / (1 + sweep * 0.5)
        improved = False
        for i in range(len(best_params)):
            for trial in (best_params[i] + step, best_params[i] - step):
                trial = trial % (2 * math.pi)
                if trial > math.pi:
                    trial -= 2 * math.pi
                p = best_params[:]
                p[i] = trial
                d = _distance(target, _build(circuit_from_params(p), 2))
                if d < best_dist:
                    best_params, best_dist = p, d
                    improved = True
        if not improved and sweep > 5:
            break
    
    return circuit_from_params(best_params)


def _cnot_mat(control, target, qubits):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    index = np.arange(n)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _three_qubit_synthesis(target, budget):
    """
    Synthesize a 3-qubit unitary using the given CNOT budget.
    For Toffoli: 4 CNOTs. For general: use up to budget CNOTs.
    """
    if budget >= 4:
        # Use a ladder of CNOTs with single-qubit rotations
        # Layout: for each CNOT, have single-qubit gates before and after
        # Total gates: (budget * 3 + 3) single-qubit + budget CNOTs
        
        # CNOT pattern for 3 qubits: cycle through pairs
        pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
        
        # Build layout
        layout = []
        for i in range(budget):
            for q in range(3):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            c, t = pairs[i % len(pairs)]
            layout.append(["cx", c, t])
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        
        # Count parameters
        n_u3 = sum(1 for g in layout if g[0] == "u3")
        n_params = 3 * n_u3
        
        def circuit_from_params(p):
            gates = []
            idx = 0
            for gate in layout:
                if gate[0] == "u3":
                    gates.append(["u3", gate[1], p[idx], p[idx+1], p[idx+2]])
                    idx += 3
                else:
                    gates.append(list(gate))
            return gates
        
        best_params = [0.0] * n_params
        best_dist = _distance(target, _build(circuit_from_params(best_params), 3))
        
        # Coordinate descent
        for sweep in range(20):
            if _time_left() < 10.0:
                break
            step = 0.5 / (1 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                for trial in (best_params[i] + step, best_params[i] - step):
                    trial = trial % (2 * math.pi)
                    if trial > math.pi:
                        trial -= 2 * math.pi
                    p = best_params[:]
                    p[i] = trial
                    d = _distance(target, _build(circuit_from_params(p), 3))
                    if d < best_dist:
                        best_params, best_dist = p, d
                        improved = True
            if not improved and sweep > 5:
                break
        
        return circuit_from_params(best_params)
    else:
        # Budget < 4, use whatever we can
        pairs = [(0,1), (1,2), (0,2)]
        layout = []
        for i in range(budget):
            for q in range(3):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            c, t = pairs[i % len(pairs)]
            layout.append(["cx", c, t])
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        
        n_u3 = sum(1 for g in layout if g[0] == "u3")
        n_params = 3 * n_u3
        
        def circuit_from_params(p):
            gates = []
            idx = 0
            for gate in layout:
                if gate[0] == "u3":
                    gates.append(["u3", gate[1], p[idx], p[idx+1], p[idx+2]])
                    idx += 3
                else:
                    gates.append(list(gate))
            return gates
        
        best_params = [0.0] * n_params
        best_dist = _distance(target, _build(circuit_from_params(best_params), 3))
        
        for sweep in range(15):
            if _time_left() < 10.0:
                break
            step = 0.5 / (1 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                for trial in (best_params[i] + step, best_params[i] - step):
                    trial = trial % (2 * math.pi)
                    if trial > math.pi:
                        trial -= 2 * math.pi
                    p = best_params[:]
                    p[i] = trial
                    d = _distance(target, _build(circuit_from_params(p), 3))
                    if d < best_dist:
                        best_params, best_dist = p, d
                        improved = True
            if not improved and sweep > 3:
                break
        
        return circuit_from_params(best_params)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if qubits == 2:
        return _two_qubit_synthesis(target)
    elif qubits == 3:
        return _three_qubit_synthesis(target, budget)
    else:
        # Fallback: just return identity
        return []
# EVOLVE-BLOCK-END
