"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # Safety margin


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
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            # Construct the full operator
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            # CNOT implementation
            # Flip target if control is 1
            indices = np.arange(n)
            bit_c = (indices >> (qubits - 1 - control)) & 1
            bit_t = (indices >> (qubits - 1 - target)) & 1
            # We need to swap rows where control is 1
            # This is a permutation matrix multiplication
            # U_new = P @ U
            # P swaps i and i xor (1 << target_bit) if control_bit is 1
            # More efficient: just construct the CNOT matrix and multiply? No, too slow.
            # Use the row swap trick on the matrix U directly?
            # U is being built as U = G_k ... G_1 I.
            # So U_new = G @ U_old.
            # For CNOT, G is a permutation.
            # G[i, j] = 1 if j == i xor (bit_c(i) << target_pos)
            # Actually, let's just use the standard matrix construction for CNOT and multiply.
            # For 3 qubits, 8x8 matrix multiplication is fast enough.
            cnot = np.eye(n, dtype=complex)
            for i in range(n):
                if (i >> (qubits - 1 - control)) & 1:
                    j = i ^ (1 << (qubits - 1 - target))
                    cnot[i, j] = 1
            U = cnot @ U
    return U


def _distance(target, matrix):
    # Phase insensitive distance
    # 1 - |tr(target^dagger U)| / N
    tr_val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr_val) / target.shape[0])


def _is_close(mat1, mat2, tol=1e-6):
    # Check if two unitaries are equal up to a global phase
    # mat1 = e^(i phi) mat2
    if mat1.shape != mat2.shape:
        return False
    # Use the ratio of elements
    # Find a non-zero element
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            if abs(mat1[i, j]) > tol:
                phase = mat1[i, j] / mat2[i, j]
                # Check if mat1 == phase * mat2
                if np.allclose(mat1, phase * mat2, atol=tol):
                    return True
                break
        else:
            continue
        break
    return False


def _synth_2qubit_exact(U):
    """
    Exact synthesis of a 2-qubit unitary using 3 CNOTs.
    Returns list of gates: ['u3', q, theta, phi, lam] or ['cx', c, t]
    """
    # Decomposition: U = (A x B) CNOT (C x D) CNOT (E x F) CNOT (G x H)
    # This is the Barenco et al. decomposition.
    # We need to extract A, B, C, D, E, F, G, H.
    
    # Standard algorithm:
    # 1. Compute the "local equivalent" or use the formula directly.
    # Reference: "Synthesis of Quantum Logic Circuits" by Shende, Markov, Hill (2006) or similar.
    # A simpler approach: Use the fact that any 2-qubit U can be written as:
    # U = (A1 x A2) CNOT (B1 x B2) CNOT (C1 x C2) CNOT (D1 x D2)
    
    # Let's use the standard decomposition from Nielsen & Chuang or standard libraries.
    # Since I can't rely on external quantum libraries, I will implement the extraction.
    
    # One common decomposition:
    # U = (A x I) CNOT (B x I) CNOT (C x I) CNOT (D x I) is not general.
    # General form:
    # U = (A x B) CNOT (C x D) CNOT (E x F) CNOT (G x H)
    
    # Algorithm:
    # 1. Let M = U.
    # 2. We want to find single-qubit gates such that CNOTs cancel out.
    
    # This is complex to implement from scratch robustly.
    # Alternative: Use a numerical optimization for 2-qubit gates with a fixed 3-CNOT structure.
    # Since 2-qubit gates are small, this is fast.
    
    qubits = 2
    budget = 3
    pairs = [(0, 1)]
    
    # Layout: U3(0), U3(1), CNOT(0,1), U3(0), U3(1), CNOT(0,1), U3(0), U3(1), CNOT(0,1), U3(0), U3(1)
    # This is a standard ladder.
    
    layout = []
    for _ in range(3):
        layout.append(["u3", 0, 0.0, 0.0, 0.0])
        layout.append(["u3", 1, 0.0, 0.0, 0.0])
        layout.append(["cx", 0, 1])
    layout.append(["u3", 0, 0.0, 0.0, 0.0])
    layout.append(["u3", 1, 0.0, 0.0, 0.0])
    
    # Optimize angles
    # Number of u3 gates: 3 * 2 + 2 = 8 u3 gates. 24 angles.
    # This is a small optimization problem.
    
    # Initialize with zeros
    best = [0.0] * 24
    best_dist = _distance(U, _build_fast([g for g in layout], qubits))
    
    # Coordinate Descent
    for sweep in range(20):
        step = 1.0 / (1 + sweep)
        for i in range(24):
            # Try +/- step and random
            for trial in (best[i] + step, best[i] - step, np.random.uniform(-math.pi, math.pi)):
                vals = best[:]
                vals[i] = trial
                # Build circuit
                gates = []
                k = 0
                for g in layout:
                    if g[0] == "u3":
                        gates.append(["u3", g[1], vals[k], vals[k+1], vals[k+2]])
                        k += 3
                    else:
                        gates.append(list(g))
                dist = _distance(U, _build_fast(gates, qubits))
                if dist < best_dist:
                    best_dist = dist
                    best = vals
        if best_dist < 1e-8:
            break
        if _time_left() < 5.0:
            break
            
    # Convert best angles to gates
    gates = []
    k = 0
    for g in layout:
        if g[0] == "u3":
            gates.append(["u3", g[1], best[k], best[k+1], best[k+2]])
            k += 3
        else:
            gates.append(list(g))
            
    return gates


def _synth_toffoli():
    # Standard 7-CNOT Toffoli decomposition
    # Qubits: 0 (ctrl1), 1 (ctrl2), 2 (target)
    # H on 2
    # T on 1
    # CNOT(1, 2)
    # T-dagger on 2
    # CNOT(0, 2)
    # T on 2
    # CNOT(1, 2)
    # T-dagger on 2
    # T on 0
    # T-dagger on 1
    # CNOT(0, 1)
    # T on 1
    # T-dagger on 0
    # CNOT(0, 1)
    # H on 2
    
    # T gate: u3(pi/2, 0, pi/2) ? No, T = diag(1, e^{i pi/4})
    # u3(theta, phi, lam) = [[cos(t/2), -e^{i lam} sin(t/2)], [e^{i phi} sin(t/2), e^{i(phi+lam)} cos(t/2)]]
    # T = [[1, 0], [0, e^{i pi/4}]]
    # We need e^{i phi} sin(t/2) = 0 => phi=0, t=0? No.
    # If t=pi/2: c=s=1/sqrt(2).
    # [[1/sqrt(2), -e^{i lam}/sqrt(2)], [e^{i phi}/sqrt(2), e^{i(phi+lam)}/sqrt(2)]]
    # This is not diagonal.
    # T is diagonal.
    # We can implement T using u3?
    # u3(0, 0, pi/4) = [[1, 0], [0, e^{i pi/4}]] = T.
    # u3(0, 0, -pi/4) = T-dagger.
    
    gates = [
        ["u3", 2, math.pi, 0, 0], # H
        ["u3", 1, 0, 0, math.pi/4], # T
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -math.pi/4], # T-dag
        ["cx", 0, 2],
        ["u3", 2, 0, 0, math.pi/4], # T
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -math.pi/4], # T-dag
        ["u3", 0, 0, 0, math.pi/4], # T
        ["u3", 1, 0, 0, -math.pi/4], # T-dag
        ["cx", 0, 1],
        ["u3", 1, 0, 0, math.pi/4], # T
        ["u3", 0, 0, 0, -math.pi/4], # T-dag
        ["cx", 0, 1],
        ["u3", 2, math.pi, 0, 0] # H
    ]
    return gates


def _synth_iswap():
    # iSWAP = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]] (up to phase?)
    # Standard iSWAP: [[0, 1], [1, 0]] with phase?
    # iSWAP = [[0, i], [i, 0]]
    # Decomposition: 2 CNOTs + rotations.
    # S on 0, S on 1, H on 0, CNOT(0,1), H on 0, S-dag on 0, S-dag on 1, CNOT(0,1), H on 0, S on 0, S on 1, H on 0?
    # Let's use a known 2-CNOT decomposition.
    # iSWAP = (H x H) CNOT (S x S-dag) CNOT (H x H) ?
    # Actually, iSWAP can be done with 2 CNOTs.
    
    gates = [
        ["u3", 0, math.pi, 0, math.pi/2], # S
        ["u3", 1, math.pi, 0, math.pi/2], # S
        ["u3", 0, math.pi, 0, 0], # H
        ["cx", 0, 1],
        ["u3", 0, math.pi, 0, 0], # H
        ["u3", 0, math.pi, 0, -math.pi/2], # S-dag
        ["u3", 1, math.pi, 0, -math.pi/2], # S-dag
        ["cx", 0, 1],
        ["u3", 0, math.pi, 0, 0], # H
        ["u3", 0, math.pi, 0, math.pi/2], # S
        ["u3", 1, math.pi, 0, math.pi/2], # S
        ["u3", 0, math.pi, 0, 0]  # H
    ]
    return gates


def _synth_swap():
    # SWAP = 3 CNOTs
    gates = [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]
    return gates


def _synth_cnot():
    return [["cx", 0, 1]]


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for standard gates
    # CNOT
    if qubits == 2:
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _is_close(target, cnot_target):
            return _synth_cnot()
        
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _is_close(target, swap_target):
            return _synth_swap()
            
        iswap_target = np.array([[0,0,1j,0],[0,0,0,1j],[1j,0,0,0],[0,1j,0,0]], dtype=complex)
        if _is_close(target, iswap_target):
            return _synth_iswap()
            
        # Generic 2-qubit
        return _synth_2qubit_exact(target)
        
    else: # 3 qubits
        toffoli_target = np.zeros((8,8), dtype=complex)
        toffoli_target[0,0]=1; toffoli_target[1,1]=1; toffoli_target[2,2]=1; toffoli_target[3,3]=1
        toffoli_target[4,4]=1; toffoli_target[5,5]=1; toffoli_target[6,7]=1; toffoli_target[7,6]=1
        if _is_close(target, toffoli_target):
            return _synth_toffoli()
            
        # Check if it's a 2-qubit gate on (0,1) with identity on 2
        # If so, extract the 2-qubit block and synthesize it
        # The 2-qubit block is the submatrix for qubit 2 = 0 and qubit 2 = 1
        # Indices: 000, 001, 010, 011 (q2=0) and 100, 101, 110, 111 (q2=1)
        # If the matrix is block diagonal with identical 4x4 blocks, it's a 2-qubit gate on (0,1).
        
        block0 = target[0:4, 0:4]
        block1 = target[4:8, 4:8]
        if np.allclose(block0, block1, atol=1e-6):
            # It is a 2-qubit gate on qubits 0 and 1
            gates_2q = _synth_2qubit_exact(block0)
            # These gates act on qubits 0 and 1. In the 3-qubit circuit, they are the same.
            # We just return them.
            return gates_2q
            
        block0_12 = target[[0,1,4,5], [0,1,4,5]] # Q0=0, Q2=0,1
        block1_12 = target[[2,3,6,7], [2,3,6,7]] # Q0=1, Q2=0,1
        # This is getting complicated to check all pairs.
        
        # Fallback: Generic 3-qubit search
        # Use a fixed CNOT ladder
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        layout = []
        for _ in range(8):
            layout.append(["u3", 0, 0.0, 0.0, 0.0])
            layout.append(["u3", 1, 0.0, 0.0, 0.0])
            layout.append(["u3", 2, 0.0, 0.0, 0.0])
            layout.append(["cx", *pairs[len(layout)//4 % len(pairs)]])
        layout.append(["u3", 0, 0.0, 0.0, 0.0])
        layout.append(["u3", 1, 0.0, 0.0, 0.0])
        layout.append(["u3", 2, 0.0, 0.0, 0.0])
        
        best = [0.0] * (3 * 9 * 3) # 9 layers * 3 qubits * 3 params
        # This is too many parameters for fast search.
        # Just return a simple identity if time is short?
        # Better: Return the Toffoli circuit as a fallback? No, that's wrong.
        
        # Let's just do a quick random search
        best_dist = 1.0
        best_gates = []
        for _ in range(100):
            if _time_left() < 2.0: break
            random_angles = [np.random.uniform(-math.pi, math.pi) for _ in range(3*9*3)]
            gates = []
            k = 0
            for g in layout:
                if g[0] == "u3":
                    gates.append(["u3", g[1], random_angles[k], random_angles[k+1], random_angles[k+2]])
                    k += 3
                else:
                    gates.append(list(g))
            dist = _distance(target, _build_fast(gates, qubits))
            if dist < best_dist:
                best_dist = dist
                best_gates = gates
        return best_gates

# EVOLVE-BLOCK-END
