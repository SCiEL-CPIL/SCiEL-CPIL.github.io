"""Rewritten program: exact quantum circuit synthesis via structured factorization."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
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


def _u3_from_unitary_2x2(U):
    """Extract theta, phi, lam from a 2x2 unitary U = [[a, b], [c, d]]
    matching the standard u3 convention:
    u3(theta, phi, lam) = [[cos(t/2), -exp(i*lam)*sin(t/2)],
                           [exp(i*phi)*sin(t/2), exp(i*(phi+lam))*cos(t/2)]]
    """
    a, b = U[0, 0], U[0, 1]
    c, d = U[1, 0], U[1, 1]
    
    if abs(a) < 1e-10:
        theta = math.pi
        if abs(b) > 1e-10:
            lam = -np.angle(b)
        else:
            lam = 0.0
        if abs(c) > 1e-10:
            phi = np.angle(c)
        else:
            phi = 0.0
        lam = lam % (2 * math.pi)
        phi = phi % (2 * math.pi)
        return theta, phi, lam
    if abs(c) < 1e-10 and abs(b) < 1e-10:
        theta = 0.0
        phi = 0.0
        lam = 0.0
        return theta, phi, lam
    
    cos_t2_sq = abs(a) ** 2
    sin_t2_sq = abs(c) ** 2
    total = cos_t2_sq + sin_t2_sq
    if total < 1e-10:
        return 0.0, 0.0, 0.0
    cos_t2_sq /= total
    sin_t2_sq /= total
    cos_t2 = math.sqrt(max(0, min(1, cos_t2_sq)))
    sin_t2 = math.sqrt(max(0, min(1, sin_t2_sq)))
    
    if cos_t2 < 1e-10:
        theta = math.pi
    elif sin_t2 < 1e-10:
        theta = 0.0
    else:
        theta = 2 * math.atan2(sin_t2, cos_t2)
    
    if sin_t2 > 1e-10:
        exp_i_lam = -b / sin_t2
        lam = np.angle(exp_i_lam)
        exp_i_phi = c / sin_t2
        phi = np.angle(exp_i_phi)
    else:
        lam = 0.0
        phi = 0.0
    
    lam = lam % (2 * math.pi)
    phi = phi % (2 * math.pi)
    theta = theta % (2 * math.pi)
    
    return theta, phi, lam


def _embed_single_gate(gate_params, qubit, qubits):
    """Create the full-space operator for a single-qubit gate."""
    theta, phi, lam = gate_params
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator


def _find_best_single_qubit(U, qubit, qubits, target_structure):
    """
    Find the best single-qubit gate G on 'qubit' such that
    G_emb @ U is as close as possible to 'target_structure'.
    target_structure is a unitary matrix.
    We minimize || G_emb @ U - target_structure ||_F.
    This is equivalent to maximizing Re(tr((G_emb @ U)^dag target_structure))
    = Re(tr(U^dag G_emb^dag target_structure)).
    Let M = U^dag target_structure. We want to maximize Re(tr(G_emb^dag M)).
    G_emb is I ... G ... I.
    This is a linear function of G.
    We can find the optimal G by projecting M onto the subspace of operators that are G on 'qubit' and I elsewhere.
    The optimal G is the 2x2 part of M that corresponds to 'qubit', projected onto unitaries.
    """
    n = 2 ** qubits
    # Extract the 2x2 "reduced" operator for qubit from M = U^dag @ target_structure
    # M acts on the full space.
    # We want to find G (2x2) such that G_emb is the best approximation.
    # The optimal G is obtained by averaging the 2x2 blocks of M over all other qubits.
    
    other_qubits = [q for q in range(qubits) if q != qubit]
    num_other = 2 ** len(other_qubits)
    bit_pos = qubits - 1 - qubit
    
    sum_blocks = np.zeros((2, 2), dtype=complex)
    for mask in range(num_other):
        row_base = 0
        col_base = 0
        for i, q in enumerate(other_qubits):
            bit_val = (mask >> i) & 1
            bit_pos_q = qubits - 1 - q
            row_base |= (bit_val << bit_pos_q)
            col_base |= (bit_val << bit_pos_q)
        
        row0 = row_base
        row1 = row_base | (1 << bit_pos)
        col0 = col_base
        col1 = col_base | (1 << bit_pos)
        
        block = np.array([[M[row0, col0], M[row0, col1]],
                          [M[row1, col0], M[row1, col1]]])
        sum_blocks += block
    
    avg_block = sum_blocks / num_other
    
    # Project avg_block onto unitary group using SVD
    W, S, Vdag = np.linalg.svd(avg_block)
    G = W @ Vdag
    
    # Extract u3 params
    theta, phi, lam = _u3_from_unitary_2x2(G)
    return (theta, phi, lam)


def synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    if np.allclose(target, np.eye(n), atol=1e-10):
        return []
    
    # Pre-define CNOT structures for 2 and 3 qubits
    # For 2 qubits, CNOTs are:
    # 0->1, 1->0, and their conjugates (which are themselves)
    # Actually, any CNOT is self-adjoint.
    # We also consider "identity" as a target structure if budget allows skipping CNOTs.
    
    if qubits == 2:
        # Decomposition: (A⊗B) CNOT (C⊗D) CNOT (E⊗F)
        # We will extract layers.
        # Layer 1: Find E, F such that (E⊗F)^dag U is close to CNOT (C⊗D) CNOT or just CNOT or I.
        # Since we have a budget, we can try to match against all possible CNOT sequences up to budget length.
        # However, the standard 3-CNOT decomposition is robust.
        # We can iteratively extract:
        # 1. Best single-qubit layer to make residual close to a CNOT.
        # 2. Apply CNOT.
        # 3. Best single-qubit layer.
        # 4. Apply CNOT.
        # 5. Best single-qubit layer.
        
        gates = []
        residual = target.copy()
        
        # Possible CNOTs: (0,1), (1,0)
        cnots = [(0, 1), (1, 0)]
        
        # We'll use a greedy approach: at each step, find the best single-qubit gate
        # that aligns the residual with the next CNOT in a sequence.
        # To be robust, we can try both CNOTs at each step and pick the one that
        # minimizes the residual error after the subsequent single-qubit layer.
        
        # Step 1: Extract single-qubit layer to prepare for first CNOT.
        # Try both CNOTs as targets.
        best_score = np.inf
        best_gates = []
        best_residual = None
        
        for c1 in cnots:
            # Target structure: CNOT(c1) @ (C⊗D) @ CNOT(c2) @ (E⊗F) ?
            # No, we extract from right to left.
            # U = L1 C1 L2 C2 L3
            # We want to find L3 (single-qubit layer) such that L3^dag U is close to C2 L2 C1 L1?
            # It's easier to go left to right in the extraction if we reverse the logic.
            # Let's stick to: U = G1 C1 G2 C2 G3.
            # We want to find G3.
            # We can try to match U against G1 C1 G2 C2 G3.
            # This is complex.
            
            # Alternative: Iterative peeling.
            # 1. Find best single-qubit gate G on qubit q that minimizes the "interaction" of U G^dag.
            # 2. Apply G.
            # 3. Repeat.
            
            pass
        
        # Let's use the standard 3-CNOT decomposition algorithm.
        # 1. Compute the "interaction" part.
        # 2. Extract single-qubit gates to bring it to canonical form.
        
        # Simpler robust method:
        # We know U can be written as (A⊗B) CNOT (C⊗D) CNOT (E⊗F).
        # We can determine E, F by finding the best product state approximation of U?
        # No, U is not a product state.
        
        # Let's use the iterative extraction with SVD projection.
        # We will extract 3 layers of single-qubit gates, interleaved with 2 CNOTs.
        # Wait, the structure is S C S C S. That's 3 single-qubit layers and 2 CNOTs.
        # Total CNOTs = 2? No, any 2-qubit unitary needs up to 3 CNOTs.
        # Structure: S C S C S C S.
        
        residual = target.copy()
        gates = []
        
        # We will extract 3 CNOTs.
        for i in range(3):
            # Find best single-qubit layer (G0, G1) such that (G0⊗G1)^dag residual is close to a CNOT.
            # We try both CNOTs.
            best_cnot = None
            best_g0 = None
            best_g1 = None
            min_dist = np.inf
            
            for c in cnots:
                # Target: CNOT(c)
                # We want to find G0, G1 such that (G0⊗G1) residual ≈ CNOT(c).
                # This is equivalent to residual ≈ (G0^dag⊗G1^dag) CNOT(c).
                # Let's find G0 first, assuming G1=I, then G1 assuming G0.
                # Iterate a few times.
                
                g0 = (0, 0, 0)
                g1 = (0, 0, 0)
                
                for _ in range(10):
                    # Find best G0 given G1
                    # Minimize || (G0⊗G1) residual - CNOT(c) ||
                    # This is equivalent to finding G0 that maximizes Re(tr(((G0⊗G1) residual)^dag CNOT(c)))
                    # = Re(tr(residual^dag (G0^dag⊗G1^dag) CNOT(c)))
                    # Let M = residual^dag @ (G1^dag on q1) @ CNOT(c)
                    # We want to find G0 that maximizes Re(tr(G0^dag M_q0))
                    
                    # Compute M for G0 extraction
                    # Apply G1^dag to residual
                    G1_op = _embed_single_gate(g1, 1, 2)
                    temp = G1_op.conj().T @ residual
                    # Target for G0: CNOT(c)
                    # We want G0 such that (G0⊗I) temp ≈ CNOT(c)
                    # This is not quite right.
                    # (G0⊗G1) residual = (G0⊗I) (I⊗G1) residual.
                    # Let R1 = (I⊗G1) residual.
                    # We want (G0⊗I) R1 ≈ CNOT(c).
                    # This means R1 ≈ (G0^dag⊗I) CNOT(c).
                    # So we want to find G0 such that R1 is close to (G0^dag⊗I) CNOT(c).
                    # This is equivalent to finding G0 such that (G0⊗I)^dag R1 is close to CNOT(c)^dag = CNOT(c).
                    # So we want to find G0 that minimizes || (G0⊗I)^dag R1 - CNOT(c) ||.
                    # This is the same as finding G0 that maximizes Re(tr(R1^dag (G0⊗I) CNOT(c))).
                    
                    # Use _find_best_single_qubit
                    # It finds G such that G_emb @ U is close to target_structure.
                    # Here U = R1, target = CNOT(c).
                    # We want G0 such that (G0⊗I) R1 ≈ CNOT(c).
                    # So call _find_best_single_qubit(R1, 0, 2, CNOT(c)).
                    
                    CNOT_c = np.eye(4, dtype=complex)
                    # Build CNOT(c)
                    for j in range(4):
                        cb = (j >> (2 - 1 - c[0])) & 1
                        if cb:
                            i = j ^ (1 << (2 - 1 - c[1]))
                        else:
                            i = j
                        CNOT_c[i, j] = 1.0
                    # Wait, CNOT matrix construction in _build is different.
                    # In _build, it's a permutation.
                    # Let's build the unitary CNOT properly.
                    CNOT_unitary = np.zeros((4, 4), dtype=complex)
                    for j in range(4):
                        cb = (j >> (2 - 1 - c[0])) & 1
                        if cb:
                            i = j ^ (1 << (2 - 1 - c[1]))
                        else:
                            i = j
                        CNOT_unitary[i, j] = 1.0
                    
                    g0 = _find_best_single_qubit(R1, 0, 2, CNOT_unitary)
                    
                    # Now find best G1 given G0
                    G0_op = _embed_single_gate(g0, 0, 2)
                    R0 = G0_op.conj().T @ residual # (G0⊗I)^dag residual
                    # We want (I⊗G1) R0 ≈ CNOT(c)
                    # So find G1 such that (I⊗G1) R0 ≈ CNOT(c)
                    g1 = _find_best_single_qubit(R0, 1, 2, CNOT_unitary)
                
                # Evaluate quality
                G0_final = _embed_single_gate(g0, 0, 2)
                G1_final = _embed_single_gate(g1, 1, 2)
                approx = (G0_final @ G1_final) @ residual
                dist = np.linalg.norm(approx - CNOT_unitary)
                
                if dist < min_dist:
                    min_dist = dist
                    best_cnot = c
                    best_g0 = g0
                    best_g1 = g1
                    
            # Apply the best gates
            gates.append(["u3", 0, best_g0[0], best_g0[1], best_g0[2]])
            gates.append(["u3", 1, best_g1[0], best_g1[1], best_g1[2]])
            
            # Apply CNOT
            # Update residual: residual = CNOT^dag @ (G1 G0)^dag @ residual ?
            # We extracted G0, G1 such that (G0⊗G1) residual ≈ CNOT.
            # So residual ≈ (G0⊗G1)^dag CNOT.
            # New residual = CNOT^dag @ (G0⊗G1) @ residual.
            # Wait, the gates are applied to the state.
            # If U = G1 G0 CNOT ...
            # We found G0, G1 such that G1 G0 residual ≈ CNOT.
            # So residual ≈ (G1 G0)^dag CNOT.
            # We want to remove G1 G0 and CNOT.
            # New residual = CNOT^dag @ (G1 G0) @ residual.
            
            G0_op = _embed_single_gate(best_g0, 0, 2)
            G1_op = _embed_single_gate(best_g1, 1, 2)
            CNOT_unitary = np.zeros((4, 4), dtype=complex)
            for j in range(4):
                cb = (j >> (2 - 1 - best_cnot[0])) & 1
                if cb:
                    i = j ^ (1 << (2 - 1 - best_cnot[1]))
                else:
                    i = j
                CNOT_unitary[i, j] = 1.0
            
            residual = CNOT_unitary.conj().T @ (G1_op @ G0_op) @ residual
            
            gates.append(["cx", best_cnot[0], best_cnot[1]])
            
        # Final single-qubit layer
        # Find G0, G1 such that (G0⊗G1) residual ≈ I
        best_g0 = None
        best_g1 = None
        min_dist = np.inf
        
        # Target is Identity
        I = np.eye(4, dtype=complex)
        
        g0 = (0, 0, 0)
        g1 = (0, 0, 0)
        for _ in range(10):
            G1_op = _embed_single_gate(g1, 1, 2)
            R1 = G1_op.conj().T @ residual
            g0 = _find_best_single_qubit(R1, 0, 2, I)
            
            G0_op = _embed_single_gate(g0, 0, 2)
            R0 = G0_op.conj().T @ residual
            g1 = _find_best_single_qubit(R0, 1, 2, I)
        
        gates.append(["u3", 0, g0[0], g0[1], g0[2]])
        gates.append(["u3", 1, g1[0], g1[1], g1[2]])
        
        # Reverse gates
        gates = list(reversed(gates))
        
        return gates

    elif qubits == 3:
        # Similar approach for 3 qubits.
        # Budget is likely 6 or 7 for Toffoli, higher for generic.
        # We will use a greedy CNOT selection.
        
        residual = target.copy()
        gates = []
        
        cnots = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
        
        for i in range(budget):
            # Find best single-qubit layer (G0, G1, G2) and best CNOT
            # such that (G0⊗G1⊗G2) residual ≈ CNOT(c)
            
            best_cnot = None
            best_gates = []
            min_dist = np.inf
            
            for c in cnots:
                # Build CNOT unitary
                CNOT_unitary = np.zeros((8, 8), dtype=complex)
                for j in range(8):
                    cb = (j >> (3 - 1 - c[0])) & 1
                    if cb:
                        i = j ^ (1 << (3 - 1 - c[1]))
                    else:
                        i = j
                    CNOT_unitary[i, j] = 1.0
                
                # Find best G0, G1, G2
                g0 = (0, 0, 0)
                g1 = (0, 0, 0)
                g2 = (0, 0, 0)
                
                for _ in range(10):
                    # Extract G2
                    G1_op = _embed_single_gate(g1, 1, 3)
                    G0_op = _embed_single_gate(g0, 0, 3)
                    R = (G0_op @ G1_op).conj().T @ residual
                    g2 = _find_best_single_qubit(R, 2, 3, CNOT_unitary)
                    
                    # Extract G1
                    G2_op = _embed_single_gate(g2, 2, 3)
                    G0_op = _embed_single_gate(g0, 0, 3)
                    R = (G0_op @ G2_op).conj().T @ residual
                    g1 = _find_best_single_qubit(R, 1, 3, CNOT_unitary)
                    
                    # Extract G0
                    G2_op = _embed_single_gate(g2, 2, 3)
                    G1_op = _embed_single_gate(g1, 1, 3)
                    R = (G1_op @ G2_op).conj().T @ residual
                    g0 = _find_best_single_qubit(R, 0, 3, CNOT_unitary)
                
                # Evaluate
                G0_op = _embed_single_gate(g0, 0, 3)
                G1_op = _embed_single_gate(g1, 1, 3)
                G2_op = _embed_single_gate(g2, 2, 3)
                approx = (G0_op @ G1_op @ G2_op) @ residual
                dist = np.linalg.norm(approx - CNOT_unitary)
                
                if dist < min_dist:
                    min_dist = dist
                    best_cnot = c
                    best_gates = [g0, g1, g2]
            
            # Apply best gates
            for q, g in enumerate(best_gates):
                gates.append(["u3", q, g[0], g[1], g[2]])
            
            # Apply CNOT
            G0_op = _embed_single_gate(best_gates[0], 0, 3)
            G1_op = _embed_single_gate(best_gates[1], 1, 3)
            G2_op = _embed_single_gate(best_gates[2], 2, 3)
            
            CNOT_unitary = np.zeros((8, 8), dtype=complex)
            for j in range(8):
                cb = (j >> (3 - 1 - best_cnot[0])) & 1
                if cb:
                    i = j ^ (1 << (3 - 1 - best_cnot[1]))
                else:
                    i = j
                CNOT_unitary[i, j] = 1.0
            
            residual = CNOT_unitary.conj().T @ (G0_op @ G1_op @ G2_op) @ residual
            gates.append(["cx", best_cnot[0], best_cnot[1]])
            
        # Final single-qubit layer
        I = np.eye(8, dtype=complex)
        g0 = (0, 0, 0)
        g1 = (0, 0, 0)
        g2 = (0, 0, 0)
        for _ in range(10):
            G1_op = _embed_single_gate(g1, 1, 3)
            G0_op = _embed_single_gate(g0, 0, 3)
            R = (G0_op @ G1_op).conj().T @ residual
            g2 = _find_best_single_qubit(R, 2, 3, I)
            
            G2_op = _embed_single_gate(g2, 2, 3)
            G0_op = _embed_single_gate(g0, 0, 3)
            R = (G0_op @ G2_op).conj().T @ residual
            g1 = _find_best_single_qubit(R, 1, 3, I)
            
            G2_op = _embed_single_gate(g2, 2, 3)
            G1_op = _embed_single_gate(g1, 1, 3)
            R = (G1_op @ G2_op).conj().T @ residual
            g0 = _find_best_single_qubit(R, 0, 3, I)
        
        gates.append(["u3", 0, g0[0], g0[1], g0[2]])
        gates.append(["u3", 1, g1[0], g1[1], g1[2]])
        gates.append(["u3", 2, g2[0], g2[1], g2[2]])
        
        gates = list(reversed(gates))
        return gates

# EVOLVE-BLOCK-END
