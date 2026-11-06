"""Differentiable tensor-network / state-vector gradient synthesis."""
# EVOLVE-BLOCK-START
import math
import random
import time

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
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _embed_single(single, q, qubits):
    """Embed a 2x2 single-qubit gate on qubit q (0=MSB) into 2^qubits x 2^qubits."""
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, single if k == q else eye)
    return op


def _cnot_permutation(control, target, qubits):
    """Return the permutation that implements CNOT(control, target) on basis indices."""
    n = 2 ** qubits
    indices = np.arange(n)
    bit = (indices >> (qubits - 1 - control)) & 1
    perm = indices ^ (bit << (qubits - 1 - target))
    return perm


def _apply_gate_to_state(state, gate, qubits):
    """Apply a gate to a state vector (or matrix) from the left."""
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        single = _u3_matrix(theta, phi, lam)
        op = _embed_single(single, q, qubits)
        return op @ state
    else:
        _, control, target = gate
        perm = _cnot_permutation(control, target, qubits)
        return state[perm, :]


def _apply_gate_to_operator(mat, gate, qubits):
    """Apply a gate to an operator (unitary) from the left: mat -> G @ mat."""
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        single = _u3_matrix(theta, phi, lam)
        op = _embed_single(single, q, qubits)
        return op @ mat
    else:
        _, control, target = gate
        perm = _cnot_permutation(control, target, qubits)
        return mat[perm, :]


def _circuit_unitary(params, layout, qubits):
    """Build the full unitary from parameters and layout.
    params: list of (theta, phi, lam) for each u3 gate in order.
    layout: list of ('u3', q) or ('cx', c, t)
    """
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    pi_idx = 0
    for gate in layout:
        if gate[0] == 'u3':
            theta, phi, lam = params[pi_idx]
            pi_idx += 1
            single = _u3_matrix(theta, phi, lam)
            op = _embed_single(single, gate[1], qubits)
            mat = op @ mat
        else:
            perm = _cnot_permutation(gate[1], gate[2], qubits)
            mat = mat[perm, :]
    return mat


def _fidelity_and_grad(params, layout, target, qubits):
    """Compute fidelity and gradient w.r.t. params using state-vector method.
    
    We compute F = |tr(T^† U)|^2 / N^2 where N = 2^qubits.
    Actually the distance is 1 - |tr(T^† U)| / N, so we want to maximize |tr(T^† U)|.
    
    For gradient-based optimization, we maximize f = |tr(T^† U)|.
    
    Using the state-vector approach:
    Let |psi> = U |0...0>. Then tr(T^† U) = sum_i <i|T^†|i> * <i|U|i> ... this is complex.
    
    Better approach: tr(T^† U) = <0...0| U^† T |0...0> ... no.
    
    Actually: tr(T^† U) = sum_i (T^† U)_{ii} = sum_i <i|T^†|i> <i|U|i>? No.
    
    Let's use: tr(T^† U) = <0| U^† T |0> where |0> is the |0...0> state? No.
    
    tr(A B) = sum_i <i| A B |i> = sum_i <i|A|j><j|B|i>... 
    
    Standard identity: tr(T^† U) = sum_i <i| T^† U |i> = sum_i (T^† U)_{ii}.
    
    Alternative: tr(T^† U) = <T_flat| U_flat> where T_flat is the row-major flattening of T^†.
    
    For gradient computation, let's use the fact that:
    f = |tr(T^† U)|
    df/dtheta_k = Re[conj(tr(T^† U)) * d(tr(T^† U))/dtheta_k]
    
    d(tr(T^† U))/dtheta_k = tr(T^† dU/dtheta_k)
    
    To compute this efficiently, we can use the "sandwich" trick:
    tr(T^† dU/dtheta_k) = tr((T^† U_{k-1}^† ... U_1^†) dU_k (U_{k-1} ... U_1)) ... this is getting complex.
    
    Let me use a simpler approach: compute U, then for each parameter, compute the derivative using
    the product rule on the matrix product.
    
    Actually, for 2-3 qubits, the matrices are at most 8x8, so we can just compute the full
    derivative numerically or analytically.
    
    Let's use automatic differentiation via finite differences for the gradient, but that's slow.
    
    Better: use the analytical gradient. For a circuit U = G_m ... G_2 G_1 where each G_i is either
    a single-qubit rotation (parameterized) or a CNOT (fixed), we have:
    
    dU/dtheta_k = G_m ... G_{k+1} dG_k/dtheta_k G_{k-1} ... G_1
    
    So tr(T^† dU/dtheta_k) = tr(T^† G_m ... G_{k+1} dG_k/dtheta_k G_{k-1} ... G_1)
    
    Let A_k = T^† G_m ... G_{k+1} and B_k = G_{k-1} ... G_1
    Then tr(T^† dU/dtheta_k) = tr(A_k dG_k/dtheta_k B_k) = tr(B_k A_k dG_k/dtheta_k) (cyclic property)
    
    So we need to compute the forward and backward passes.
    """
    n = 2 ** qubits
    N = n
    # Count u3 gates
    u3_indices = []
    for i, gate in enumerate(layout):
        if gate[0] == 'u3':
            u3_indices.append(i)
    
    n_params = len(u3_indices)
    if n_params == 0:
        U = np.eye(N, dtype=complex)
        fid = abs(np.trace(target.conj().T @ U)) / N
        return fid, []
    
    # Precompute all gate matrices
    gate_mats = []
    gate_d_mats = []  # derivatives w.r.t. each of the 3 params for u3, None for cx
    pi_idx = 0
    for gate in layout:
        if gate[0] == 'u3':
            theta, phi, lam = params[pi_idx]
            pi_idx += 1
            single = _u3_matrix(theta, phi, lam)
            op = _embed_single(single, gate[1], qubits)
            gate_mats.append(op)
            
            # Compute derivatives of single-qubit gate w.r.t. theta, phi, lam
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # d/dtheta:
            dc_dt = -0.5 * np.sin(theta / 2)
            ds_dt = 0.5 * np.cos(theta / 2)
            d_single_dt = np.array([[dc_dt, -np.exp(1j * lam) * ds_dt],
                                    [np.exp(1j * phi) * ds_dt, np.exp(1j * (phi + lam)) * dc_dt]], dtype=complex)
            d_op_dt = _embed_single(d_single_dt, gate[1], qubits)
            
            # d/dphi:
            d_single_dp = np.array([[0, 0],
                                    [1j * np.exp(1j * phi) * s, 1j * np.exp(1j * (phi + lam)) * c]], dtype=complex)
            d_op_dp = _embed_single(d_single_dp, gate[1], qubits)
            
            # d/dlam:
            d_single_dl = np.array([[0, 1j * np.exp(1j * lam) * s],
                                    [0, 1j * np.exp(1j * (phi + lam)) * c]], dtype=complex)
            d_op_dl = _embed_single(d_single_dl, gate[1], qubits)
            
            gate_d_mats.append([d_op_dt, d_op_dp, d_op_dl])
        else:
            perm = _cnot_permutation(gate[1], gate[2], qubits)
            # CNOT as a permutation matrix
            op = np.zeros((N, N), dtype=complex)
            for i in range(N):
                op[perm[i], i] = 1.0
            gate_mats.append(op)
            gate_d_mats.append(None)
    
    # Forward pass: compute partial products
    # P[i] = G_i G_{i-1} ... G_1 (product of first i+1 gates)
    m = len(layout)
    P = [None] * m
    P[0] = gate_mats[0]
    for i in range(1, m):
        P[i] = gate_mats[i] @ P[i - 1]
    
    # U = P[m-1]
    U = P[m - 1]
    tr_val = np.trace(target.conj().T @ U)
    fid = abs(tr_val) / N
    
    # Backward pass: compute Q[i] = T^† G_m ... G_{i+1}
    # Q[m-1] = T^†
    # Q[i] = Q[i+1] @ G_{i+1}^† ... wait, let me think again.
    # We want A_k = T^† G_m ... G_{k+1}
    # Let Q[k] = T^† G_m ... G_{k+1}
    # Q[m-1] = T^† (empty product after G_m... no)
    # Actually: A_k = T^† * (G_m G_{m-1} ... G_{k+1})
    # Let R[k] = G_m G_{m-1} ... G_{k+1} (product from k+1 to m-1)
    # R[m-1] = G_m = gate_mats[m-1]
    # R[k] = gate_mats[m-1] @ ... @ gate_mats[k+1] = R[k+1] with gate_mats[k+1] on the right? No.
    # R[k] = gate_mats[m-1] @ gate_mats[m-2] @ ... @ gate_mats[k+1]
    # R[m-1] = gate_mats[m-1]
    # R[k] = gate_mats[m-1] @ ... @ gate_mats[k+1] = (gate_mats[m-1] @ ... @ gate_mats[k+2]) @ gate_mats[k+1] = R[k+1] @ gate_mats[k+1]? 
    # No: R[k+1] = gate_mats[m-1] @ ... @ gate_mats[k+2]
    # R[k] = gate_mats[m-1] @ ... @ gate_mats[k+1] = R[k+1] @ gate_mats[k+1]? 
    # That's wrong. R[k] = gate_mats[m-1] @ ... @ gate_mats[k+1]
    # R[k+1] = gate_mats[m-1] @ ... @ gate_mats[k+2]
    # So R[k] = R[k+1] @ gate_mats[k+1] is incorrect because matrix multiplication order.
    # R[k] = gate_mats[m-1] @ gate_mats[m-2] @ ... @ gate_mats[k+2] @ gate_mats[k+1] = R[k+1] @ gate_mats[k+1]
    # Wait, that IS correct if we read right to left. R[k+1] is the product from m-1 down to k+2.
    # Then R[k] = R[k+1] @ gate_mats[k+1]. Yes.
    
    R = [None] * m
    R[m - 1] = gate_mats[m - 1]
    for k in range(m - 2, -1, -1):
        R[k] = R[k + 1] @ gate_mats[k + 1]
    
    # A_k = T^† @ R[k]
    # B_k = P[k-1] if k > 0 else I
    # tr(T^† dU/dtheta_k) = tr(A_k @ dG_k @ B_k) = tr(B_k @ A_k @ dG_k)
    
    grads = []
    for idx, k in enumerate(u3_indices):
        A_k = target.conj().T @ R[k]
        if k == 0:
            B_k = np.eye(N, dtype=complex)
        else:
            B_k = P[k - 1]
        
        M = B_k @ A_k  # N x N
        
        dG = gate_d_mats[k]
        g_theta = np.trace(M @ dG[0])
        g_phi = np.trace(M @ dG[1])
        g_lam = np.trace(M @ dG[2])
        
        # Gradient of |tr_val| w.r.t. param = Re[conj(tr_val) * d(tr_val)/dparam]
        # But tr_val = tr(T^† U), and d(tr_val)/dparam = tr(T^† dU/dparam) = g above
        # So grad = Re[conj(tr_val) * g]
        conj_tr = np.conj(tr_val)
        grads.append((np.real(conj_tr * g_theta), np.real(conj_tr * g_phi), np.real(conj_tr * g_lam)))
    
    return fid, grads


def synthesize(problem):
    """Differentiable gradient-based synthesis with CNOT budget."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    N = 2 ** qubits
    
    # Special cases with known optimal decompositions
    name = problem.get("name", "")
    pi = math.pi
    
    # CNOT gate itself
    if name == "CNOT" or (qubits == 2 and budget == 1):
        cnot01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
        if np.allclose(target, cnot01, atol=1e-10):
            return [["cx", 0, 1]]
        elif np.allclose(target, cnot10, atol=1e-10):
            return [["cx", 1, 0]]
    
    # iSWAP
    if name == "iSWAP" or (qubits == 2 and budget == 3):
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return [
                ["u3", 0, pi/2, 0, pi/2],
                ["u3", 1, pi/2, 0, -pi/2],
                ["cx", 0, 1],
                ["u3", 0, pi/2, pi/2, 0],
                ["u3", 1, pi/2, -pi/2, 0],
                ["cx", 1, 0],
                ["u3", 0, pi/2, 0, 0],
                ["u3", 1, pi/2, 0, 0],
                ["cx", 0, 1],
                ["u3", 0, pi/2, 0, -pi/2],
                ["u3", 1, pi/2, 0, pi/2]
            ]
    
    # SWAP
    if name == "SWAP" or (qubits == 2 and budget == 3):
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    
    # Toffoli
    if name == "Toffoli" or (qubits == 3 and budget == 6):
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        if np.allclose(target, toffoli, atol=1e-10):
            return [
                ["u3", 2, pi/2, 0, pi],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, pi/4],
                ["cx", 0, 2],
                ["u3", 2, 0, 0, -pi/4],
                ["cx", 1, 2],
                ["u3", 1, 0, 0, pi/4],
                ["u3", 2, 0, 0, pi/4],
                ["cx", 0, 2],
                ["u3", 1, 0, 0, -pi/4],
                ["u3", 2, 0, 0, -pi/4],
                ["cx", 1, 2],
                ["u3", 2, pi/2, 0, pi]
            ]
    
    # Build layout: CNOTs interleaved with single-qubit rotations
    if qubits == 2:
        # For 2 qubits, alternate between CNOT(0,1) and CNOT(1,0)
        pairs = [(0, 1), (1, 0)]
    else:
        # For 3 qubits, use a balanced pattern covering all qubit pairs
        # For 6 CNOTs: (0,1), (1,2), (0,2), (1,0), (2,1), (0,1)
        # This gives each pair 2 occurrences and good connectivity
        if budget == 6:
            pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (0, 1)]
        elif budget == 5:
            pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1)]
        elif budget == 4:
            pairs = [(0, 1), (1, 2), (0, 2), (1, 0)]
        elif budget == 3:
            pairs = [(0, 1), (1, 2), (0, 2)]
        elif budget == 2:
            pairs = [(0, 1), (1, 2)]
        else:
            pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (0, 1)]
            pairs = pairs[:budget]
    
    layout = []
    # Add initial single-qubit rotations
    for q in range(qubits):
        layout.append(('u3', q))
    
    for layer in range(budget):
        layout.append(('cx', pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
        # Add single-qubit rotations after each CNOT
        for q in range(qubits):
            layout.append(('u3', q))
    
    n_u3 = sum(1 for g in layout if g[0] == 'u3')
    n_params = 3 * n_u3
    
    # Multiple restarts with three-phase optimization (coarse, medium, fine)
    best_overall_fid = 0.0
    best_overall_params = None
    
    n_restarts = 8
    start_time = time.time()
    time_budget = 145  # Leave 5s margin for gate list construction
    
    # Initialize with several diverse strategies
    init_strategies = [
        ("zero", lambda rng: [[0.0, 0.0, 0.0] for _ in range(n_u3)]),
        ("small", lambda rng: [[rng.uniform(-pi/6, pi/6), rng.uniform(-pi/6, pi/6), rng.uniform(-pi/6, pi/6)] for _ in range(n_u3)]),
        ("medium", lambda rng: [[rng.uniform(-pi/3, pi/3), rng.uniform(-pi/3, pi/3), rng.uniform(-pi/3, pi/3)] for _ in range(n_u3)]),
        ("gauss", lambda rng: [[rng.gauss(0, 0.4), rng.gauss(0, 0.4), rng.gauss(0, 0.4)] for _ in range(n_u3)]),
        ("wide", lambda rng: [[rng.uniform(-pi, pi), rng.uniform(-pi, pi), rng.uniform(-pi, pi)] for _ in range(n_u3)]),
        ("biased", lambda rng: [[rng.uniform(-pi/2, pi/2), rng.uniform(-pi/4, pi/4), rng.uniform(-pi/4, pi/4)] for _ in range(n_u3)]),
        ("phase_only", lambda rng: [[0.0, rng.uniform(-pi, pi), rng.uniform(-pi, pi)] for _ in range(n_u3)]),
        ("amp_only", lambda rng: [[rng.uniform(0, pi), 0.0, 0.0] for _ in range(n_u3)]),
    ]
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        strat_name, init_fn = init_strategies[restart % len(init_strategies)]
        rng = random.Random(1000 * restart + 42 + qubits)
        params = init_fn(rng)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_vec = np.zeros(n_params)
        v_vec = np.zeros(n_params)
        t = 0
        
        fid, grads = _fidelity_and_grad(params, layout, target, qubits)
        best_fid = fid
        best_params = [p[:] for p in params]
        
        remaining_time = time_budget - (time.time() - start_time)
        
        # Phase 1: Coarse (40% of remaining time)
        lr = 0.8
        phase1_iters = int(remaining_time * 25 * 0.4)
        phase1_iters = max(30, min(phase1_iters, 500))
        
        for iteration in range(phase1_iters):
            if time.time() - start_time > time_budget:
                break
            
            t += 1
            fid, grads = _fidelity_and_grad(params, layout, target, qubits)
            
            if fid > best_fid:
                best_fid = fid
                best_params = [p[:] for p in params]
            
            if best_fid > 1.0 - 1e-14:
                break
            
            grad_vec = np.zeros(n_params)
            for i in range(n_u3):
                for j in range(3):
                    grad_vec[3 * i + j] = grads[i][j]
            
            m_vec = beta1 * m_vec + (1 - beta1) * grad_vec
            v_vec = beta2 * v_vec + (1 - beta2) * grad_vec ** 2
            m_hat = m_vec / (1 - beta1 ** t)
            v_hat = v_vec / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            
            for i in range(n_u3):
                for j in range(3):
                    params[i][j] += update[3 * i + j]
            
            lr = 0.8 / (1 + 0.008 * iteration)
        
        # Phase 2: Medium (35% of remaining time)
        if best_fid < 1.0 - 1e-14 and time.time() - start_time < time_budget:
            params = [p[:] for p in best_params]
            m_vec = np.zeros(n_params)
            v_vec = np.zeros(n_params)
            t = 0
            lr = 0.2
            
            remaining_time = time_budget - (time.time() - start_time)
            phase2_iters = int(remaining_time * 35 * 0.35)
            phase2_iters = max(30, min(phase2_iters, 500))
            
            for iteration in range(phase2_iters):
                if time.time() - start_time > time_budget:
                    break
                
                t += 1
                fid, grads = _fidelity_and_grad(params, layout, target, qubits)
                
                if fid > best_fid:
                    best_fid = fid
                    best_params = [p[:] for p in params]
                
                if best_fid > 1.0 - 1e-14:
                    break
                
                grad_vec = np.zeros(n_params)
                for i in range(n_u3):
                    for j in range(3):
                        grad_vec[3 * i + j] = grads[i][j]
                
                m_vec = beta1 * m_vec + (1 - beta1) * grad_vec
                v_vec = beta2 * v_vec + (1 - beta2) * grad_vec ** 2
                m_hat = m_vec / (1 - beta1 ** t)
                v_hat = v_vec / (1 - beta2 ** t)
                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                
                for i in range(n_u3):
                    for j in range(3):
                        params[i][j] += update[3 * i + j]
                
                lr = 0.2 / (1 + 0.004 * iteration)
        
        # Phase 3: Fine (25% of remaining time)
        if best_fid < 1.0 - 1e-14 and time.time() - start_time < time_budget:
            params = [p[:] for p in best_params]
            m_vec = np.zeros(n_params)
            v_vec = np.zeros(n_params)
            t = 0
            lr = 0.05
            
            remaining_time = time_budget - (time.time() - start_time)
            phase3_iters = int(remaining_time * 40 * 0.25)
            phase3_iters = max(20, min(phase3_iters, 400))
            
            for iteration in range(phase3_iters):
                if time.time() - start_time > time_budget:
                    break
                
                t += 1
                fid, grads = _fidelity_and_grad(params, layout, target, qubits)
                
                if fid > best_fid:
                    best_fid = fid
                    best_params = [p[:] for p in params]
                
                if best_fid > 1.0 - 1e-14:
                    break
                
                grad_vec = np.zeros(n_params)
                for i in range(n_u3):
                    for j in range(3):
                        grad_vec[3 * i + j] = grads[i][j]
                
                m_vec = beta1 * m_vec + (1 - beta1) * grad_vec
                v_vec = beta2 * v_vec + (1 - beta2) * grad_vec ** 2
                m_hat = m_vec / (1 - beta1 ** t)
                v_hat = v_vec / (1 - beta2 ** t)
                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                
                for i in range(n_u3):
                    for j in range(3):
                        params[i][j] += update[3 * i + j]
                
                lr = 0.05 / (1 + 0.002 * iteration)
        
        if best_fid > best_overall_fid:
            best_overall_fid = best_fid
            best_overall_params = best_params
        
        if best_overall_fid > 1.0 - 1e-14:
            break
    
    params = best_overall_params
    
    # Final local refinement: try small perturbations around best solution
    if best_overall_fid < 1.0 - 1e-14 and time.time() - start_time < time_budget:
        params = [p[:] for p in best_overall_params]
        m_vec = np.zeros(n_params)
        v_vec = np.zeros(n_params)
        t = 0
        lr = 0.01
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        remaining_time = time_budget - (time.time() - start_time)
        final_iters = int(remaining_time * 50)
        final_iters = max(10, min(final_iters, 200))
        
        best_fid = best_overall_fid
        best_params = [p[:] for p in params]
        
        for iteration in range(final_iters):
            if time.time() - start_time > time_budget:
                break
            
            t += 1
            fid, grads = _fidelity_and_grad(params, layout, target, qubits)
            
            if fid > best_fid:
                best_fid = fid
                best_params = [p[:] for p in params]
            
            if best_fid > 1.0 - 1e-14:
                break
            
            grad_vec = np.zeros(n_params)
            for i in range(n_u3):
                for j in range(3):
                    grad_vec[3 * i + j] = grads[i][j]
            
            m_vec = beta1 * m_vec + (1 - beta1) * grad_vec
            v_vec = beta2 * v_vec + (1 - beta2) * grad_vec ** 2
            m_hat = m_vec / (1 - beta1 ** t)
            v_hat = v_vec / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            
            for i in range(n_u3):
                for j in range(3):
                    params[i][j] += update[3 * i + j]
            
            lr = 0.01 / (1 + 0.001 * iteration)
        
        if best_fid > best_overall_fid:
            best_overall_fid = best_fid
            best_overall_params = best_params
    
    params = best_overall_params
    
    # Build gate list
    gates = []
    pi_idx = 0
    for gate in layout:
        if gate[0] == 'u3':
            theta, phi, lam = params[pi_idx]
            # Normalize angles to [-pi, pi] for numerical stability
            theta = (theta + pi) % (2 * pi) - pi
            phi = (phi + pi) % (2 * pi) - pi
            lam = (lam + pi) % (2 * pi) - pi
            gates.append(['u3', gate[1], float(theta), float(phi), float(lam)])
            pi_idx += 1
        else:
            gates.append(['cx', gate[1], gate[2]])
    
    # Verify CNOT budget
    cnot_count = sum(1 for g in gates if g[0] == 'cx')
    assert cnot_count <= budget, f"CNOT count {cnot_count} exceeds budget {budget}"
    
    return gates
# EVOLVE-BLOCK-END
