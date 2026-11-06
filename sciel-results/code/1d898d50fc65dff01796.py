"""Optimized product-formula coefficients for Heisenberg-chain time evolution."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    lam = 0.8  # anisotropy
    h = 0.3    # field
    
    # XXZ Hamiltonian terms:
    # S_i^+ S_j^- + S_i^- S_j^+ + lam * S_i^z S_j^z
    # For spin-1/2: S^+ = |0><1|, S^- = |1><0|, S^z = diag(1/2, -1/2)
    # On 2 sites: S^+S^- + S^-S^+ = [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]] (in |00>,|01>,|10>,|11> basis)
    # Actually let's use standard basis |0>,|1> for spin up/down
    # S^+|1> = |0>, S^-|0> = |1>
    # S^+S^- + S^-S^+ on two qubits:
    # |00> -> S^-_1|00> = |10>, S^+_2|10> = |11>... let me just compute directly
    
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N - 1):
        for j in range(N - 1):
            if j <= i:
                continue
            # Compute S_i^+ S_j^- + S_i^- S_j^+ + lam * S_i^z S_j^z
            # S^+ = [[0,1],[0,0]], S^- = [[0,0],[1,0]], S^z = [[0.5,0],[0,-0.5]]
            Sp = np.array([[0, 1], [0, 0]], dtype=complex)
            Sm = np.array([[0, 0], [1, 0]], dtype=complex)
            Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
            
            for state in range(dim):
                bits = [(state >> k) & 1 for k in range(N)]
                # Apply S_i^+ S_j^-
                # S^- on j: flips bit j from 0 to 1 (if bit is 0)
                # S^+ on i: flips bit i from 1 to 0 (if bit is 1)
                # Wait, let me be careful. In physics, S^+ raises spin (|down> -> |up>)
                # Let's say bit 0 = spin up (|0>), bit 1 = spin down (|1>)
                # S^+ |1> = |0> (raises), S^- |0> = |1> (lowers)
                
                # S_i^+ S_j^-(|state>): first apply S_j^- then S_i^+
                # S_j^-: if bits[j] == 0, flip to 1
                if bits[j] == 0:
                    new_bits = list(bits)
                    new_bits[j] = 1
                    # Now apply S_i^+: if new_bits[i] == 1, flip to 0
                    if new_bits[i] == 1:
                        new_bits[i] = 0
                        new_state = sum(b << k for k, b in enumerate(new_bits))
                        H[new_state, state] += 1.0
                
                # S_i^- S_j^+(|state>): first apply S_j^+ then S_i^-
                # S_j^+: if bits[j] == 1, flip to 0
                if bits[j] == 1:
                    new_bits = list(bits)
                    new_bits[j] = 0
                    # Now apply S_i^-: if new_bits[i] == 0, flip to 1
                    if new_bits[i] == 0:
                        new_bits[i] = 1
                        new_state = sum(b << k for k, b in enumerate(new_bits))
                        H[new_state, state] += 1.0
                
                # lam * S_i^z S_j^z
                sz_i = 0.5 if bits[i] == 0 else -0.5
                sz_j = 0.5 if bits[j] == 0 else -0.5
                H[state, state] += lam * sz_i * sz_j
    
    # Field: h * sum_i S_i^z
    for i in range(N):
        for state in range(dim):
            bits = [(state >> k) & 1 for k in range(N)]
            sz = 0.5 if bits[i] == 0 else -0.5
            H[state, state] += h * sz
    
    # Split into even and odd bonds
    # Even bonds: (0,1), (2,3), (4,5) -> index 0,2,4
    # Odd bonds: (1,2), (3,4) -> index 1,3
    # Wait, for N=6: bonds are (0,1),(1,2),(2,3),(3,4),(4,5)
    # Even bonds (bond index even): (0,1), (2,3), (4,5)
    # Odd bonds (bond index odd): (1,2), (3,4)
    
    A = np.zeros((dim, dim), dtype=complex)  # odd bonds
    B = np.zeros((dim, dim), dtype=complex)  # even bonds
    
    for i in range(N - 1):
        for j in range(N - 1):
            if j <= i:
                continue
            bond_idx = i  # bond between site i and i+1
            # Actually bond i is between sites i and i+1
            # Even bonds: i=0,2,4; Odd bonds: i=1,3
            
            # Reconstruct the bond contribution
            Sp = np.array([[0, 1], [0, 0]], dtype=complex)
            Sm = np.array([[0, 0], [1, 0]], dtype=complex)
            Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
            
            for state in range(dim):
                bits = [(state >> k) & 1 for k in range(N)]
                
                # S_i^+ S_j^- + S_i^- S_j^+
                if bits[j] == 0:
                    new_bits = list(bits)
                    new_bits[j] = 1
                    if new_bits[i] == 1:
                        new_bits[i] = 0
                        new_state = sum(b << k for k, b in enumerate(new_bits))
                        if bond_idx % 2 == 0:
                            B[new_state, state] += 1.0
                        else:
                            A[new_state, state] += 1.0
                
                if bits[j] == 1:
                    new_bits = list(bits)
                    new_bits[j] = 0
                    if new_bits[i] == 0:
                        new_bits[i] = 1
                        new_state = sum(b << k for k, b in enumerate(new_bits))
                        if bond_idx % 2 == 0:
                            B[new_state, state] += 1.0
                        else:
                            A[new_state, state] += 1.0
                
                # lam * S_i^z S_j^z
                sz_i = 0.5 if bits[i] == 0 else -0.5
                sz_j = 0.5 if bits[j] == 0 else -0.5
                contrib = lam * sz_i * sz_j
                if bond_idx % 2 == 0:
                    B[state, state] += contrib
                else:
                    A[state, state] += contrib
    
    # Field split: half to A, half to B
    for i in range(N):
        for state in range(dim):
            bits = [(state >> k) & 1 for k in range(N)]
            sz = 0.5 if bits[i] == 0 else -0.5
            half_h = h * sz * 0.5
            A[state, state] += half_h
            B[state, state] += half_h
    
    return A, B


def _compute_error(coeffs, A, B, times):
    """Compute mean spectral norm error."""
    a_list, b_list = coeffs
    dim = A.shape[0]
    errors = []
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(len(a_list)):
            expB = expm(-1j * b_list[k] * t * B)
            expA = expm(-1j * a_list[k] * t * A)
            U = expB @ expA @ U
        exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - exact, 2)
        errors.append(err)
    return np.mean(errors)


def _objective(x, A, B, times):
    """Objective function for optimization."""
    n = len(x) // 2
    a = x[:n]
    b = x[n:]
    # Enforce sum constraints via penalty
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    penalty = 100.0 * (sum_a - 1.0)**2 + 100.0 * (sum_b - 1.0)**2
    if penalty > 1.0:
        return penalty
    return _compute_error((a.tolist(), b.tolist()), A, B, times)


def coefficients(stages):
    """Optimized 12-stage product formula coefficients."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    n = stages  # 12
    
    # Try a few different initial guesses and optimization strategies
    best_err = np.inf
    best_a = [1.0 / n] * n
    best_b = [1.0 / n] * n
    
    # Strategy 1: Symmetric Suzuki-Trotter based initial guess
    # For a 12-stage symmetric scheme, we can use patterns from high-order splitting
    # Try several initializations
    
    initializations = []
    
    # Uniform
    initializations.append([1.0/n] * (2*n))
    
    # Symmetric: a = [c1,c2,...,c6,c6,...,c2,c1], b = [d1,d2,...,d6,d6,...,d2,d1]
    # with sum = 1
    # Simple symmetric: a_k = a_{13-k}, b_k = b_{13-k}
    # Try a pattern inspired by 4th order Strang with substeps
    # For 6 substeps of Strang: a = [1/12]*12, b = [1/6,0,1/6,0,1/6,0,1/6,0,1/6,0,1/6,0] -> sums don't work
    # Let's just try a few random symmetric patterns
    
    np.random.seed(42)
    for trial in range(5):
        # Random symmetric coefficients
        a_half = np.random.uniform(0.05, 0.25, n//2)
        b_half = np.random.uniform(0.05, 0.25, n//2)
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        # Normalize
        a_full = a_full / np.sum(a_full)
        b_full = b_full / np.sum(b_full)
        initializations.append(list(a_full) + list(b_full))
    
    # Also try a structured one: alternate large/small
    a_struct = [0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15]
    b_struct = [0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1]
    a_struct = np.array(a_struct) / np.sum(a_struct)
    b_struct = np.array(b_struct) / np.sum(b_struct)
    initializations.append(list(a_struct) + list(b_struct))
    
    for init in initializations:
        if time.time() > deadline - 20.0:
            break
        
        x0 = np.array(init, dtype=float)
        
        # Use Nelder-Mead (no gradient needed, robust)
        try:
            res = minimize(_objective, x0, args=(A, B, times), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.success or res.fun < best_err:
                x_opt = res.x
                a_opt = x_opt[:n]
                b_opt = x_opt[n:]
                # Normalize to ensure exact sum = 1
                a_opt = a_opt / np.sum(a_opt)
                b_opt = b_opt / np.sum(b_opt)
                # Clip to [-50, 50]
                a_opt = np.clip(a_opt, -50, 50)
                b_opt = np.clip(b_opt, -50, 50)
                # Re-normalize after clipping
                a_opt = a_opt / np.sum(a_opt)
                b_opt = b_opt / np.sum(b_opt)
                
                err = _compute_error((a_opt.tolist(), b_opt.tolist()), A, B, times)
                if err < best_err:
                    best_err = err
                    best_a = a_opt.tolist()
                    best_b = b_opt.tolist()
        except Exception:
            pass
    
    # Final polish: try L-BFGS-B with bounds on the best solution found
    if time.time() < deadline - 15.0:
        try:
            x0 = np.array(best_a + best_b, dtype=float)
            bounds = [(-50, 50)] * (2 * n)
            res = minimize(_objective, x0, args=(A, B, times), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_err:
                x_opt = res.x
                a_opt = x_opt[:n]
                b_opt = x_opt[n:]
                a_opt = a_opt / np.sum(a_opt)
                b_opt = b_opt / np.sum(b_opt)
                a_opt = np.clip(a_opt, -50, 50)
                b_opt = np.clip(b_opt, -50, 50)
                a_opt = a_opt / np.sum(a_opt)
                b_opt = b_opt / np.sum(b_opt)
                
                err = _compute_error((a_opt.tolist(), b_opt.tolist()), A, B, times)
                if err < best_err:
                    best_err = err
                    best_a = a_opt.tolist()
                    best_b = b_opt.tolist()
        except Exception:
            pass
    
    # Safety: ensure constraints are met
    best_a = [float(x) for x in best_a]
    best_b = [float(x) for x in best_b]
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x / sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x / sb for x in best_b]
    # Clip
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    # Re-normalize
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x / sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x / sb for x in best_b]
    
    return best_a, best_b
