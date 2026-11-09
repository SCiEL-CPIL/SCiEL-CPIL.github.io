"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """Optimized 12-stage product formula coefficients via limited search."""
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    # Build Hamiltonians for 6-site open XXZ Heisenberg chain
    N = 6
    gamma = 0.8
    h = 0.3
    dim = 2 ** N
    
    # Single-site operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def site_op(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    # Build A (odd bonds + half field) and B (even bonds + half field)
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for j in range(N - 1):
        term = 0.5 * (site_op(j, sx) @ site_op(j+1, sx) + 
                      site_op(j, sy) @ site_op(j+1, sy) + 
                      gamma * site_op(j, sz) @ site_op(j+1, sz))
        if j % 2 == 0:  # even bond -> B
            B += term
        else:  # odd bond -> A
            A += term
    
    # Add half field to each
    for j in range(N):
        field_term = 0.5 * h * site_op(j, sz)
        if j % 2 == 0:  # even site -> B
            B += field_term
        else:  # odd site -> A
            A += field_term
    
    # Exact propagator
    H = A + B
    
    # Precompute eigendecompositions for speed
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    wH, vH = np.linalg.eigh(H)
    
    def expm_from_eig(w, v, t):
        return (v * np.exp(-1j * w * t)) @ v.conj().T
    
    def spectral_norm_error(a, b):
        """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
        errors = []
        for t in [0.25, 0.5, 1.0]:
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                U = expm_from_eig(wB, vB, b[k] * t) @ U
                U = expm_from_eig(wA, vA, a[k] * t) @ U
            U_exact = expm_from_eig(wH, vH, t)
            diff = U - U_exact
            errors.append(np.linalg.svd(diff, compute_uv=False)[0])
        return np.mean(errors)
    
    # Initial guess: Strang splitting
    substeps = stages // 2
    a0 = [0.5 / substeps] * stages
    b0 = []
    for _ in range(substeps):
        b0 += [1.0 / substeps, 0.0]
    while len(b0) < stages:
        b0.append(0.0)
    
    best_a, best_b = a0[:], b0[:]
    best_err = spectral_norm_error(best_a, best_b)
    
    # Coordinate descent with time limit
    rng = np.random.default_rng(42)
    step_size = 0.01
    
    while time.time() < deadline:
        improved = False
        for i in range(stages):
            if time.time() > deadline:
                break
            # Try adjusting a[i]
            for delta in [-step_size, step_size]:
                a_try = best_a[:]
                a_try[i] += delta
                # Adjust a[0] to maintain sum = 1
                a_try[0] += (1 - sum(a_try))
                a_try = [max(-50, min(50, x)) for x in a_try]
                err = spectral_norm_error(a_try, best_b)
                if err < best_err:
                    best_a = a_try
                    best_err = err
                    improved = True
            # Try adjusting b[i]
            for delta in [-step_size, step_size]:
                b_try = best_b[:]
                b_try[i] += delta
                b_try[0] += (1 - sum(b_try))
                b_try = [max(-50, min(50, x)) for x in b_try]
                err = spectral_norm_error(best_a, b_try)
                if err < best_err:
                    best_b = b_try
                    best_err = err
                    improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    # Final check: ensure valid output
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    # Normalize to ensure sum = 1
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if a_sum != 0:
        best_a = [x / a_sum for x in best_a]
    if b_sum != 0:
        best_b = [x / b_sum for x in best_b]
    
    return best_a, best_b
# EVOLVE-BLOCK-END
