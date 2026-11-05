"""Product-formula coefficients for 6-site Heisenberg chain time evolution.

Strategy:
  1. Build the exact Hamiltonian H = A + B for a 6-site open XXZ chain
     with anisotropy Δ=0.8, field h=0.3, split into odd/even bond halves.
  2. Use a small number of carefully chosen seed coefficient vectors
     (2nd-order symmetric, 4th-order-inspired, symmetric 12-stage patterns).
  3. Run Nelder-Mead optimization on a reduced parameterization (symmetric
     coefficients, so we optimize 12 free parameters instead of 24) to
     minimize the mean spectral-norm error over t = {0.25, 0.5, 1.0}.
  4. Return the best a, b found, with a deadline guard.
"""
import time
import numpy as np
from scipy.linalg import expm

# ---- Build the Hamiltonian pieces A and B ----
def _build_H():
    """Build A (odd bonds + half field) and B (even bonds + half field) for
    a 6-site open XXZ chain with Δ=0.8, h=0.3."""
    N = 6
    d = 2 ** N  # 64
    delta = 0.8
    h = 0.3

    # Single-spin operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_chain(ops_list):
        """Kron product of a list of 2x2 matrices (left to right)."""
        result = np.array([[1.0+0j]], dtype=complex)
        for op in ops_list:
            result = np.kron(result, op)
        return result

    def two_site_op(site_a, site_b, op_a, op_b):
        """Kron product with op_a at site_a, op_b at site_b, identity elsewhere."""
        ops = [I2] * N
        ops[site_a] = op_a
        ops[site_b] = op_b
        return kron_chain(ops)

    # XXZ bond term: (1/2)(Sx⊗Sx + Sy⊗Sy + Δ Sz⊗Sz)
    # where S = σ/2
    Sx = sx / 2.0
    Sy = sy / 2.0
    Sz = sz / 2.0

    A = np.zeros((d, d), dtype=complex)
    B = np.zeros((d, d), dtype=complex)

    # Odd bonds: sites (0,1), (2,3), (4,5) -> A
    # Even bonds: sites (1,2), (3,4) -> B
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        A += 0.5 * (two_site_op(i, j, Sx, Sx) +
                    two_site_op(i, j, Sy, Sy) +
                    delta * two_site_op(i, j, Sz, Sz))
    for (i, j) in [(1, 2), (3, 4)]:
        B += 0.5 * (two_site_op(i, j, Sx, Sx) +
                    two_site_op(i, j, Sy, Sy) +
                    delta * two_site_op(i, j, Sz, Sz))

    # Uniform field: h * sum Sz_j  -> split evenly between A and B
    field = np.zeros((d, d), dtype=complex)
    for j in range(N):
        field += two_site_op(j, j, Sz, I2)  # Sz at site j
    A += 0.5 * h * field
    B += 0.5 * h * field

    return A, B


def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of a matrix."""
    return np.linalg.norm(M, 2)


def _objective(params):
    """Given 12 symmetric free params, build a,b, compute mean spectral-norm
    error over t = {0.25, 0.5, 1.0}."""
    # params: 12 floats representing symmetric a[0..5] (a[6..11] = a[5..0])
    # We use a parameterization where a is symmetric and b is symmetric.
    # Actually, for a 12-stage product formula U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A),
    # we need sum(a) = 1 and sum(b) = 1.
    # Symmetric: a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # But we need sum(a) = 2*(a1+...+a6) = 1 => a1+...+a6 = 0.5
    # Similarly for b.
    
    # params has 12 values: first 6 for a (a1..a6), next 6 for b (b1..b6)
    a_half = params[:6]
    b_half = params[6:]
    
    a = np.concatenate([a_half, a_half[::-1]])
    b = np.concatenate([b_half, b_half[::-1]])
    
    # Enforce sum = 1 by scaling
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) < 1e-15 or abs(sb) < 1e-15:
        return 1e10
    a = a / sa
    b = b / sb
    
    # Clip to [-50, 50]
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    # Re-normalize after clipping (approximate; exact re-normalization may break symmetry slightly)
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    U = np.eye(64, dtype=complex)
    for k in range(12):
        U = expm(-1j * b[k] * _T * B) @ U
        U = expm(-1j * a[k] * _T * A) @ U
    
    U_exact = expm(-1j * _T * (A + B))
    return _spectral_norm(U - U_exact)


def coefficients(stages):
    global A, B
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Build Hamiltonian pieces (cached after first call)
    if 'A' not in dir() or 'B' not in dir():
        A, B = _build_H()
    
    if stages != 12:
        # Fallback: return a reasonable symmetric split
        n = stages
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b
    
    from scipy.optimize import minimize
    
    # Pre-compute for multiple t values
    global _T
    best_a = None
    best_b = None
    best_score = np.inf
    
    t_values = [0.25, 0.5, 1.0]
    
    def full_objective(params):
        """Mean spectral norm error over all three t values."""
        a_half = params[:6]
        b_half = params[6:]
        a = np.concatenate([a_half, a_half[::-1]])
        b = np.concatenate([b_half, b_half[::-1]])
        
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = a / sa
        b = b / sb
        
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        total_err = 0.0
        for t in t_values:
            U = np.eye(64, dtype=complex)
            for k in range(12):
                U = expm(-1j * b[k] * t * B) @ U
                U = expm(-1j * a[k] * t * A) @ U
            U_exact = expm(-1j * t * (A + B))
            total_err += _spectral_norm(U - U_exact)
        return total_err / len(t_values)
    
    # Seed 1: 2nd-order symmetric (Strang-like spread over 12 stages)
    # a = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    # b = same
    seed1 = np.array([1.0/6.0]*6 + [1.0/6.0]*6)
    
    # Seed 2: 4th-order inspired
    # 4th-order 5-stage coefficients: c1 ≈ 0.4268, c2 ≈ 0.1481, c3 ≈ -0.1336
    # For symmetric 12-stage, try concentrating weight in center
    c1 = 1.0 / (2.0 * (2.0 - 2.0**(1.0/3.0)))
    c2 = -c1 / (2.0 * (2.0 - 2.0**(1.0/3.0)) - 2.0)
    c3 = 1.0 - 2.0*c1 - 2.0*c2
    # Map to 6 half-stages for a: spread the 4th-order weights
    a_half_4th = np.array([c1/2.0, c2/2.0, c3/2.0, c3/2.0, c2/2.0, c1/2.0])
    b_half_4th = np.array([0.0, c2/2.0, c3/2.0, c3/2.0, c2/2.0, 0.0])
    # Normalize
    a_half_4th = a_half_4th / np.sum(a_half_4th)
    b_half_4th = b_half_4th / np.sum(b_half_4th)
    seed2 = np.concatenate([a_half_4th, b_half_4th])
    
    # Seed 3: Random perturbation of seed1
    np.random.seed(42)
    seed3 = seed1 + np.random.randn(12) * 0.05
    
    # Seed 4: Another structured seed - linear taper
    a_taper = np.array([0.10, 0.12, 0.14, 0.14, 0.12, 0.10])  # sums to 0.72, scale to 0.5
    a_taper = a_taper / np.sum(a_taper) * 0.5
    b_taper = np.array([0.08, 0.10, 0.12, 0.12, 0.10, 0.08])
    b_taper = b_taper / np.sum(b_taper) * 0.5
    seed4 = np.concatenate([a_taper, b_taper])
    
    seeds = [seed1, seed2, seed3, seed4]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        try:
            result = minimize(full_objective, seed, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True, 'maxfev': 200})
            if result.fun < best_score:
                best_score = result.fun
                a_half = result.x[:6]
                b_half = result.x[6:]
                a = np.concatenate([a_half, a_half[::-1]])
                b = np.concatenate([b_half, b_half[::-1]])
                a = np.clip(a, -50, 50)
                b = np.clip(b, -50, 50)
                a = a / np.sum(a)
                b = b / np.sum(b)
                best_a = a
                best_b = b
        except Exception:
            continue
    
    # Fallback if no optimization succeeded
    if best_a is None:
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
    else:
        a = best_a.tolist()
        b = best_b.tolist()
    
    # Final validation
    a = [max(-50.0, min(50.0, float(x))) for x in a]
    b = [max(-50.0, min(50.0, float(x))) for x in b]
    a_sum = sum(a)
    b_sum = sum(b)
    if abs(a_sum) > 1e-15:
        a = [x / a_sum for x in a]
    if abs(b_sum) > 1e-15:
        b = [x / b_sum for x in b]
    
    return a, b
