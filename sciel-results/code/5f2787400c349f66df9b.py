"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def coefficients(stages):
    """Optimize product-formula coefficients via local search on spectral-norm error."""
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Build the 6-site open XXZ Heisenberg chain Hamiltonians
    # Anisotropy gamma=0.8, field h=0.3
    N = 6
    gamma = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Build full Hamiltonian H = A + B
    # A: odd bonds (1-2, 3-4, 5-6) + half field
    # B: even bonds (2-3, 4-5) + half field
    # Wait, for 6 sites open chain:
    # Bonds: 1-2, 2-3, 3-4, 4-5, 5-6
    # Odd bonds (between site i and i+1 where i is odd): 1-2, 3-4, 5-6
    # Even bonds (between site i and i+1 where i is even): 2-3, 4-5

    def site_op(op, site):
        """Apply operator op on given site (0-indexed), identity elsewhere."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    def bond_op(i, j):
        """Build S_i · S_j = 0.5*(Sx_i*Sx_j + Sy_i*Sy_j + gamma*Sz_i*Sz_j)"""
        # Sx*Sx + Sy*Sy = 2*(Sx_i*Sx_j + Sy_i*Sy_j) ... wait
        # Actually S_i · S_j = Sx_i*Sx_j + Sy_i*Sy_j + gamma*Sz_i*Sz_j
        # where S = sigma/2
        # So Sx_i*Sx_j = 0.25 * sx_i * sx_j
        sx_i = site_op(sx, i) * 0.5
        sx_j = site_op(sx, j) * 0.5
        sy_i = site_op(sy, i) * 0.5
        sy_j = site_op(sy, j) * 0.5
        sz_i = site_op(sz, i) * 0.5
        sz_j = site_op(sz, j) * 0.5

        return sx_i @ sx_j + sy_i @ sy_j + gamma * (sz_i @ sz_j)

    # Build A (odd bonds) and B (even bonds)
    # Each gets half the uniform field: -h * sum_i Sz_i / 2
    field_total = -h * sum(site_op(sz, i) * 0.5 for i in range(N))
    half_field = 0.5 * field_total

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)

    # Odd bonds: (0,1), (2,3), (4,5)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_op(i, j)
    A += half_field

    # Even bonds: (1,2), (3,4)
    for i, j in [(1, 2), (3, 4)]:
        B += bond_op(i, j)
    B += half_field

    H = A + B

    # Precompute exact propagators at the three evaluation times
    t_evals = [0.25, 0.5, 1.0]

    from scipy.linalg import expm
    exact_prop = [expm(-1j * t * H) for t in t_evals]

    def objective(coeffs_a, coeffs_b):
        """Compute mean spectral norm error over the three time points."""
        errors = []
        for t, U_exact in zip(t_evals, exact_prop):
            U = np.eye(2**N, dtype=complex)
            for k in range(stages):
                # exp(-i * b_k * t * B) then exp(-i * a_k * t * A)
                if abs(coeffs_b[k]) > 1e-15:
                    U = expm(-1j * coeffs_b[k] * t * B) @ U
                if abs(coeffs_a[k]) > 1e-15:
                    U = expm(-1j * coeffs_a[k] * t * A) @ U
            err = np.linalg.norm(U - U_exact, 2)
            errors.append(err)
        return np.mean(errors)

    # Initial guess: equal splitting
    a0 = [1.0 / stages] * stages
    b0 = [1.0 / stages] * stages

    # Local search using Nelder-Mead or simple coordinate descent
    # Use scipy's minimize with Nelder-Mead for speed
    from scipy.optimize import minimize

    def obj_flat(x):
        """Flatten coefficients and compute objective."""
        a = x[:stages]
        b = x[stages:]
        return objective(a, b)

    x0 = np.array(a0 + b0)
    best_x = x0.copy()
    best_obj = obj_flat(x0)

    # Try a few random restarts and local optimizations
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 30:
            break

        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb around best solution found so far
            scale = max(0.01, 0.1 / (restart + 1))
            x_start = best_x + np.random.normal(0, scale, len(x0))
            # Clamp to [-50, 50]
            x_start = np.clip(x_start, -50, 50)

        try:
            res = minimize(obj_flat, x_start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass

        if time.time() > deadline - 30:
            break

    # Extract best coefficients
    a_best = best_x[:stages].tolist()
    b_best = best_x[stages:].tolist()

    # Enforce sum constraints: sum(a) = 1, sum(b) = 1
    # Adjust by scaling
    sum_a = sum(a_best)
    sum_b = sum(b_best)
    if abs(sum_a) > 1e-15:
        a_best = [x / sum_a for x in a_best]
    if abs(sum_b) > 1e-15:
        b_best = [x / sum_b for x in b_best]

    # Clamp to [-50, 50] and ensure finite
    a_best = [max(-50, min(50, x)) if np.isfinite(x) else 0.0 for x in a_best]
    b_best = [max(-50, min(50, x)) if np.isfinite(x) else 0.0 for x in b_best]

    # Final check: ensure sums are close to 1
    # If scaling broke things, fall back to equal split
    if abs(sum(a_best) - 1) > 1e-6 or abs(sum(b_best) - 1) > 1e-6:
        a_best = [1.0 / stages] * stages
        b_best = [1.0 / stages] * stages

    return a_best, b_best
# EVOLVE-BLOCK-END
