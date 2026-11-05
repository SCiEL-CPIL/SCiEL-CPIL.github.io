"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import expm
import time

def coefficients(stages):
    """Optimized 12-stage product formula for 6-site Heisenberg chain."""
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    # Build the 6-site open XXZ Heisenberg chain
    # Sites 0-5, bonds between adjacent sites
    # XXZ Hamiltonian: sum_j [Jx/2 (S_j^x S_{j+1}^x + S_j^y S_{j+1}^y) + Jz S_j^z S_{j+1}^z] - h sum_j S_j^z
    # Anisotropy = Jz/Jx = 0.8, field h = 0.3
    # Let Jx = 1, so Jz = 0.8, h = 0.3
    
    # Single-site spin operators (spin-1/2)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    n_sites = 6
    dim = 2 ** n_sites
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def identity(n):
        return np.eye(2 ** n, dtype=complex)
    
    # Build bond operators
    def bond_op(site1, site2, op1, op2):
        """Create op1 ⊗ op2 on sites site1, site2, identity elsewhere"""
        ops = []
        for i in range(n_sites):
            if i == site1:
                ops.append(op1)
            elif i == site2:
                ops.append(op2)
            else:
                ops.append(np.eye(2, dtype=complex))
        return kron_all(ops)
    
    Jx = 1.0
    Jz = 0.8
    h = 0.3
    
    # A: odd bonds (0-1, 2-3, 4-5) + half field
    # B: even bonds (1-2, 3-4) + half field
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Even bonds: (1,2), (3,4)
    even_bonds = [(1, 2), (3, 4)]
    
    for (i, j) in odd_bonds:
        A += Jx/2 * (bond_op(i, j, Sx, Sx) + bond_op(i, j, Sy, Sy)) + Jz * bond_op(i, j, Sz, Sz)
    
    for (i, j) in even_bonds:
        B += Jx/2 * (bond_op(i, j, Sx, Sx) + bond_op(i, j, Sy, Sy)) + Jz * bond_op(i, j, Sz, Sz)
    
    # Half field on each
    for s in range(n_sites):
        field_op = np.zeros((dim, dim), dtype=complex)
        ops = [np.eye(2, dtype=complex)] * n_sites
        ops[s] = Sz
        field_op = kron_all(ops)
        A += (-h / 2.0) * field_op
        B += (-h / 2.0) * field_op
    
    # Precompute expm for various coefficients? No, coefficients vary.
    # But we can cache based on the coefficient values.
    
    def compute_error(a_coeffs, b_coeffs, t_values=(0.25, 0.5, 1.0)):
        """Compute mean spectral norm error."""
        total_err = 0.0
        for t in t_values:
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                ak = a_coeffs[k]
                bk = b_coeffs[k]
                if abs(bk) > 1e-15:
                    U = expm(-1j * bk * t * B) @ U
                if abs(ak) > 1e-15:
                    U = expm(-1j * ak * t * A) @ U
            U_exact = expm(-1j * t * (A + B))
            err = np.linalg.norm(U - U_exact, ord=2)
            total_err += err
        return total_err / len(t_values)
    
    # Initial guess: symmetric 4th-order-like coefficients
    # For 12 stages, use a symmetric pattern
    # Symmetric: a_k = a_{13-k}, b_k = b_{13-k}
    
    # Start with a reasonable symmetric 4th-order base
    # 4th order Suzuki: x = 1/(2 - 2^(1/3))
    x4 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    
    # For 12 stages, replicate a 3-stage 4th-order pattern 4 times? 
    # Or use a better symmetric 12-stage scheme.
    
    # Let's use a symmetric pattern with 6 unique a and 6 unique b
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Initial guess from 4th-order coefficients scaled
    a_init = [x4/4, (1-2*x4)/4, x4/4, x4/4, (1-2*x4)/4, x4/4]
    b_init = [(1-x4)/4, (2*x4-1)/4, x4/4, x4/4, (2*x4-1)/4, (1-x4)/4]
    
    # Normalize to ensure sum = 1
    a_sum = sum(a_init)
    b_sum = sum(b_init)
    a_init = [x / a_sum for x in a_init]
    b_init = [x / b_sum for x in b_init]
    
    a_full_init = a_init + list(reversed(a_init))
    b_full_init = b_init + list(reversed(b_init))
    
    # Optimize using scipy
    from scipy.optimize import minimize
    
    def objective(x):
        a_unique = x[:6]
        b_unique = x[6:12]
        a_full = a_unique + list(reversed(a_unique))
        b_full = b_unique + list(reversed(b_unique))
        # Check bounds
        if any(abs(v) > 50 for v in a_full + b_full):
            return 1e10
        # Check sums
        a_sum = sum(a_full)
        b_sum = sum(b_full)
        if abs(a_sum - 1) > 1e-6 or abs(b_sum - 1) > 1e-6:
            return 1e10 + abs(a_sum - 1) + abs(b_sum - 1)
        return compute_error(a_full, b_full)
    
    # Constraints: sum(a_full) = 1, sum(b_full) = 1
    # With symmetry, sum(a_full) = 2*sum(a_unique), so sum(a_unique) = 0.5
    # Similarly sum(b_unique) = 0.5
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x[:6]) - 0.5},
        {'type': 'eq', 'fun': lambda x: sum(x[6:12]) - 0.5}
    ]
    
    bounds = [(-50, 50)] * 12
    
    x0 = a_init + b_init
    
    best_x = x0
    best_obj = objective(x0)
    
    # Try SLSQP
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        if result.success or result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except:
        pass
    
    if time.time() > deadline:
        pass
    else:
        # Try Nelder-Mead as backup
        try:
            result2 = minimize(objective, best_x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False})
            if result2.fun < best_obj:
                best_x = result2.x
                best_obj = result2.fun
        except:
            pass
    
    if time.time() > deadline:
        pass
    else:
        # Polish with tighter tolerance
        try:
            result3 = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                             constraints=constraints,
                             options={'maxiter': 100, 'ftol': 1e-15, 'disp': False})
            if result3.fun < best_obj:
                best_x = result3.x
                best_obj = result3.fun
        except:
            pass
    
    a_unique = best_x[:6]
    b_unique = best_x[6:12]
    a_full = a_unique + list(reversed(a_unique))
    b_full = b_unique + list(reversed(b_unique))
    
    # Final normalization to ensure exact sum = 1
    a_sum = sum(a_full)
    b_sum = sum(b_full)
    a_full = [x / a_sum for x in a_full]
    b_full = [x / b_sum for x in b_full]
    
    # Clamp to [-50, 50]
    a_full = [max(-50, min(50, x)) for x in a_full]
    b_full = [max(-50, min(50, x)) for x in b_full]
    
    # Re-normalize after clamping
    a_sum = sum(a_full)
    b_sum = sum(b_full)
    if a_sum != 0:
        a_full = [x / a_sum for x in a_full]
    if b_sum != 0:
        b_full = [x / b_sum for x in b_full]
    
    # Ensure exactly 12 elements
    while len(a_full) < stages:
        a_full.append(0.0)
        b_full.append(0.0)
    a_full = a_full[:stages]
    b_full = b_full[:stages]
    
    return a_full, b_full

# EVOLVE-BLOCK-END
