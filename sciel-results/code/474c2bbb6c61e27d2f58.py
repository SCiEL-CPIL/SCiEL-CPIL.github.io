import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    d = 0.8  # Anisotropy
    h = 0.3  # Field

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)

    # Precompute per-site single-site operators
    site_ops = {}
    for site in range(n):
        for name, op in (("sx", sx), ("sy", sy), ("sz", sz)):
            ops = [ident] * n
            ops[site] = op
            r = ops[0]
            for o in ops[1:]:
                r = np.kron(r, o)
            site_ops[(site, name)] = r

    def bond_op(i, j):
        return (site_ops[(i, "sx")] @ site_ops[(j, "sx")]
                + site_ops[(i, "sy")] @ site_ops[(j, "sy")]
                + d * site_ops[(i, "sz")] @ site_ops[(j, "sz")])

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)

    half_field = 0.5 * h
    for site in range(n):
        A += half_field * site_ops[(site, "sz")]
        B += half_field * site_ops[(site, "sz")]

    return A, B

def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin

    n = stages
    if n != 12:
        # Fallback to simple Strang if stages not 12, though task implies 12
        substeps = n // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < n:
            a.append(0.0)
            b.append(0.0)
        return a, b

    A, B = _build_hamiltonians()
    H = A + B
    times = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    Uexact = {}
    for t in times:
        Uexact[t] = expm(-1j * t * H)

    # We assume symmetric splitting: a_k = a_{12-k}, b_k = b_{12-k}
    # We optimize the first 6 elements of a and b.
    # Constraint: sum(a) = 1, sum(b) = 1.
    # Since a is symmetric, sum(a) = 2 * sum(a[0:6]) = 1 => sum(a[0:6]) = 0.5.
    # We optimize 5 free variables for a[0:5] and derive a[5].
    # Same for b.
    # Total 10 variables.
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        
        a_half = np.array(params[0:5])
        b_half = np.array(params[5:10])
        
        # Derive the 6th element to satisfy sum constraint
        a_5 = 0.5 - np.sum(a_half)
        b_5 = 0.5 - np.sum(b_half)
        
        a_opt = np.concatenate([a_half, [a_5]])
        b_opt = np.concatenate([b_half, [b_5]])
        
        # Check bounds [-50, 50]
        if np.any(np.abs(a_opt) > 50) or np.any(np.abs(b_opt) > 50):
            return 1e10
            
        # Construct full symmetric lists
        a_full = np.concatenate([a_opt, a_opt[::-1]])
        b_full = np.concatenate([b_opt, b_opt[::-1]])
        
        # Ensure sums are exactly 1 (floating point fix)
        a_full = a_full / np.sum(a_full)
        b_full = b_full / np.sum(b_full)
        
        errors = []
        dim = A.shape[0]
        I = np.eye(dim, dtype=complex)
        
        for t in times:
            U = I.copy()
            # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
            # Order k=1..12
            for k in range(12):
                # Note: The task says "in order k=1..12". 
                # Usually product formulas are applied right-to-left in matrix multiplication if written as U = M_1 M_2 ...
                # But here it says "product formula U(t) = prod_k ... in order k=1..12".
                # This implies U = (ExpB_1 ExpA_1) (ExpB_2 ExpA_2) ...
                # So we start with I, and multiply on the left by each step?
                # Or right?
                # Standard convention: U(t) = U_{N} ... U_1. 
                # If the product is written left-to-right as k=1..12, it usually means the first operator applied is k=1.
                # In matrix multiplication U = M_12 ... M_1, the state is transformed by M_1 first.
                # Let's assume U = M_12 @ ... @ M_1 @ I.
                # So we accumulate U = M_k @ U starting from I? No, that would be M_12 @ ... @ M_1.
                # Let's trace: 
                # Step 1: U = M_1
                # Step 2: U = M_2 @ U = M_2 @ M_1
                # ...
                # Step 12: U = M_12 @ ... @ M_1
                # This is the standard "time-ordered" product where the earliest time is on the right.
                
                step_b = expm(-1j * b_full[k] * t * B)
                step_a = expm(-1j * a_full[k] * t * A)
                M_k = step_b @ step_a
                U = M_k @ U
            
            err = np.linalg.norm(U - Uexact[t], 2)
            errors.append(err)
            
        return float(np.mean(errors))

    # Initial guess: Strang splitting
    # For 12 stages, Strang is a_k = 1/24, b_k = 1/12? 
    # Actually, simple Strang repeated 6 times:
    # a = [1/12, 1/12, ...] ? No.
    # Strang: 0.5 A, 1 B, 0.5 A.
    # If we repeat 6 times: 
    # a: [0.5/6, 0, 0.5/6, 0.5/6, 0, 0.5/6, ...]
    # b: [0, 1/6, 0, 0, 1/6, 0, ...]
    # This is complex. Let's just use uniform:
    # a_k = 1/12, b_k = 1/12.
    # This corresponds to a simple split.
    
    x0_a = np.array([1.0/12.0] * 5)
    x0_b = np.array([1.0/12.0] * 5)
    x0 = np.concatenate([x0_a, x0_b])
    
    # Optimize
    best_x = x0
    best_err = objective(x0)
    
    try:
        # Use L-BFGS-B or Nelder-Mead
        # L-BFGS-B is good for bounded problems, but we have no explicit bounds on the 5 free vars, 
        # only the derived 6th must be in [-50, 50].
        # We'll use Nelder-Mead for robustness with the non-smoothness of the spectral norm (though it's smooth).
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except Exception:
        pass

    # Extract best coefficients
    a_half = np.array(best_x[0:5])
    b_half = np.array(best_x[5:10])
    a_5 = 0.5 - np.sum(a_half)
    b_5 = 0.5 - np.sum(b_half)
    
    a_opt = np.concatenate([a_half, [a_5]])
    b_opt = np.concatenate([b_half, [b_5]])
    
    a_full = np.concatenate([a_opt, a_opt[::-1]])
    b_full = np.concatenate([b_opt, b_opt[::-1]])
    
    # Final cleanup
    a_full = np.clip(a_full, -50.0, 50.0)
    b_full = np.clip(b_full, -50.0, 50.0)
    
    # Renormalize to ensure sum=1
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b

    return a_full.tolist(), b_full.tolist()
