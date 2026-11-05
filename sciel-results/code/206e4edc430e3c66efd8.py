import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def _build_hamiltonians():
    """Construct A and B for the 6-site XXZ chain."""
    n_sites = 6
    dim = 2 ** n_sites
    
    # Identity matrix for sites
    I2 = np.eye(2, dtype=complex)
    
    def kron_list(ops):
        """Kronecker product of a list of operators."""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    # Precompute site operators
    # For site j (0-indexed), the operator is I ... I op I ... I
    # It is more efficient to build the full matrix directly for small n
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    anisotropy = 0.8
    field = 0.3
    
    # Field part
    # H_field = 0.3 * sum sz_j
    # Split evenly: 0.15 * sum sz_j to A and B
    field_part = np.zeros((dim, dim), dtype=complex)
    for j in range(n_sites):
        ops = [I2] * n_sites
        ops[j] = sz
        field_part += kron_list(ops)
    
    A += 0.15 * field_part
    B += 0.15 * field_part
    
    # Bond part
    # H_bond(j) = 0.5 * (sx_j sx_{j+1} + sy_j sy_{j+1}) + 0.8 * sz_j sz_{j+1}
    for j in range(n_sites - 1):
        ops_sx_sx = [I2] * n_sites
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        term_sx_sx = kron_list(ops_sx_sx)
        
        ops_sy_sy = [I2] * n_sites
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        term_sy_sy = kron_list(ops_sy_sy)
        
        ops_sz_sz = [I2] * n_sites
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        term_sz_sz = kron_list(ops_sz_sz)
        
        bond_ham = 0.5 * (term_sx_sx + term_sy_sy) + anisotropy * term_sz_sz
        
        if j % 2 == 0:  # j=0, 2, 4 (1-indexed 1, 3, 5) -> Odd bonds
            A += bond_ham
        else:           # j=1, 3 (1-indexed 2, 4) -> Even bonds
            B += bond_ham
            
    return A, B

# Pre-compute Hamiltonians
A_H, B_H = _build_hamiltonians()
AB_H = A_H + B_H

# Pre-compute exact propagators for the specific t values
t_values = [0.25, 0.5, 1.0]
U_exact = [expm(-1j * t * AB_H) for t in t_values]

def _objective(x):
    """
    x: array of 22 elements.
    x[0:11] are a[0]..a[10]. a[11] = 1 - sum(x[0:11])
    x[11:22] are b[0]..b[10]. b[11] = 1 - sum(x[11:22])
    """
    a_free = x[0:11]
    b_free = x[11:22]
    
    a_11 = 1.0 - np.sum(a_free)
    b_11 = 1.0 - np.sum(b_free)
    
    # Check bounds
    if not (-50.0 <= a_11 <= 50.0) or not (-50.0 <= b_11 <= 50.0):
        return 1e10
        
    a = np.concatenate([a_free, [a_11]])
    b = np.concatenate([b_free, [b_11]])
    
    # Check bounds for all
    if np.any(a < -50.0) or np.any(a > 50.0) or np.any(b < -50.0) or np.any(b > 50.0):
        return 1e10
        
    total_error = 0.0
    for i, t in enumerate(t_values):
        U_t = np.eye(64, dtype=complex)
        for k in range(12):
            # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
            # Note: The formula is prod_k exp(-i b_k t B) exp(-i a_k t A)
            # This implies U_total = (F_12 ... F_1) where F_k = exp(-i b_k t B) exp(-i a_k t A)
            # So we multiply from right to left if we start with identity and left multiply?
            # U_new = U_old @ F_k
            # Let's verify: U = F_12 ... F_1.
            # Start U = I.
            # U = I @ F_1 = F_1
            # U = F_1 @ F_2
            # ...
            # U = F_1 ... F_12.
            # This matches the order k=1..12.
            
            expB = expm(-1j * b[k] * t * B_H)
            expA = expm(-1j * a[k] * t * A_H)
            F_k = expB @ expA
            U_t = U_t @ F_k
            
        diff = U_t - U_exact[i]
        # Spectral norm is the largest singular value
        err = np.linalg.norm(diff, 2)
        total_error += err
        
    return total_error / len(t_values)

def coefficients(stages):
    """
    Optimize coefficients for the 12-stage product formula.
    """
    # If stages is not 12, we still return 12 elements as per contract?
    # Contract says: "two lists of 12 finite numbers".
    # The function signature is coefficients(stages).
    # We assume stages=12 is the only case, but we should handle it gracefully.
    # If stages != 12, we can pad or truncate, but the contract implies 12.
    # We will optimize for 12 and return 12.
    
    # Initial guess: simple splitting
    # a = [1/12]*12, b = [1/12]*12
    x0 = np.array([1.0/12.0] * 22)
    
    bounds = [(-50.0, 50.0)] * 22
    
    # Use differential evolution
    # We need to be careful with time.
    # DE is stochastic and can be slow.
    # Let's try a smaller population and fewer iterations.
    
    best_x = x0
    best_val = _objective(x0)
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            tol=1e-4, 
            popsize=15, 
            seed=42,
            init='latinhypercube',
            # Update callback to check time
            callback=None 
        )
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
            
    except Exception:
        pass
        
    # Check time
    if time.time() > _DEADLINE:
        # Return current best
        a_free = best_x[0:11]
        b_free = best_x[11:22]
        a_11 = 1.0 - np.sum(a_free)
        b_11 = 1.0 - np.sum(b_free)
        a = np.concatenate([a_free, [a_11]])
        b = np.concatenate([b_free, [b_11]])
        return list(a), list(b)
    
    # Refine with local optimization (L-BFGS-B) if time permits
    # This can significantly reduce error.
    try:
        from scipy.optimize import minimize
        # Bounds for local opt: we need to map to unconstrained or use bounds.
        # L-BFGS-B supports bounds.
        # We have 22 variables with bounds [-50, 50].
        # The constraint sum(a)=1, sum(b)=1 is handled by the parameterization.
        # But L-BFGS-B doesn't know about the parameterization constraint (sum=1).
        # We must use the parameterized objective.
        
        res = minimize(
            _objective, 
            best_x, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass

    a_free = best_x[0:11]
    b_free = best_x[11:22]
    a_11 = 1.0 - np.sum(a_free)
    b_11 = 1.0 - np.sum(b_free)
    
    # Final safety check
    if not (-50.0 <= a_11 <= 50.0) or not (-50.0 <= b_11 <= 50.0):
        # Fallback to seed
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
    else:
        a = np.concatenate([a_free, [a_11]])
        b = np.concatenate([b_free, [b_11]])
        
    return list(a), list(b)

