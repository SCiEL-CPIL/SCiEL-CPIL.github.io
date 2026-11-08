"""Iterative rank-reduction search for 2x3*3x10 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _tensor_residual(U, V, W):
    """Return dict of (i,j,k)->(i,k) residuals: for each term, U[i,j]*V[j,k]*W[k,i] should sum to 1 at (i,k)."""
    n, m, p = DIMENSIONS
    # Initialize sums to 0
    sums = np.zeros((n, p), dtype=object)
    rank = U.shape[1]
    for t in range(rank):
        Ut = U[:, t]
        Vt = V[:, t]
        Wt = W[:, t]
        # For each i,j,k where all three are nonzero, add to sums[i,k]
        for i in range(n):
            for j in range(m):
                uij = Ut[i * m + j]
                if uij == 0:
                    continue
                for k in range(p):
                    vjk = Vt[j * p + k]
                    if vjk == 0:
                        continue
                    wki = Wt[k * n + i]
                    if wki == 0:
                        continue
                    sums[i, k] = sums[i, k] + uij * vjk * wki
    # Check if all are 1
    ok = True
    for i in range(n):
        for k in range(p):
            if sums[i, k] != 1:
                ok = False
    return ok

def _try_eliminate(U, V, W, t0, deadline):
    """Try to eliminate term t0 by expressing its vector as a linear combination of others.
    Returns (new_U, new_V, new_W) if successful, else None."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    if rank <= 1:
        return None
    if time.time() > deadline:
        return None
    
    # We want to find coefficients c_s (for s != t0) such that:
    # U[:, t0] = sum_s c_s * U[:, s]
    # V[:, t0] = sum_s c_s * V[:, s]
    # W[:, t0] = sum_s c_s * W[:, s]
    # AND the tensor identity still holds.
    
    # Actually, the tensor identity is: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for all i,j,k -> i,k
    # If we replace term t0 with sum_s c_s * term_s, the contribution of t0 becomes:
    # sum_s c_s * U[i,j,s] V[j,k,s] W[k,i,s]
    # The new total is: sum_{s != t0} U[i,j,s] V[j,k,s] W[k,i,s] + sum_s c_s * U[i,j,s] V[j,k,s] W[k,i,s]
    # = sum_{s != t0} (1 + c_s) U[i,j,s] V[j,k,s] W[k,i,s] + c_{t0} (if we kept it, but we're removing it)
    # Wait, if we express U[:,t0] = sum_s c_s U[:,s], etc., then the term t0's contribution is:
    # (sum_s c_s U[i,j,s]) * (sum_s' c_s' V[j,k,s']) * (sum_s'' c_s'' W[k,i,s''])
    # This is a triple sum, which is NOT equal to sum_s c_s U[i,j,s]V[j,k,s]W[k,i,s] unless c is a delta.
    # So this approach is flawed. The tensor is a 3-way product, not a linear one.
    
    # Correct approach: We want to find a linear combination of terms that replaces t0.
    # Let the new term be T_new = sum_{s in S} c_s * T_s, where T_s is the tensor of term s.
    # The tensor of term t is U_t ⊗ V_t ⊗ W_t (in the appropriate indices).
    # We need: sum_{s in S} c_s (U_s ⊗ V_s ⊗ W_s) = U_t0 ⊗ V_t0 ⊗ W_t0
    # This is a linear system in c_s, with equations for each (i,j,k) -> (i,k) component.
    # Number of equations: n*p = 20
    # Number of unknowns: rank - 1 (we exclude t0, or include it with coefficient -1)
    # Let's set up: sum_{s != t0} c_s (U_s[i,j] V_s[j,k] W_s[k,i]) = U_t0[i,j] V_t0[j,k] W_t0[k,i] for all i,k
    # But wait, the tensor component is for specific i,j,k. The equation is for each (i,k), summing over j?
    # No, the tensor decomposition is: M_{i,j,k}^{i',j',k'} = sum_t U_{i,j,t} V_{j',k',t} W_{k',i',t}
    # Here i'=i, j'=j, k'=k. So M_{i,j,k}^{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1
    # And M_{i,j,k}^{i',j',k'} = 0 if (i,j,k) != (i',j',k')
    # So the constraints are:
    # 1. For all i,j,k: sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1
    # 2. For all i,j,k,i',j',k' with (i,j,k) != (i',j',k'): sum_t U_{i,j,t} V_{j',k',t} W_{k',i',t} = 0
    
    # This is a huge number of equations. n*m*p = 60 for the "1" constraints, and (n*p)^2 * m - n*m*p for the "0" constraints?
    # Actually, the tensor has n*m*p * n*m*p components.
    # Let's just use the "1" constraints and check the "0" constraints later, or use all.
    
    # Let's stick to the "1" constraints first, as they are the most direct.
    # If we can satisfy the "1" constraints, the "0" constraints might be automatically satisfied if the structure is preserved.
    # But that's not guaranteed.
    
    # Let's set up the linear system for the "1" constraints.
    # Unknowns: c_s for s != t0.
    # Equation for (i,k): sum_{j} sum_{s != t0} c_s U_{i,j,s} V_{j,k,s} W_{k,i,s} = sum_{j} U_{i,j,t0} V_{j,k,t0} W_{k,i,t0}
    # Wait, the LHS is sum_s c_s * (sum_j U_{i,j,s} V_{j,k,s} W_{k,i,s})
    # The RHS is sum_j U_{i,j,t0} V_{j,k,t0} W_{k,i,t0}
    
    # Let's compute the "contribution vector" for each term t:
    # C_t[i,k] = sum_j U_{i,j,t} V_{j,k,t} W_{k,i,t}
    # Then the constraint is: sum_t C_t[i,k] = 1 for all i,k.
    # If we replace t0 with sum_s c_s T_s, the new contribution is sum_s c_s C_s.
    # So we need: sum_{s != t0} C_s[i,k] + sum_s c_s C_s[i,k] = 1
    # => sum_s c_s C_s[i,k] = 1 - sum_{s != t0} C_s[i,k] = C_t0[i,k] (since sum_all C_t = 1)
    # So we need: sum_{s != t0} c_s C_s[i,k] = C_t0[i,k] for all i,k.
    # This is a linear system in c_s, with n*p equations.
    
    # Let's compute C_t for all t.
    C = np.zeros((rank, n, p), dtype=object)
    for t in range(rank):
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                C[t, i, k] = s
    
    # Now set up the system: A * c = b
    # A[i,k, s] = C[s, i, k] for s != t0
    # b[i,k] = C[t0, i, k]
    
    # Number of equations: n*p = 20
    # Number of unknowns: rank - 1
    
    # We need at least as many unknowns as equations to have a solution, but we want a sparse or simple solution.
    # Let's try to solve with least squares or exact integer solution.
    
    # For now, let's just check if C_t0 is in the span of the other C_s.
    # If yes, we can eliminate t0.
    
    # Build matrix A (20 x (rank-1)) and vector b (20)
    idx = [s for s in range(rank) if s != t0]
    A = np.array([[C[s, i, k] for s in idx] for i in range(n) for k in range(p)], dtype=object)
    b = np.array([C[t0, i, k] for i in range(n) for k in range(p)], dtype=object)
    
    # Solve A * c = b
    # Since we're using object dtype, we can't use np.linalg.solve directly.
    # Let's convert to float and solve, then round.
    try:
        A_float = np.array(A, dtype=float)
        b_float = np.array(b, dtype=float)
        # Check if b is in the column space of A
        # Use np.linalg.lstsq
        c_float, residuals, rank_A, s = np.linalg.lstsq(A_float, b_float, rcond=None)
        # Check if the solution is exact
        if np.allclose(A_float @ c_float, b_float, atol=1e-9):
            # Round c to nearest integer or half-integer
            c_rounded = np.round(c_float * 2) / 2
            # Verify the rounded solution
            if np.allclose(A_float @ c_rounded, b_float, atol=1e-9):
                # Check if the coefficients are "simple" (e.g., small integers)
                # For now, accept if they are within a reasonable range
                if np.all(np.abs(c_rounded) < 100):
                    # Construct new U, V, W without term t0
                    new_rank = rank - 1
                    new_U = np.zeros((n * m, new_rank), dtype=object)
                    new_V = np.zeros((m * p, new_rank), dtype=object)
                    new_W = np.zeros((p * n, new_rank), dtype=object)
                    new_idx = 0
                    for s in idx:
                        # Add c_s * term_s to the new term
                        new_U[:, new_idx] = c_rounded[new_idx] * U[:, s]
                        new_V[:, new_idx] = c_rounded[new_idx] * V[:, s]
                        new_W[:, new_idx] = c_rounded[new_idx] * W[:, s]
                        new_idx += 1
                    # Verify the full tensor identity
                    if _tensor_residual(new_U, new_V, new_W):
                        return new_U, new_V, new_W
    except Exception:
        pass
    
    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 150s
    
    U, V, W = _schoolbook()
    rank = U.shape[1]
    
    # Iteratively try to eliminate terms
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for t0 in range(rank):
            if time.time() > deadline:
                break
            result = _try_eliminate(U, V, W, t0, deadline)
            if result is not None:
                U, V, W = result
                rank = U.shape[1]
                improved = True
                break  # Restart the loop with the new rank
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
