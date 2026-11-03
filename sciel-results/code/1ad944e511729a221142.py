"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural idea:
  Split A (2x3) into A0 (2x1, j=0) and A2 (2x2, j=1,2).
  - A0 * B0 (2x1 * 1x10): naive, 2*1*10 = 20 multiplications.
  - A2 * B2 (2x2 * 2x10): split B2 into 5 blocks of 2x2, apply Strassen (7 mults each) = 35.
  Total rank = 55.

Refinement:
  Try to reduce rank below 55 by:
  1. Merging Strassen terms across different B-blocks
  2. Local optimization of coefficients
  3. Random linear combinations of existing terms
  4. Optimized block-based construction
  5. Faster verification using matrix multiplication
  6. ALS with tensor structure
  7. L-BFGS-B gradient-based optimization with bilinear structure exploitation
"""

import numpy as np
import random
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using matrix multiplication.
    For each (i, k), check that U[i,:,r] * W[k,i,r] * V[:,k,r] sums to I_m.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            # M_ik[r, j] = sum_r U[i, j, r] * W[k, i, r] * V[j, k, r]
            # We want M_ik = I_m
            # M_ik[j1, j2] = sum_r U[i, j1, r] * W[k, i, r] * V[j2, k, r]
            # = (U[i] * diag(W[k,i])) @ V[:,k].T
            Uw = U_mat[i] * W_mat[k, i]  # shape (m, rank)
            M_ik = Uw @ V_mat[:, k].T  # shape (m, m)
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(1000):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _check_validity_residual(U, V, W, n, m, p, deadline, tol=1e-6):
    """Compute residual error for a decomposition."""
    R = U.shape[1]
    err = 0.0
    V_mat = V.reshape(m, p, R)
    U_mat = U.reshape(n, m, R)
    W_mat = W.reshape(p, n, R)
    
    for i in range(n):
        if time.time() > deadline:
            return float('inf')
        for k in range(p):
            U_i = U_mat[i, :, :]
            V_k = V_mat[:, k, :]
            W_k_i = W_mat[k, i, :]
            M_ik = U_i @ np.diag(W_k_i) @ V_k.T
            diff = M_ik - np.eye(m)
            err += np.sum(diff**2)
    return err


def _try_merge_strassen_terms(U55, V55, W55, n, m, p, target_rank, deadline):
    """
    Try to merge pairs of Strassen terms from different B-blocks.
    """
    rank55 = 55
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    for drop_idx in range(rank55):
        if time.time() > deadline:
            return None
        
        keep_idx = [i for i in range(rank55) if i != drop_idx]
        U_keep = U55[:, keep_idx]
        V_keep = V55[:, keep_idx]
        W_keep = W55[:, keep_idx]
        
        n_eq = min(500, (n*m) * (m*p) * (p*n))
        random.seed(42)
        a_indices = [random.randint(0, n*m-1) for _ in range(n_eq)]
        b_indices = [random.randint(0, m*p-1) for _ in range(n_eq)]
        c_indices = [random.randint(0, p*n-1) for _ in range(n_eq)]
        
        A = np.zeros((n_eq, 54), dtype=np.float64)
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            for r in range(54):
                A[eq, r] = U_keep[a, r] * V_keep[b, r] * W_keep[c, r]
        
        b_vec = np.zeros(n_eq, dtype=np.float64)
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                b_vec[eq] = 1.0
            b_vec[eq] -= U55[a, drop_idx] * V55[b, drop_idx] * W55[c, drop_idx]
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except Exception:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        U_new = np.zeros((n*m, 54), dtype=np.float64)
        V_new = np.zeros((m*p, 54), dtype=np.float64)
        W_new = np.zeros((p*n, 54), dtype=np.float64)
        for r in range(54):
            U_new[:, r] = coeffs[r] * U_keep[:, r]
            V_new[:, r] = coeffs[r] * V_keep[:, r]
            W_new[:, r] = coeffs[r] * W_keep[:, r]
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _als_search(n, m, p, rank, deadline, U_init, V_init, W_init):
    """
    ALS-based search for rank decomposition, initialized near a known solution.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    T_flat = T.reshape(nm, mp * pn)
    T_flat_v = T.reshape(mp, nm * pn)
    T_flat_w = T.reshape(pn, nm * mp)
    
    max_restarts = 15
    for trial in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(trial * 1000 + rank)
        
        # Initialize near the known solution
        if rank <= 55:
            U_c = U_init[:, :rank].copy() + 0.1 * np.random.randn(nm, rank)
            V_c = V_init[:, :rank].copy() + 0.1 * np.random.randn(mp, rank)
            W_c = W_init[:, :rank].copy() + 0.1 * np.random.randn(pn, rank)
        else:
            U_c = np.random.randn(nm, rank)
            V_c = np.random.randn(mp, rank)
            W_c = np.random.randn(pn, rank)
        
        # Alternating least squares
        for it in range(20):
            if time.time() > deadline:
                break
            
            # Solve for U
            M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, rank)
            try:
                U_c = T_flat @ np.linalg.pinv(M.T)
            except Exception:
                break
            
            # Solve for V
            N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, rank)
            try:
                V_c = T_flat_v @ np.linalg.pinv(N.T)
            except Exception:
                break
            
            # Solve for W
            P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, rank)
            try:
                W_c = T_flat_w @ np.linalg.pinv(P.T)
            except Exception:
                break
        
        # Check convergence
        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
        error = np.max(np.abs(T_check - T))
        
        if error < 1e-8:
            # Try to round to half-integers
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            
            T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.max(np.abs(T_check2 - T)) < 1e-10:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
            else:
                # Use unrounded if close enough
                if error < 1e-10:
                    return U_c.tolist(), V_c.tolist(), W_c.tolist()
    
    return None


def _lbfgs_search(n, m, p, rank, deadline, U_init, V_init, W_init):
    """
    L-BFGS-B gradient-based optimization exploiting bilinear structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    U_init = np.array(U_init, dtype=np.float64)
    V_init = np.array(V_init, dtype=np.float64)
    W_init = np.array(W_init, dtype=np.float64)
    
    max_restarts = 10
    for trial in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(trial * 100 + rank)
        
        # Initialize near the known solution with small perturbation
        if rank <= 55:
            U_c = U_init[:, :rank].copy() + 0.05 * np.random.randn(nm, rank)
            V_c = V_init[:, :rank].copy() + 0.05 * np.random.randn(mp, rank)
            W_c = W_init[:, :rank].copy() + 0.05 * np.random.randn(pn, rank)
        else:
            U_c = np.random.randn(nm, rank) * 0.5
            V_c = np.random.randn(mp, rank) * 0.5
            W_c = np.random.randn(pn, rank) * 0.5
        
        params = np.concatenate([U_c.flatten(), V_c.flatten(), W_c.flatten()])
        
        def objective_and_grad(params):
            if time.time() > deadline - 5:
                return float('inf'), np.zeros_like(params)
            
            U_p = params[:nm*rank].reshape(n, m, rank)
            V_p = params[nm*rank:nm*rank + mp*rank].reshape(m, p, rank)
            W_p = params[nm*rank + mp*rank:].reshape(p, n, rank)
            
            err = 0.0
            grad_U = np.zeros_like(U_p)
            grad_V = np.zeros_like(V_p)
            grad_W = np.zeros_like(W_p)
            
            for i in range(n):
                for k in range(p):
                    U_i = U_p[i]  # shape (m, rank)
                    V_k = V_p[:, k]  # shape (m, rank)
                    W_ki = W_p[k, i]  # shape (rank,)
                    
                    M_ik = (U_i * W_ki) @ V_k.T
                    diff = M_ik - np.eye(m)
                    err += np.sum(diff**2)
                    
                    # Gradient computation
                    dM_dU = np.diag(W_ki) @ V_k.T  # shape (m, m)
                    dM_dV = U_i @ np.diag(W_ki)  # shape (m, m)
                    
                    grad_U[i] = 2 * (diff @ dM_dU.T)  # shape (m, rank)
                    grad_V[:, k] = 2 * (dM_dV @ diff.T)  # shape (m, rank)
                    
                    for r in range(rank):
                        dM_dWr = np.outer(U_i[:, r], V_k[:, r])  # shape (m, m)
                        grad_W[k, i, r] += 2 * np.sum(diff * dM_dWr)
            
            return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res = minimize(objective_and_grad, params, jac=True, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < 1e-8:
                U_sol = res.x[:nm*rank].reshape(n*m, rank)
                V_sol = res.x[nm*rank:nm*rank + mp*rank].reshape(m*p, rank)
                W_sol = res.x[nm*rank + mp*rank:].reshape(p*n, rank)
                
                # Verify
                err = _check_validity_residual(U_sol, V_sol, W_sol, n, m, p, deadline)
                if err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    err_r = _check_validity_residual(U_r, V_r, W_r, n, m, p, deadline)
                    if err_r < 1e-10:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    else:
                        # Use unrounded if close enough
                        if err < 1e-10:
                            return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        except Exception:
            pass
    
    return None


def _targeted_search(n, m, p, rank, deadline, values=None):
    """
    Targeted search for a rank-`rank` decomposition.
    """
    if values is None:
        values = [0, 0.5, 1, -0.5, -1]

    U55, V55, W55 = _build_strassen55(n, m, p)
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)

    if rank <= 54:
        result = _try_merge_strassen_terms(U55, V55, W55, n, m, p, rank, deadline)
        if result is not None:
            return result

    for trial in range(200):
        if time.time() > deadline:
            break

        idx = random.sample(range(55), rank)
        U_trial = U55[:, idx].copy()
        V_trial = V55[:, idx].copy()
        W_trial = W55[:, idx].copy()

        if _verify_fast(U_trial, V_trial, W_trial, n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()

    for trial in range(200):
        if time.time() > deadline:
            break

        U_trial = np.zeros((n * m, rank), dtype=np.float64)
        V_trial = np.zeros((m * p, rank), dtype=np.float64)
        W_trial = np.zeros((p * n, rank), dtype=np.float64)

        for r in range(rank):
            n_terms = random.randint(1, 3)
            idx = random.sample(range(55), n_terms)
            coeffs = [random.choice([1, -1]) for _ in range(n_terms)]
            for c, i in zip(coeffs, idx):
                U_trial[:, r] += c * U55[:, i]
                V_trial[:, r] += c * V55[:, i]
                W_trial[:, r] += c * W55[:, i]

        if _verify_fast(U_trial, V_trial, W_trial, n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()

    for trial in range(50):
        if time.time() > deadline:
            break
        
        idx = random.sample(range(55), min(rank, 55))
        U_basis = U55[:, idx]
        V_basis = V55[:, idx]
        W_basis = W55[:, idx]
        
        n_eq = min(500, (n*m) * (m*p) * (p*n))
        a_indices = [random.randint(0, n*m-1) for _ in range(n_eq)]
        b_indices = [random.randint(0, m*p-1) for _ in range(n_eq)]
        c_indices = [random.randint(0, p*n-1) for _ in range(n_eq)]
        
        A = np.zeros((n_eq, len(idx)), dtype=np.float64)
        b_vec = np.zeros(n_eq, dtype=np.float64)
        
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
            
            for r in range(len(idx)):
                A[eq, r] = U_basis[a, r] * V_basis[b, r] * W_basis[c, r]
            b_vec[eq] = target
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except Exception:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        U_new = U_basis * coeffs
        V_new = V_basis * coeffs
        W_new = W_basis * coeffs
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()

    # Try ALS search
    if time.time() < deadline - 10:
        result = _als_search(n, m, p, rank, deadline - 5, U55, V55, W55)
        if result is not None:
            return result

    # Try L-BFGS search
    if time.time() < deadline - 5:
        result = _lbfgs_search(n, m, p, rank, deadline, U55, V55, W55)
        if result is not None:
            return result

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    values = [0, 0.5, 1, -0.5, -1]

    for r in range(54, 49, -1):
        if time.time() > deadline - 15:
            break

        result = _targeted_search(n, m, p, r, deadline - 10)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best, "V": V_best, "W": W_best}
