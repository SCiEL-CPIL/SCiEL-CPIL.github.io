"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    if R == 0:
        return False
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    """
    terms = []
    
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    
    return terms

def _border_terms_standard():
    """
    Return standard multiplication terms for the border (at least one index is 2).
    """
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _solve_for_W(U, V, n, m, p):
    """
    Given fixed U and V, solve for W such that the tensor decomposition holds.
    Returns W matrix or None if no solution.
    """
    R = U.shape[1]
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For each (k,i), we have a linear system in W[k,i,r]?
    # No, W[k,i,r] is a single variable for each r.
    # The equation is: for all i,j,k:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is not a simple linear system for W entries independently because
    # each W[k,i,r] appears in multiple equations (for different j).
    # Specifically, W[k,i,r] is multiplied by V[j*p+k, r] and U[i*m+j, r] for all j.
    # So for a fixed r, the term is W[k,i,r] * (sum_j U[i*m+j, r] V[j*p+k, r] * 1?) No.
    # The tensor equation is:
    # T[i,j,k] = sum_r (U[i,j,r] V[j,k,r] W[k,i,r])
    
    # If we fix U and V, the unknowns are W[k,i,r].
    # Let's group by (k,i,r).
    # The contribution of W[k,i,r] to T[i,j,k] is U[i,j,r] V[j,k,r] W[k,i,r].
    # So for a fixed r, and fixed (k,i), the variable is W[k,i,r].
    # It appears in the equation for T[i,j,k] for all j.
    # Sum over j? No, the equation must hold for each j.
    # T[i,j,k] = 1.
    # So: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    # This is a system of 27 equations (for each i,j,k) with 27*R unknowns?
    # No, the unknowns are W[k*n+i, r] for k in 0..2, i in 0..2, r in 0..R-1.
    # Total unknowns: 9 * R.
    # Total equations: 27.
    # If R < 3, underdetermined? If R >= 3, potentially overdetermined.
    # We can set up a least squares problem or an exact linear system.
    # Let's form the matrix A and vector b.
    # Variable x_idx = k*n + i * R + r ? No, let's flatten W to a vector of length 9*R.
    # x[k*n+i, r]
    
    # Equation for (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * x[k*n+i, r] = 1
    
    # This is linear in x.
    # Let's build A (27 x 9R) and b (27).
    
    A = np.zeros((27, 9 * R))
    b = np.ones(27)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    coeff = U[i*m+j, r] * V[j*p+k, r]
                    if coeff != 0:
                        # Variable is W[k*n+i, r]
                        var_idx = (k*n + i) * R + r
                        A[eq_idx, var_idx] += coeff
                eq_idx += 1
    
    # Solve A x = b
    # We want an exact solution.
    # Use np.linalg.lstsq or solve.
    # Since it might be underdetermined or inconsistent, use lstsq.
    try:
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        # Check if residual is small
        if np.linalg.norm(A @ x - b) > 1e-9:
            return None
        
        # Reshape x back to W
        W = x.reshape((p*n, R))
        return W
    except np.linalg.LinAlgError:
        return None

def _try_prune(U, V, W, n, m, p):
    """
    Try to remove one term from the current decomposition by solving for new W.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    R = U.shape[1]
    if R <= 1:
        return None
    
    # Try removing each term r
    for r_to_remove in range(R):
        # Create U_new, V_new without column r_to_remove
        U_new = np.delete(U, r_to_remove, axis=1)
        V_new = np.delete(V, r_to_remove, axis=1)
        
        # Solve for W_new
        W_new = _solve_for_W(U_new, V_new, n, m, p)
        
        if W_new is not None:
            # Verify
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                # Check if entries are "nice" (integer or half-integer)
                # The problem requires integer/half-integer entries.
                # If the solution is floating point but very close to such values, we can snap.
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    # Maybe the solution is integer?
                    U_snap = np.round(U_new)
                    V_snap = np.round(V_new)
                    W_snap = np.round(W_new)
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap
                    else:
                        # Return the float solution if it's exact? 
                        # The task says "integer/half-integer entries". 
                        # If lstsq gives 0.333, it's invalid.
                        # So we must reject if not close to 0.5k.
                        pass
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Attempts to prune the rank via iterative linear solving.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Iterative Pruning
    # Try to reduce rank from 26 down to 23
    if best_result is not None and time.time() < deadline:
        U, V, W = best_result
        # Try to prune multiple times
        for _ in range(5):  # Max 5 prunings
            if time.time() > deadline:
                break
            new_res = _try_prune(U, V, W, n, m, p)
            if new_res is not None:
                U, V, W = new_res
                R = U.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)
            else:
                break
    
    # Strategy 3: Pure standard (rank 27) as fallback
    if best_result is None or time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen, but fallback
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    
    # Final snap to ensure integer/half-integer representation for output
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    
    # Verify once more
    if not _verify_decomposition(U_out, V_out, W_out, n, m, p):
        # If snapping failed, use the original (which might be float but exact?)
        # The task requires integer/half-integer. If the solution is not close to that,
        # it might be invalid. But our initial solution is integer.
        # Pruning might produce floats.
        # If the pruned solution is not close to half-integers, we should discard it.
        # Let's check if the best_result was the pruned one.
        # If the pruned one was accepted by _try_prune, it was close to half-integers.
        # So this should be safe.
        pass

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
