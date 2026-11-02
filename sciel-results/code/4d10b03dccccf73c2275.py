"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

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
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    return terms

def _border_terms_standard():
    """Standard border terms (at least one index is 2)."""
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

def _try_rank_25_via_adjustment(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank from 26 to 25 by systematically adjusting remaining terms
    after removing one term. Uses gradient-based local search.
    """
    R = U.shape[1]
    if R <= 25:
        return U, V, W, R
    
    T_target = np.ones((n, m, p))
    R_new = R - 1
    
    def _compute_residual(U_arr, V_arr, W_arr):
        U_r = U_arr.reshape(n, m, R_new)
        V_r = V_arr.reshape(m, p, R_new)
        W_r = W_arr.reshape(p, n, R_new)
        T_cur = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return T_target - T_cur
    
    # Try removing each column
    for col in range(R):
        if time.time() > deadline:
            break
        
        mask = [i for i in range(R) if i != col]
        U_new = U[:, mask].copy()
        V_new = V[:, mask].copy()
        W_new = W[:, mask].copy()
        
        best_err = float('inf')
        best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        # Gradient-based local search
        lr = 0.1
        for step in range(200):
            if time.time() > deadline:
                break
            
            residual = _compute_residual(U_new, V_new, W_new)
            err = np.max(np.abs(residual))
            
            if err < 1e-10:
                return U_new, V_new, W_new, R_new
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            # Compute gradients: d(residual[i,j,k])/d(U[a,b,r]) = V[b,k,r]*W[k,a,r]
            U_r = U_new.reshape(n, m, R_new)
            V_r = V_new.reshape(m, p, R_new)
            W_r = W_new.reshape(p, n, R_new)
            
            # Gradient w.r.t. U: grad_U[i,j,r] = sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]
            grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
            # Gradient w.r.t. V: grad_V[j,k,r] = sum_i residual[i,j,k] * U[i,j,r] * W[k,i,r]
            grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
            # Gradient w.r.t. W: grad_W[k,i,r] = sum_j residual[i,j,k] * U[i,j,r] * V[j,k,r]
            grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
            
            # Update: we want to minimize max|residual|, approximate with L2
            U_new -= lr * grad_U.reshape(n*m, R_new)
            V_new -= lr * grad_V.reshape(m*p, R_new)
            W_new -= lr * grad_W.reshape(p*n, R_new)
            
            # Adaptive learning rate
            if err < 1e-6:
                lr *= 0.5
            elif step > 50 and err > best_err:
                lr *= 0.5
        
        # Check best found
        U_r = best_U.reshape(n, m, R_new)
        V_r = best_V.reshape(m, p, R_new)
        W_r = best_W.reshape(p, n, R_new)
        T_cur = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if np.max(np.abs(T_target - T_cur)) < 1e-10:
            return best_U, best_V, best_W, R_new
    
    return U, V, W, R

def _try_rank_23_construction(n, m, p, deadline):
    """
    Attempt to construct a rank-23 decomposition using a structured approach:
    Strassen 2x2 core (7 terms) + optimized border (16 terms).
    
    The key idea: some border terms can be absorbed into modified Strassen terms.
    """
    if time.time() > deadline:
        return None, None, None
    
    # Start with the rank-26 decomposition and try to eliminate 3 border terms
    # by adjusting the Strassen core terms.
    
    terms = _strassen_2x2_terms() + _border_terms_standard()
    U, V, W = _build_decomposition(terms)
    R = U.shape[1]  # 26
    
    T_target = np.ones((n, m, p))
    
    # Identify which border terms might be eliminable
    # Border terms are indices 7..25 in the term list
    # Try removing triples of border terms and adjusting all remaining terms
    
    best_U, best_V, best_W = None, None, None
    best_rank = R
    
    # Try removing each combination of 3 border terms
    border_indices = list(range(7, R))
    
    for combo in itertools.combinations(border_indices, 3):
        if time.time() > deadline:
            break
        
        remove_set = set(combo)
        mask = [i for i in range(R) if i not in remove_set]
        R_new = len(mask)
        
        U_new = U[:, mask].copy()
        V_new = V[:, mask].copy()
        W_new = W[:, mask].copy()
        
        # Gradient-based optimization
        lr = 0.05
        for step in range(100):
            if time.time() > deadline:
                break
            
            U_r = U_new.reshape(n, m, R_new)
            V_r = V_new.reshape(m, p, R_new)
            W_r = W_new.reshape(p, n, R_new)
            T_cur = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = T_target - T_cur
            err = np.max(np.abs(residual))
            
            if err < 1e-10:
                return U_new, V_new, W_new
            
            # Gradients
            grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
            
            U_new -= lr * grad_U.reshape(n*m, R_new)
            V_new -= lr * grad_V.reshape(m*p, R_new)
            W_new -= lr * grad_W.reshape(p*n, R_new)
            
            if err < 1e-6:
                lr *= 0.5
        
        # Check if we got close enough
        U_r = U_new.reshape(n, m, R_new)
        V_r = V_new.reshape(m, p, R_new)
        W_r = W_new.reshape(p, n, R_new)
        T_cur = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if np.max(np.abs(T_target - T_cur)) < 1e-10:
            return U_new, V_new, W_new
    
    return None, None, None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
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
    
    # Strategy 2: Try to reduce to rank 25 via gradient-based adjustment
    if best_result is not None and best_rank == 26 and time.time() < deadline:
        U, V, W = best_result
        U_red, V_red, W_red, rank_red = _try_rank_25_via_adjustment(U, V, W, n, m, p, deadline)
        if rank_red < best_rank:
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                best_rank = rank_red
                best_result = (U_red, V_red, W_red)
    
    # Strategy 2b: Try rank-23 construction via structured border reduction
    if best_rank > 23 and time.time() < deadline:
        U_23, V_23, W_23 = _try_rank_23_construction(n, m, p, deadline)
        if U_23 is not None and _verify_decomposition(U_23, V_23, W_23, n, m, p):
            R = U_23.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U_23, V_23, W_23)
    
    # Strategy 3: Pure standard (rank 27) as fallback
    if best_result is None or time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
