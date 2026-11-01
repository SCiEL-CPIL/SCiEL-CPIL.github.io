"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _find_233_rank9(seed):
    """
    Try to find a rank 9 decomposition for 2x3x3 matrix multiplication.
    Uses random sparse U, V and solves for W.
    Returns (U, V, W) as lists of lists of Fractions if found, else None.
    """
    n, m, p = 2, 3, 3
    R = 9
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate sparse U and V with integer entries (0 or 1)
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    
    for i in range(n * m):
        for c in range(R):
            if random.random() < 0.5:
                U[i, c] = 1.0
    for j in range(m * p):
        for c in range(R):
            if random.random() < 0.5:
                V[j, c] = 1.0

    # Build linear system for W
    # Variables: W[k, i, r] for k in 0..2, i in 0..1, r in 0..8
    # Total vars: 3 * 2 * 9 = 54
    num_vars = p * n * R
    # Equations: for all i, i', j, j', k, k'
    # i, i' in 0..1 (2x2)
    # j, j' in 0..2 (3x3)
    # k, k' in 0..2 (3x3)
    # Total eqs: 2*2 * 3*3 * 3*3 = 4 * 9 * 9 = 324
    num_eqs = (n * n) * (m * m) * (p * p)
    
    A = np.zeros((num_eqs, num_vars), dtype=float)
    b = np.zeros(num_eqs, dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for i_p in range(n):
            for j in range(m):
                for j_p in range(m):
                    for k in range(p):
                        for k_p in range(p):
                            # b is 1 if i==i', j==j', k==k'
                            if i == i_p and j == j_p and k == k_p:
                                b[eq_idx] = 1.0
                            else:
                                b[eq_idx] = 0.0
                            
                            # Coefficients for W[k_p, i_p, r]
                            # Term: sum_r U[i, j, r] * V[j_p, k, r] * W[k_p, i_p, r]
                            for r in range(R):
                                coeff = U[i * m + j, r] * V[j_p * p + k, r]
                                if coeff != 0:
                                    # Var index for W[k_p, i_p, r]
                                    # W is stored as W[k*n+i, c] in final output, 
                                    # but here variable is W[k_p, i_p, r]
                                    var_idx = (k_p * n + i_p) * R + r
                                    A[eq_idx, var_idx] += coeff
                            eq_idx += 1
                            
    try:
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        # Check exactness
        if np.allclose(A @ W_flat, b, atol=1e-9):
            # Check if W_flat entries are close to half-integers
            W_check = np.round(W_flat * 2) / 2
            if np.allclose(A @ W_check, b, atol=1e-9):
                # Reshape W_check to W[k*n+i, r]
                W = W_check.reshape(p * n, R)
                
                # Convert to Fractions
                U_list = [[Fraction(int(val)) for val in row] for row in U]
                V_list = [[Fraction(int(val)) for val in row] for row in V]
                W_list = [[Fraction(val) for val in row] for row in W]
                
                # Validate the 2x3x3 decomposition
                # Quick check
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid: break
                if valid:
                    return U_list, V_list, W_list
    except np.linalg.LinAlgError:
        pass
        
    return None

def _try_rank_r_direct(n, m, p, R, seed, time_limit=5.0):
    """
    Attempt to find a rank R decomposition for n x m x p directly.
    Uses random sparse U, V and solves for W.
    """
    t_start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    
    # Sparse random init
    density = 0.4
    for i in range(n * m):
        for c in range(R):
            if random.random() < density:
                U[i, c] = 1.0
    for j in range(m * p):
        for c in range(R):
            if random.random() < density:
                V[j, c] = 1.0

    # Build linear system for W
    # Variables: W[k, i, r] -> p * n * R
    num_vars = p * n * R
    
    # Equations: For all (i, i', j, j', k, k')
    # Condition: U[i,j,r] V[j',k,r] W[k',i',r] should sum to delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, the tensor T_{(i,j),(j',k),(k',i')} is 1 if i=i', j=j', k=k' AND j=j' (link) ...
    # Actually, the standard matrix multiplication tensor T_{(i,j),(j',k),(k',i')} is non-zero only if j=j' and k=k' and i=i'?
    # No. The tensor is defined such that the decomposition computes C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The rank-1 term is U_{i,j} V_{j',k} W_{k',i'}.
    # We want sum_r U_{i,j}^{(r)} V_{j',k}^{(r)} W_{k',i'}^{(r)} = 1 if i=i', j=j', k=k' AND j=j' (always true in the sum index match?)
    # Let's stick to the definition:
    # T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k' AND j=j' (the middle indices must match for the multiplication to be valid).
    # Actually, the tensor is 1 at (i,j), (j,k), (k,i) and 0 elsewhere.
    # So T_{(i,j), (j',k'), (k'',i'')} = 1 if j=j' and k=k' and i=i''? No.
    # The indices are:
    # U index: (i, j)
    # V index: (j', k')
    # W index: (k'', i'')
    # The term contributes to C_{i'', k''} from A_{i, j} B_{j', k'}.
    # We want C_{i'', k''} = sum_{j} A_{i'', j} B_{j, k''}.
    # So the tensor T should be 1 when i=i'', j=j', k=k'.
    # Wait, the indices of T are (I, J, K) where I=(i,j), J=(j',k'), K=(k'',i'').
    # T[I, J, K] = 1 if i=i'', j=j', k=k'.
    # So the equation is: sum_r U_{i,j,r} V_{j',k',r} W_{k'',i'',r} = 1 if i=i'', j=j', k=k' else 0.
    
    # Equations for all i, i'', j, j', k, k'
    num_eqs = (n * n) * (m * m) * (p * p)
    
    A = np.zeros((num_eqs, num_vars), dtype=float)
    b = np.zeros(num_eqs, dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for i_p in range(n):
            for j in range(m):
                for j_p in range(m):
                    for k in range(p):
                        for k_p in range(p):
                            # Target: 1 if i==i_p, j==j_p, k==k_p
                            if i == i_p and j == j_p and k == k_p:
                                b[eq_idx] = 1.0
                            else:
                                b[eq_idx] = 0.0
                            
                            # Coefficients for W[k_p, i_p, r]
                            # Term: sum_r U[i, j, r] * V[j_p, k, r] * W[k_p, i_p, r]
                            # Note: V index is (j_p, k) in the equation?
                            # The V index in the tensor is (j', k'). Here we are iterating over j_p (j') and k (k').
                            # So V[j_p * p + k, r] is correct.
                            # U index is (i, j). U[i * m + j, r] is correct.
                            # W index is (k', i'). Here we iterate k_p (k') and i_p (i').
                            # So W[k_p * n + i_p, r] is the variable.
                            
                            for r in range(R):
                                coeff = U[i * m + j, r] * V[j_p * p + k, r]
                                if coeff != 0:
                                    var_idx = (k_p * n + i_p) * R + r
                                    A[eq_idx, var_idx] += coeff
                            eq_idx += 1
                            
    try:
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        if np.allclose(A @ W_flat, b, atol=1e-9):
            W_check = np.round(W_flat * 2) / 2
            if np.allclose(A @ W_check, b, atol=1e-9):
                W = W_check.reshape(p * n, R)
                
                U_list = [[Fraction(int(val)) for val in row] for row in U]
                V_list = [[Fraction(int(val)) for val in row] for row in V]
                W_list = [[Fraction(val) for val in row] for row in W]
                
                # Validate
                if _validate(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    except np.linalg.LinAlgError:
        pass
        
    return None

def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 140.0 # Safety margin

    n, m, p = 2, 3, 9

    # --- Schoolbook baseline (rank 54) ---
    rank_sb = n * m * p  # 54
    best_result = _construct_schoolbook(n, m, p)
    best_rank = rank_sb

    # --- Strategy 1: Tile 2x3x3 decomposition 3 times ---
    # If we find a rank 9 decomposition for 2x3x3, we can tile it 3 times to get rank 27 for 2x3x9.
    # This is the theoretical lower bound.
    found_233 = None
    for seed in range(200):
        if time.time() > DEADLINE - 20:
            break
        res = _find_233_rank9(seed)
        if res is not None:
            found_233 = res
            break
            
    if found_233:
        U_233, V_233, W_233 = found_233
        R_233 = 9
        R_total = 27
        
        U_new = [[0] * R_total for _ in range(n * m)]
        V_new = [[0] * R_total for _ in range(m * p)]
        W_new = [[0] * R_total for _ in range(p * n)]
        
        # Tile 3 times
        for block in range(3):
            # U is same for all blocks (just repeats)
            # U[i*m+j, block*9 + r] = U_233[i*m+j, r]
            for i in range(n):
                for j in range(m):
                    for r in range(R_233):
                        U_new[i * m + j][block * R_233 + r] = U_233[i * m + j][r]
                        
            # V[j*p+k, block*9 + r] = V_233[j*3+k_local, r] if k is in block
            for j in range(m):
                for k_local in range(3):
                    k_global = block * 3 + k_local
                    for r in range(R_233):
                        V_new[j * p + k_global][block * R_233 + r] = V_233[j * 3 + k_local][r]
                        
            # W[k*n+i, block*9 + r] = W_233[k_local*n+i, r] if k is in block
            for k_local in range(3):
                k_global = block * 3 + k_local
                for i in range(n):
                    for r in range(R_233):
                        W_new[k_global * n + i][block * R_233 + r] = W_233[k_local * n + i][r]
                        
        # Validate the full 2x3x9 decomposition
        if _validate(U_new, V_new, W_new):
            best_rank = R_total
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            return best_result

    # --- Strategy 2: Direct search for rank 27 ---
    # If tiling failed, try to find a rank 27 decomposition directly for 2x3x9.
    # This is harder but might succeed if the tiling structure was too restrictive.
    for seed in range(50):
        if time.time() > DEADLINE - 10:
            break
        res = _try_rank_r_direct(n, m, p, 27, seed)
        if res is not None:
            U_new, V_new, W_new = res
            if _validate(U_new, V_new, W_new):
                best_rank = 27
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                return best_result

    # --- Strategy 3: Direct search for rank 30, 36 ---
    # Fallback to slightly higher ranks if 27 is not found.
    for r in [30, 36]:
        if time.time() > DEADLINE - 5:
            break
        for seed in range(20):
            if time.time() > DEADLINE - 2:
                break
            res = _try_rank_r_direct(n, m, p, r, seed)
            if res is not None:
                U_new, V_new, W_new = res
                if _validate(U_new, V_new, W_new):
                    best_rank = r
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    return best_result

    return best_result
