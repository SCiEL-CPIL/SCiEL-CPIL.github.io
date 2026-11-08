"""Optimized Strassen-based decomposition for 2x3x8 matrix multiplication."""
import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Build the rank 44 candidate (Strassen blocks)
    def _build_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen 2x2x2
        S_U = [
            [1, 0, 0, 1], 
            [0, 0, 1, 1], 
            [1, 0, 0, 0], 
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [-1, 0, 1, 0],
            [0, 1, 0, -1] 
        ]
        S_V = [
            [1, 0, 0, 1], 
            [1, 0, 0, 0], 
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [0, 0, 1, 1]  
        ]
        S_W = [
            [1, 0, 0, 1], 
            [0, 1, 0, -1],
            [0, 0, 1, 1], 
            [1, 1, 0, 0], 
            [-1, 0, 1, 0],
            [0, 0, 0, 1], 
            [1, 0, 0, 0]  
        ]

        col = 0
        # Part 1: A1 (2x2) * B1 (2x8) -> 4 blocks of 2x2x2
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li*2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la*2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk*2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
                if not _elapsed_ok():
                    return None

        # Part 2: A2 (2x1) * B2 (1x8) -> 16 schoolbook terms
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if not _elapsed_ok():
                    return None

        return {"U": U, "V": V, "W": W}

    candidate_44 = _build_rank_44()
    if candidate_44 is None:
        return _schoolbook()
        
    if _validate(candidate_44):
        best_candidate = candidate_44
    else:
        return _schoolbook()

    # Attempt to reduce rank using numerical optimization
    # We try to find a rank 43 solution
    target_rank = 43
    if _elapsed_ok():
        U44 = np.array(candidate_44["U"], dtype=float)
        V44 = np.array(candidate_44["V"], dtype=float)
        W44 = np.array(candidate_44["W"], dtype=float)
        
        # Remove the last column (index 43)
        U_init = U44[:, :target_rank].flatten()
        V_init = V44[:, :target_rank].flatten()
        W_init = W44[:, :target_rank].flatten()
        
        x0 = np.concatenate([U_init, V_init, W_init])
        
        def loss(x):
            U = x[:U_init.size].reshape(n*m, target_rank)
            V = x[U_init.size:U_init.size+V_init.size].reshape(m*p, target_rank)
            W = x[U_init.size+V_init.size:].reshape(p*n, target_rank)
            
            # Calculate error: sum_c U_ia,c V_ak,c W_ki,c - 1
            # We can vectorize this using einsum
            # T_ia,ak,ki = sum_c U_ia,c V_ak,c W_ki,c
            # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # Contract over c
            # Result shape: (n, m, p, n)
            # We want sum_c U[n,m,c] V[m,p,c] W[p,n,c]
            # einsum('nmc,mpc,pnc->npmn')
            T = np.einsum('nmc,mpc,pnc->npmn', U_r, V_r, W_r)
            
            # The target is 1 for all n,m,p? 
            # Wait, the tensor T_ia,ak,ki should be 1 for all i,a,k.
            # So the target is a tensor of ones with shape (n, m, p, n)?
            # No, the indices are i, a, k. The result of the sum is for each i, a, k.
            # So T has shape (n, m, p). The last index 'n' in my einsum is 'i' again?
            # U is (i, a, c). V is (a, k, c). W is (k, i, c).
            # Sum over c gives T(i, a, k, i)? No.
            # U_r[n, m, c] is U_{i, a, c}.
            # V_r[m, p, c] is V_{a, k, c}.
            # W_r[p, n, c] is W_{k, i, c}.
            # The product is U_{i,a,c} V_{a,k,c} W_{k,i,c}.
            # Sum over c gives S_{i,a,k,i}.
            # But the tensor decomposition condition is:
            # For each i, a, k: sum_c U_{i,a,c} V_{a,k,c} W_{k,i,c} = 1.
            # Note that W is indexed by k, i.
            # So the sum depends on i, a, k.
            # My einsum 'nmc,mpc,pnc->npmn' produces an array of shape (n, m, p, n).
            # The indices are i, a, k, i.
            # So T[i, a, k, i] should be 1?
            # Yes, for a fixed i, a, k, the value is T[i, a, k, i].
            # So we want T[i, a, k, i] = 1 for all i, a, k.
            # This means we take the diagonal over the first and last axis?
            # Or just check T[i, a, k, i].
            
            # Extract the diagonal
            # T has shape (2, 3, 8, 2).
            # We want T[i, a, k, i].
            # This is not a standard diagonal if n != p? Here n=2, p=8.
            # We can just reshape and sum?
            # No, we need the specific entries.
            # Let's create a target tensor of ones with shape (n, m, p, n) and mask?
            # Actually, we can just compute the error for the specific indices.
            # err = 0
            # for i in range(n):
            #   for a in range(m):
            #     for k in range(p):
            #       err += (T[i,a,k,i] - 1)**2
            
            # Vectorized:
            # Create indices
            # i = np.arange(n)
            # We want T[i, a, k, i].
            # We can use np.einsum again?
            # Or just take the diagonal of the first and last axis.
            # np.diagonal(T, axis1=0, axis2=3) gives shape (m, p, n)? No.
            # np.diagonal(T, axis1=0, axis2=3) on (n,m,p,n) gives (m,p,n)?
            # Let's check: T[0, :, :, 0], T[1, :, :, 1].
            # Yes, it gives an array of shape (m, p, n) ? No, (m, p) with i removed?
            # If we remove axis 0 and 3, we are left with m, p.
            # But the values are T[i, a, k, i].
            # So diag = np.diagonal(T, axis1=0, axis2=3) has shape (m, p, n) ?
            # No, diagonal removes the two axes. So shape is (m, p).
            # Wait, T is (n, m, p, n).
            # diagonal(axis1=0, axis2=3) -> shape (m, p).
            # The values are T[i, a, k, i].
            # So we want diag[a, k] = 1?
            # But diag depends on i?
            # No, diagonal takes T[i, a, k, i] for each i.
            # If n=2, we get two arrays?
            # No, np.diagonal returns an array where the diagonal elements are stacked.
            # If axis1=0 and axis2=3, the resulting array has shape (m, p, n) ?
            # Let's check numpy docs.
            # "The diagonal is returned as an array with the same shape as the input, with the diagonal elements removed?"
            # No, "The diagonal is returned as a 1-D array"?
            # If axis1 and axis2 are specified, the result has the remaining dimensions.
            # So shape (m, p) is incorrect if n > 1.
            # Actually, if T is (2, 3, 8, 2), diagonal(0, 3) gives shape (3, 8, 2) ?
            # No, it gives shape (3, 8) ?
            # Let's test mentally.
            # T[i, a, k, i].
            # If we fix i, we get a 3x8 matrix.
            # So the result should be (2, 3, 8).
            # Yes, the remaining dimensions are m, p. And the diagonal index i is kept?
            # No, the diagonal operation removes the two axes.
            # So if T is (n, m, p, n), diagonal(0, 3) removes axis 0 and 3.
            # The result has shape (m, p).
            # But this loses the information of which i was used?
            # No, it takes T[0, :, :, 0] and T[1, :, :, 1] and... what?
            # It can't stack them if the shapes don't match?
            # Actually, np.diagonal returns an array of shape (n, m, p) ?
            # No, it returns an array where the diagonal elements are extracted.
            # If the two axes have the same size, the result has the size of that axis in the new position?
            # No, the result has the remaining dimensions.
            # So if T is (2, 3, 8, 2), diagonal(0, 3) has shape (3, 8).
            # This implies it only takes one diagonal?
            # No, it takes the diagonal for each "slice" of the remaining dimensions?
            # Actually, it returns an array of shape (m, p, n) ?
            # Let's assume it returns (n, m, p) or similar.
            # To be safe, I'll just loop. It's small (48 elements).
            
            err = 0.0
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        val = T[i, a, k, i]
                        err += (val - 1.0)**2
            return err

        # Optimize
        res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-8})
        
        if _elapsed_ok():
            x_opt = res.x
            U_opt = x_opt[:U_init.size].reshape(n*m, target_rank)
            V_opt = x_opt[U_init.size:U_init.size+V_init.size].reshape(m*p, target_rank)
            W_opt = x_opt[U_init.size+V_init.size:].reshape(p*n, target_rank)
            
            # Round to nearest 0.5
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Convert to list of lists of Fractions/Ints
            def to_list(arr):
                return [[(float(x) if x != int(x) else int(x)) for x in row] for row in arr]
            
            candidate_43 = {
                "U": to_list(U_rounded),
                "V": to_list(V_rounded),
                "W": to_list(W_rounded)
            }
            
            if _validate(candidate_43):
                return candidate_43

    return best_candidate

# EVOLVE-BLOCK-END
