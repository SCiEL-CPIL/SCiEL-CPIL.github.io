import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the rank-42 Strassen decomposition
    rank_full = 42
    U_full = np.zeros((nm, rank_full), dtype=float)
    V_full = np.zeros((mp, rank_full), dtype=float)
    W_full = np.zeros((pn, rank_full), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset, U, V, W):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        U[a10, offset + 1] += 1.0; U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0; W[w11, offset + 1] -= 1.0
        
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0; V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0; W[w11, offset + 2] += 1.0
        
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0; V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0; W[w10, offset + 3] += 1.0
        
        U[a00, offset + 4] += 1.0; U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0; W[w01, offset + 4] += 1.0
        
        U[a10, offset + 5] += 1.0; U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0; V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        U[a01, offset + 6] += 1.0; U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0; V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U_full, V_full, W_full)
                offset += 7

    def verify(U_mat, V_mat, W_mat, rank, tol=1e-8):
        # Build the full tensor T[a,b,c] = 1 if valid (i,j,k), else 0
        # Then compute sum_r U[a,r]*V[b,r]*W[c,r] for all (a,b,c) and check
        # Vectorized: for each r, add outer(U[:,r], V[:,r]) * W[:,r] to a 3D array
        # But nm*mp*pn = 8*24*12 = 2304, which is small. Let's just do it directly.
        # For each (a,b,c) where the tensor should be 1 or 0, compute the sum.
        
        # Valid entries: (i,j,k) -> a=i*m+j, b=j*p+k, c=k*n+i
        # There are n*m*p = 48 valid entries.
        # Total entries: 8*24*12 = 2304. So 2256 zero entries to check.
        
        # Fast approach: compute the full 3D tensor product sum
        # T_computed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This can be done as: for each r, T += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
        # But that's rank iterations. Alternatively:
        # T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        
        T_computed = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T_computed[a, b, c] - 1.0) > tol:
                        return False
        
        # Check zero entries: build mask of valid entries
        valid_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_mask[a, b, c] = True
        
        # Check all zero entries are close to 0
        zero_entries = ~valid_mask
        max_zero = np.max(np.abs(T_computed[zero_entries]))
        if max_zero > tol:
            return False
        
        return True

    # Check if rank-42 works (should always work)
    if verify(U_full, V_full, W_full, rank_full):
        best_rank = rank_full
        best_U, best_V, best_W = U_full, V_full, W_full
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank))
        best_V = np.zeros((mp, best_rank))
        best_W = np.zeros((pn, best_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to reduce rank by removing factors and solving a linear system.
    # Strategy: Fix W, then for each c, the equation is:
    # sum_t W[c,t] * U[a,t] * V[b,t] = T[a,b,c]
    # If we fix W, this is bilinear in U and V. But if we also fix V, it's linear in U.
    # 
    # Better approach: Use the fact that we have 42 factors from Strassen.
    # Try removing 2 factors. For the remaining 40, fix W to the Strassen values,
    # then solve for U and V using a linear system.
    #
    # For fixed W, the constraint is: for each (a,b,c),
    # sum_t W[c,t] * U[a,t] * V[b,t] = T[a,b,c]
    # 
    # This can be viewed as: for each c, define M_c[a,b] = T[a,b,c]
    # Then sum_t W[c,t] * (U[:,t] outer V[:,t]) = M_c
    # 
    # If we fix V, this is linear in U. Let's fix V to the Strassen V (with removed cols),
    # and solve for U.
    #
    # For fixed V and W, the equation for each (a,b,c) is:
    # sum_t U[a,t] * (W[c,t] * V[b,t]) = T[a,b,c]
    # Let D[b,t] = W[c,t] * V[b,t] -- but this depends on c, so it's not a single matrix.
    #
    # Actually, let's think differently. For each c, we have:
    # sum_t W[c,t] * U[a,t] * V[b,t] = T[a,b,c]
    # For fixed c, this is: (U * diag(W[c,:]) * V^T)[a,b] = T[a,b,c]
    # where U * diag(W[c,:]) means U[:,t] * W[c,t] for each t.
    #
    # So: sum_t W[c,t] * (U[:,t] outer V[:,t]) = T[:,:,c]
    #
    # If we fix V, then for each c, we need:
    # sum_t (W[c,t] * V[:,t]) * U[:,t]^T = T[:,:,c]
    # Let Q[c,t] = W[c,t] * V[:,t] (a vector of length mp)
    # Then: sum_t U[:,t] outer Q[c,t] = T[:,:,c]
    #
    # This is linear in U! For each c, we have nm*mp equations:
    # For each (a,b): sum_t U[a,t] * Q[c,t,b] = T[a,b,c]
    # where Q[c,t,b] = W[c,t] * V[b,t]
    #
    # So for each c, we have a linear system in the nm*r variables U[a,t].
    # But the same U[a,t] appears across different c's. So we need to solve a
    # coupled system.
    #
    # Let's vectorize: stack all equations. For each c, a, b:
    # sum_t U[a,t] * W[c,t] * V[b,t] = T[a,b,c]
    #
    # This is a linear system in U (nm * r unknowns) with nm * mp * pn equations.
    # But many equations are redundant. Let's just set it up and use least squares.
    
    found_better = False
    # Try all C(42,2) = 861 pairs, but in a smart order
    # Prioritize pairs from different blocks
    
    all_pairs = list(combinations(range(42), 2))
    # Sort by: pairs from different blocks first
    def pair_priority(pair):
        b1 = pair[0] // 7
        b2 = pair[1] // 7
        return (0 if b1 != b2 else 1, b1, b2, pair[0], pair[1])
    all_pairs.sort(key=pair_priority)
    
    for remove_pair in all_pairs:
        if time.time() > deadline - 10:
            break
        
        r = 42 - len(remove_pair)
        if r >= best_rank:
            continue
        
        keep_indices = [t for t in range(42) if t not in remove_pair]
        
        # Fix W and V to Strassen values (with removed columns)
        W_fixed = W_full[:, keep_indices]
        V_fixed = V_full[:, keep_indices]
        
        # Build linear system for U
        # Unknowns: U[a,t] for a in [0,nm), t in [0,r)
        # Equations: for each (a,b,c): sum_t U[a,t] * W_fixed[c,t] * V_fixed[b,t] = T[a,b,c]
        
        # Number of unknowns: nm * r
        num_unknowns = nm * r
        
        # We'll build the system A * x = y where x is U flattened
        # For each equation (a,b,c), the row of A has nonzeros at positions (a, t)
        # with value W_fixed[c,t] * V_fixed[b,t]
        
        # Use sparse representation
        rows = []
        cols = []
        vals = []
        rhs = []
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Determine T[a,b,c]
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    for t_idx, t in enumerate(keep_indices):
                        coeff = W_fixed[c, t_idx] * V_fixed[b, t_idx]
                        if abs(coeff) > 1e-15:
                            row_idx = len(rhs)
                            col_idx = a * r + t_idx
                            rows.append(row_idx)
                            cols.append(col_idx)
                            vals.append(coeff)
                    
                    rhs.append(target)
        
        num_eqs = len(rhs)
        
        if num_eqs == 0:
            continue
        
        # Solve least squares
        A = np.zeros((num_eqs, num_unknowns))
        for i in range(len(rows)):
            A[rows[i], cols[i]] = vals[i]
        
        y = np.array(rhs)
        
        try:
            U_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, y, rcond=None)
        except:
            continue
        
        U_trial = U_flat.reshape(nm, r)
        
        # Verify
        if verify(U_trial, V_fixed, W_fixed, r):
            best_rank = r
            best_U = U_trial
            best_V = V_fixed
            best_W = W_fixed
            found_better = True
            break
    
    # If no improvement found, return the rank-42 solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

