"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    for rank in range(20, 28):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        result = _try_structured_search(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to known rank 28
    U, V, W = _strassen_28(n, m, p)
    return {"U": U, "V": V, "W": W}

def _try_structured_search(n, m, p, rank):
    """
    Try to find a rank-r decomposition using structured search.
    Key insight: T[i,j,k] = 1 for all i,j,k.
    We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all (i,j,k).
    
    Strategy: For each j, the V[j,:,r] and W[:,i,r] interact through k.
    Fix V and W, solve for U linearly. Then optimize V and W.
    """
    deadline = START_TIME + TIME_LIMIT - 10
    
    # Precompute: for each (i,j,k), the equation is sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # Group by j: for fixed j, we have n*p = 8 equations involving U[i,j,:], V[j,k,:], W[k,i,:]
    
    best_result = None
    np.random.seed(7)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Initialize with small integer/half-integer values
        U = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=(n*m, rank))
        V = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=(m*p, rank))
        W = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=(p*n, rank))
        
        # Local search: try to reduce the max absolute error
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Compute errors for all (i,j,k)
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        errors[i,j,k] = s - 1.0
            
            max_err = np.max(np.abs(errors))
            if max_err < 1e-10:
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return {
                        "U": [[float(x) for x in row] for row in U],
                        "V": [[float(x) for x in row] for row in V],
                        "W": [[float(x) for x in row] for row in W]
                    }
            
            # Pick the worst (i,j,k) and try to fix it
            worst_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
            wi, wj, wk = worst_idx
            
            # Try adjusting one element of U, V, or W
            improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                mat = np.random.randint(3)
                if mat == 0:
                    # Adjust U[wi, wj, r]
                    r = np.random.randint(rank)
                    old = U[wi*m+wj, r]
                    # Try to adjust to reduce error
                    # dU such that dU * V[wj,wk,r] * W[wk,wi,r] reduces error
                    denom = V[wj*p+wk, r] * W[wk*n+wi, r]
                    if abs(denom) > 1e-6:
                        delta = -errors[wi,wj,wk] / denom * 0.5
                        new_val = old + delta
                        # Snap to half-integer
                        new_val = np.round(new_val * 2) / 2
                        U[wi*m+wj, r] = new_val
                        # Check if improved
                        new_err = 0.0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    s = sum(U[i2*m+j2, rr] * V[j2*p+k2, rr] * W[k2*n+i2, rr] for rr in range(rank))
                                    new_err += (s - 1.0)**2
                        old_err = np.sum(errors**2)
                        if new_err < old_err * 0.99:
                            improved = True
                            break
                        else:
                            U[wi*m+wj, r] = old
                elif mat == 1:
                    # Adjust V[wj, wk, r]
                    r = np.random.randint(rank)
                    old = V[wj*p+wk, r]
                    denom = U[wi*m+wj, r] * W[wk*n+wi, r]
                    if abs(denom) > 1e-6:
                        delta = -errors[wi,wj,wk] / denom * 0.5
                        new_val = old + delta
                        new_val = np.round(new_val * 2) / 2
                        V[wj*p+wk, r] = new_val
                        new_err = 0.0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    s = sum(U[i2*m+j2, rr] * V[j2*p+k2, rr] * W[k2*n+i2, rr] for rr in range(rank))
                                    new_err += (s - 1.0)**2
                        old_err = np.sum(errors**2)
                        if new_err < old_err * 0.99:
                            improved = True
                            break
                        else:
                            V[wj*p+wk, r] = old
                else:
                    # Adjust W[wk, wi, r]
                    r = np.random.randint(rank)
                    old = W[wk*n+wi, r]
                    denom = U[wi*m+wj, r] * V[wj*p+wk, r]
                    if abs(denom) > 1e-6:
                        delta = -errors[wi,wj,wk] / denom * 0.5
                        new_val = old + delta
                        new_val = np.round(new_val * 2) / 2
                        W[wk*n+wi, r] = new_val
                        new_err = 0.0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    s = sum(U[i2*m+j2, rr] * V[j2*p+k2, rr] * W[k2*n+i2, rr] for rr in range(rank))
                                    new_err += (s - 1.0)**2
                        old_err = np.sum(errors**2)
                        if new_err < old_err * 0.99:
                            improved = True
                            break
                        else:
                            W[wk*n+wi, r] = old
            
            if not improved:
                # Stuck, break and try new initialization
                break
    
    return None

def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    # Trim to actual used columns
    used_cols = set()
    for row in U:
        for c, val in enumerate(row):
            if val != 0:
                used_cols.add(c)
    for row in V:
        for c, val in enumerate(row):
            if val != 0:
                used_cols.add(c)
    for row in W:
        for c, val in enumerate(row):
            if val != 0:
                used_cols.add(c)
    
    if len(used_cols) < rank:
        sorted_cols = sorted(used_cols)
        col_map = {old: new for new, old in enumerate(sorted_cols)}
        U = [[row[col_map[c]] for c in sorted_cols] for row in U]
        V = [[row[col_map[c]] for c in sorted_cols] for row in V]
        W = [[row[col_map[c]] for c in sorted_cols] for row in W]
    
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return U, V, W
