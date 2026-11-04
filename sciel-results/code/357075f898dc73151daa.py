import time
import random
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n*m)])
        vc = np.array([V[j][c] for j in range(m*p)])
        wc = np.array([W[k][c] for k in range(p*n)])
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2 block at row i0..i0+1, col j0..j0+1, producing k0..k0+1."""
    a_coeffs = [
        [(0,1), (1,1)],
        [(1,0), (1,1)],
        [(0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (0,0)],
        [(0,1), (1,1)],
    ]
    a_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    
    b_coeffs = [
        [(0,0), (1,1)],
        [(0,0)],
        [(1,0), (1,1)],
        [(0,1), (0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (1,1)],
    ]
    b_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    
    c_contribs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    for t in range(7):
        c = col + t
        for (di, dj), coeff in zip(a_coeffs[t], a_signs[t]):
            row = (i0 + di) * m + (j0 + dj)
            U[row][c] += coeff
        for (dj, dk), coeff in zip(b_coeffs[t], b_signs[t]):
            row = (j0 + dj) * p + (k0 + dk)
            V[row][c] += coeff
        for (dk, di), coeff in c_contribs[t].items():
            row = (k0 + dk) * n + (i0 + di)
            W[row][c] += coeff
    
    return col + 7

def _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col):
    """Schoolbook for sub-block of size hi x hj x hk."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] += 1
                V[(j0 + dj) * p + (k0 + dk)][col] += 1
                W[(k0 + dk) * n + (i0 + di)][col] += 1
                col += 1
    return col

def _construct_hybrid(splits_i, splits_j, splits_k, use_strassen_for_2x2):
    """Construct using block decomposition."""
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    i_offsets = [0]
    for s in splits_i[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in splits_j[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in splits_k[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                
                if hi == 2 and hj == 2 and hk == 2 and use_strassen_for_2x2:
                    col = _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col)
    
    # Trim trailing zero columns
    last_col = 0
    for c in range(col - 1, -1, -1):
        nz = False
        for row in range(n * m):
            if U[row][c] != 0:
                nz = True
                break
        if not nz:
            for row in range(m * p):
                if V[row][c] != 0:
                    nz = True
                    break
        if not nz:
            for row in range(p * n):
                if W[row][c] != 0:
                    nz = True
                    break
        if nz:
            last_col = c
            break
    final_rank = last_col + 1
    
    U_trim = [row[:final_rank] for row in U]
    V_trim = [row[:final_rank] for row in V]
    W_trim = [row[:final_rank] for row in W]
    
    return {"U": U_trim, "V": V_trim, "W": W_trim, "_rank": final_rank, "_col": col}

def _all_compositions(total, min_part=1, max_parts=4):
    """Generate all compositions of total into parts >= min_part, up to max_parts parts."""
    results = []
    for num in range(1, max_parts + 1):
        for comp in itertools.product(range(min_part, total + 1), repeat=num):
            if sum(comp) == total:
                results.append(comp)
    return results

def _try_structured_search(deadline):
    """
    Try structured decompositions using combinations of Strassen blocks
    and schoolbook, with careful attention to overlapping regions.
    
    Strategy: For 3x5x7, we can try:
    - Split into (2+1) x (2+2+1) x (2+2+2+1) type patterns
    - Use Strassen on 2x2x2 subblocks
    - Use schoolbook for the rest
    - Also try non-standard coefficient patterns
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    # Try various split patterns
    comp_i = _all_compositions(n, max_parts=3)
    comp_j = _all_compositions(m, max_parts=3)
    comp_k = _all_compositions(p, max_parts=4)
    
    for ci in comp_i:
        if time.time() > deadline - 15:
            break
        for cj in comp_j:
            if time.time() > deadline - 15:
                break
            for ck in comp_k:
                if time.time() > deadline - 15:
                    break
                for use_str in [True, False]:
                    if time.time() > deadline - 15:
                        break
                    try:
                        result = _construct_hybrid(ci, cj, ck, use_str)
                        rank = result["_rank"]
                        if _verify(result["U"], result["V"], result["W"]):
                            if rank < best_rank:
                                best_rank = rank
                                best = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    except Exception:
                        pass
    return best

def _try_coefficient_optimization(deadline, base_result):
    """
    Try to optimize coefficients of a base decomposition to reduce rank.
    This is a heuristic approach: start from a valid decomposition and try
    to merge columns by finding linear dependencies.
    """
    n, m, p = DIMENSIONS
    if base_result is None:
        return None
    
    U = [row[:] for row in base_result["U"]]
    V = [row[:] for row in base_result["V"]]
    W = [row[:] for row in base_result["W"]]
    r = len(U[0])
    best_rank = r
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find linear dependencies among columns
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try removing columns that are linear combinations of others
    # This is a heuristic - try each column and see if it can be expressed
    # as a combination of the first few columns
    
    for col_to_remove in range(r - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        if col_to_remove >= best_rank:
            continue
            
        # Try to see if column col_to_remove is a linear combination
        # of the other columns (this is expensive, so we do a simplified check)
        
        # Simple check: try small integer coefficients for the first few columns
        # This is very heuristic
        if best_rank - col_to_remove > 5:  # Only try if we're not too far off
            continue
            
    return None  # This approach is too expensive; return the base result

def _try_direct_search(deadline, start_rank=85):
    """
    Try to find a low-rank decomposition via structured random search.
    Instead of pure random, use patterns inspired by Strassen.
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    random.seed(99999)
    
    for target_rank in range(start_rank, min(n*m*p, start_rank + 20)):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random attempts with structured initialization
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with a mix of schoolbook-like and Strassen-like patterns
            for c in range(target_rank):
                pattern = random.choice(['schoolbook', 'strassen_mix', 'sparse'])
                
                if pattern == 'schoolbook':
                    # Pick a random (i,j,k) triple
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    U[i*m+j][c] = random.choice([-1, 1])
                    V[j*p+k][c] = random.choice([-1, 1])
                    W[k*n+i][c] = random.choice([-1, 1])
                    
                elif pattern == 'strassen_mix':
                    # Pick 2-4 entries in each factor with signs
                    for _ in range(random.randint(2, 4)):
                        ri = random.randint(0, n*m-1)
                        U[ri][c] = random.choice([-1, 1])
                    for _ in range(random.randint(2, 4)):
                        rj = random.randint(0, m*p-1)
                        V[rj][c] = random.choice([-1, 1])
                    for _ in range(random.randint(2, 4)):
                        rk = random.randint(0, p*n-1)
                        W[rk][c] = random.choice([-1, 1])
                        
                else:  # sparse
                    num_u = random.randint(1, 2)
                    indices_u = random.sample(range(n*m), num_u)
                    for idx in indices_u:
                        U[idx][c] = random.choice([-1, 1])
                    
                    num_v = random.randint(1, 2)
                    indices_v = random.sample(range(m*p), num_v)
                    for idx in indices_v:
                        V[idx][c] = random.choice([-1, 1])
                    
                    num_w = random.randint(1, 2)
                    indices_w = random.sample(range(p*n), num_w)
                    for idx in indices_w:
                        W[idx][c] = random.choice([-1, 1])
            
            if _verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best = {"U": U, "V": V, "W": W}
                break  # Found valid for this rank, try lower
    
    return best

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 110
    best_result = None
    best_rank = 106
    
    # 1. Try schoolbook first as baseline
    schoolbook = _construct_schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # 2. Try systematic hybrid search with various splits
    if time.time() < deadline - 30:
        hybrid = _try_structured_search(deadline)
        if hybrid is not None and _verify(hybrid["U"], hybrid["V"], hybrid["W"]):
            r = len(hybrid["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = hybrid
    
    # 3. Try direct structured search if time remains
    if time.time() < deadline - 25 and best_rank > 95:
        direct_result = _try_direct_search(deadline, start_rank=max(85, best_rank - 15))
        if direct_result is not None and _verify(direct_result["U"], direct_result["V"], direct_result["W"]):
            r = len(direct_result["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = direct_result
    
    if best_result is None:
        return schoolbook
    return best_result
