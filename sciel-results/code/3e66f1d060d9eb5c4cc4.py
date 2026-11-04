"""Combined: block decomposition with Strassen for 2x2 sub-blocks, verified with fallback."""

import numpy as np
import time
import random

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
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

def _strassen_2x2(U, V, W, i_base, j_base, k_base, r_offset):
    """
    Add Strassen rank-7 terms for a 2x2x2 sub-block.
    i_base: starting row of A (0 or 1)
    j_base: starting col of A / row of B (0 or 1)
    k_base: starting col of B (0 or 1)
    r_offset: starting column index in U,V,W
    """
    n, m, p = 2, 3, 5
    
    # Local indices within the 2x2 block:
    # A rows: i_base, i_base+1; A cols: j_base, j_base+1
    # B rows: j_base, j_base+1; B cols: k_base, k_base+1
    # C rows: k_base, k_base+1; C cols: i_base, i_base+1
    
    a0 = i_base * m + j_base
    a1 = i_base * m + j_base + 1
    a2 = (i_base + 1) * m + j_base
    a3 = (i_base + 1) * m + j_base + 1
    
    b0 = j_base * p + k_base
    b1 = j_base * p + k_base + 1
    b2 = (j_base + 1) * p + k_base
    b3 = (j_base + 1) * p + k_base + 1
    
    c0 = k_base * n + i_base
    c1 = k_base * n + i_base + 1
    c2 = (k_base + 1) * n + i_base
    c3 = (k_base + 1) * n + i_base + 1
    
    r = r_offset
    
    # M1 = (A00+A10)(B00+B10)
    U[a0][r] += 1
    U[a2][r] += 1
    V[b0][r] += 1
    V[b2][r] += 1
    W[c0][r] += 1
    W[c3][r] += 1
    r += 1
    
    # M2 = (A10+A11)(B00)
    U[a2][r] += 1
    U[a3][r] += 1
    V[b0][r] += 1
    W[c1][r] += 1
    W[c3][r] += -1
    r += 1
    
    # M3 = (A00)(B01-B11)
    U[a0][r] += 1
    V[b1][r] += 1
    V[b3][r] += -1
    W[c2][r] += 1
    W[c3][r] += 1
    r += 1
    
    # M4 = (A11)(B10-B00)
    U[a3][r] += 1
    V[b2][r] += 1
    V[b0][r] += -1
    W[c0][r] += 1
    W[c1][r] += 1
    r += 1
    
    # M5 = (A00+A01)(B11)
    U[a0][r] += 1
    U[a1][r] += 1
    V[b3][r] += 1
    W[c0][r] += -1
    W[c2][r] += 1
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[a2][r] += 1
    U[a0][r] += -1
    V[b0][r] += 1
    V[b1][r] += 1
    W[c3][r] += 1
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[a1][r] += 1
    U[a3][r] += -1
    V[b2][r] += 1
    V[b3][r] += 1
    W[c0][r] += 1
    r += 1
    
    return r_offset + 7

def _verify_decomposition(U, V, W, rank, n, m, p):
    """Verify that the decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all matching terms give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check non-matching terms give 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                if i_a == i_c and j_a == j_b:
                    continue
                    
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[a, r_idx] * V_arr[b, r_idx] * W_arr[c, r_idx]
                if abs(s) > 1e-9:
                    return False
    
    return True

def _try_random_decomposition(rank, n, m, p, deadline, max_iters=5000):
    """Try to find a valid decomposition with given rank using random initialization + refinement."""
    U_arr = np.zeros((n * m, rank))
    V_arr = np.zeros((m * p, rank))
    W_arr = np.zeros((p * n, rank))
    
    # Random initialization with small values
    for r in range(rank):
        # Randomly assign some entries
        for _ in range(3):
            idx_u = random.randint(0, n * m - 1)
            idx_v = random.randint(0, m * p - 1)
            idx_w = random.randint(0, p * n - 1)
            val = random.choice([-1, 1])
            U_arr[idx_u, r] = val
            V_arr[idx_v, r] = val
            W_arr[idx_w, r] = val
    
    # Greedy refinement: try to fix errors
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute all errors
        errors = {}
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    is_match = (i_a == i_c and j_a == j_b)
                    target = 1.0 if is_match else 0.0
                    
                    s = 0.0
                    for r_idx in range(rank):
                        s += U_arr[a, r_idx] * V_arr[b, r_idx] * W_arr[c, r_idx]
                    
                    err = abs(s - target)
                    if err > 1e-9:
                        errors[(a, b, c)] = (s, target, err)
        
        if not errors:
            return U_arr, V_arr, W_arr
        
        # Pick the largest error and try to fix it
        max_err_key = max(errors, key=lambda k: errors[k][2])
        a, b, c = max_err_key
        s, target, err = errors[max_err_key]
        
        # Try adjusting one of the rank-1 components
        # Find which component contributes most to this error
        contributions = []
        for r_idx in range(rank):
            contrib = U_arr[a, r_idx] * V_arr[b, r_idx] * W_arr[c, r_idx]
            contributions.append((abs(contrib), r_idx, contrib))
        
        if contributions:
            contributions.sort(reverse=True)
            _, r_idx, contrib = contributions[0]
            
            # Try flipping or adjusting this component
            # If we need to increase the sum, flip sign if contrib has wrong sign
            if target > s:
                # Need to increase
                if contrib < 0:
                    # Flip one factor
                    if U_arr[a, r_idx] != 0:
                        U_arr[a, r_idx] *= -1
                    elif V_arr[b, r_idx] != 0:
                        V_arr[b, r_idx] *= -1
                    elif W_arr[c, r_idx] != 0:
                        W_arr[c, r_idx] *= -1
            else:
                # Need to decrease
                if contrib > 0:
                    if U_arr[a, r_idx] != 0:
                        U_arr[a, r_idx] *= -1
                    elif V_arr[b, r_idx] != 0:
                        V_arr[b, r_idx] *= -1
                    elif W_arr[c, r_idx] != 0:
                        W_arr[c, r_idx] *= -1
    
    return None

def _strassen_rank28():
    """The known rank-28 Strassen-based decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: Block (i=0,1, j=2, k=0..4): A[:,2] * B[2,:] -> C[:,0..4]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: Block (i=0,1, j=0,1, k=4): A[:,:2] * B[:,4] -> C[4,:]
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: Block (i=0,1, j=0,1, k=0,1): Strassen
    r = _strassen_2x2(U, V, W, i_base=0, j_base=0, k_base=0, r_offset=r)
    
    # Part 4: Block (i=0,1, j=0,1, k=2,3): Strassen
    r = _strassen_2x2(U, V, W, i_base=0, j_base=0, k_base=2, r_offset=r)
    
    return U, V, W, rank

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with known rank-28 solution
    U28, V28, W28, rank28 = _strassen_rank28()
    best_result = {"U": U28, "V": V28, "W": W28}
    best_rank = rank28
    
    # Try to improve with lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But practical lower bounds are higher. Try ranks from 27 down.
    
    for target_rank in range(27, 14, -1):
        if time.time() > deadline:
            break
        
        # Try random initialization with greedy refinement
        result = _try_random_decomposition(target_rank, n, m, p, deadline, max_iters=2000)
        
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Verify
            if _verify_decomposition(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), target_rank, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist()
                }
                break  # Found a valid decomposition, stop
    
    if time.time() > deadline:
        return best_result
    
    # Final verification of best result
    U_final = np.array(best_result["U"], dtype=float)
    V_final = np.array(best_result["V"], dtype=float)
    W_final = np.array(best_result["W"], dtype=float)
    final_rank = len(U_final[0])
    
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], final_rank, n, m, p):
        return _schoolbook(n, m, p)
    
    return best_result
