"""Rank-32 (or lower) decomposition for 2x2x9 matrix multiplication.

Base: 4 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4) = 32.
Then attempt pair-term dependency removal to reduce rank to 31 or 30.

The index mapping follows the official trace convention W[k*n+i, r].
"""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_base_decomposition(n, m, p):
    """Build the base rank-32 decomposition."""
    rank = 4 * 7 + 4  # 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0; V[0*p + k][col] = 1.0; W[k*n + 0][col] = 1.0; col += 1
        U[1][col] = 1.0; V[1*p + k][col] = 1.0; W[k*n + 0][col] = 1.0; col += 1
        U[2][col] = 1.0; V[0*p + k][col] = 1.0; W[k*n + 1][col] = 1.0; col += 1
        U[3][col] = 1.0; V[1*p + k][col] = 1.0; W[k*n + 1][col] = 1.0; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    return U, V, W

def _try_remove_single_term(U, V, W, n, m, p, r_idx):
    """Try removing a single term r_idx by adjusting other terms.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
    rank = len(U[0])
    if r_idx < 0 or r_idx >= rank:
        return False, None, None, None
    
    # For each other term s, check if we can find a scalar alpha such that
    # removing term r_idx and adding alpha * term s preserves the tensor.
    # This requires: for all (i,j,k):
    #   U[i,j,r]*V[j,k,r]*W[k,i,r] = alpha * U[i,j,s]*V[j,k,s]*W[k,i,s]
    # for all (i,j,k) where the left side is nonzero.
    
    # Collect the "signature" of term r_idx: for each (i,j,k) with nonzero contribution
    sig_r = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                if abs(val) > 1e-12:
                    sig_r[(i,j,k)] = val
    
    if not sig_r:
        # Term is already zero, just remove it
        new_U = [row[:r_idx] + row[r_idx+1:] for row in U]
        new_V = [row[:r_idx] + row[r_idx+1:] for row in V]
        new_W = [row[:r_idx] + row[r_idx+1:] for row in W]
        return True, new_U, new_V, new_W
    
    # Try each other term s
    for s in range(rank):
        if s == r_idx:
            continue
        
        # Check if term s can absorb term r_idx
        # Need: sig_r[(i,j,k)] / (U[i,j,s]*V[j,k,s]*W[k,i,s]) = constant for all (i,j,k) in sig_r
        alphas = []
        valid = True
        for (i,j,k), val in sig_r.items():
            denom = U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
            if abs(denom) < 1e-12:
                valid = False
                break
            a = val / denom
            alphas.append(a)
        
        if not valid or not alphas:
            continue
        
        # Check all alphas are approximately equal
        a0 = alphas[0]
        if all(abs(a - a0) < 1e-9 for a in alphas):
            # Found a valid absorption: term r_idx = a0 * term s
            # Remove term r_idx
            new_U = [row[:r_idx] + row[r_idx+1:] for row in U]
            new_V = [row[:r_idx] + row[r_idx+1:] for row in V]
            new_W = [row[:r_idx] + row[r_idx+1:] for row in W]
            # Adjust term s (now at index s-1 if s > r_idx, else s)
            s_new = s - 1 if s > r_idx else s
            for i in range(n*m):
                new_U[i][s_new] += a0 * U[i][s]
            for j in range(m*p):
                new_V[j][s_new] += a0 * V[j][s]
            for k in range(p*n):
                new_W[k][s_new] += a0 * W[k][s]
            return True, new_U, new_V, new_W
    
    return False, None, None, None

def _try_remove_pair(U, V, W, n, m, p, r1, r2):
    """Try removing two terms r1, r2 by adjusting other terms.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
    rank = len(U[0])
    if r1 < 0 or r1 >= rank or r2 < 0 or r2 >= rank:
        return False, None, None, None
    
    # Collect signatures of terms r1 and r2
    sig = {}
    for (i,j,k) in [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]:
        v1 = U[i*m+j][r1] * V[j*p+k][r1] * W[k*n+i][r1]
        v2 = U[i*m+j][r2] * V[j*p+k][r2] * W[k*n+i][r2]
        if abs(v1) > 1e-12 or abs(v2) > 1e-12:
            sig[(i,j,k)] = (v1, v2)
    
    if not sig:
        # Both terms are zero
        remove = sorted([r1, r2])
        new_U = [row[:remove[0]] + row[remove[0]+1:remove[1]] + row[remove[1]+1:] for row in U]
        new_V = [row[:remove[0]] + row[remove[0]+1:remove[1]] + row[remove[1]+1:] for row in V]
        new_W = [row[:remove[0]] + row[remove[0]+1:remove[1]] + row[remove[1]+1:] for row in W]
        return True, new_U, new_V, new_W
    
    # Try to find scalars a, b such that for each other term s:
    #   a * U[i,j,s]*V[j,k,s]*W[k,i,s] + b * (something) = v1 + v2
    # This is more complex. For each pair of terms (s1, s2), try to find a, b such that:
    #   v1 + v2 = a * (U[i,j,s1]*V[j,k,s1]*W[k,i,s1]) + b * (U[i,j,s2]*V[j,k,s2]*W[k,i,s2])
    # for all (i,j,k).
    
    # Simplified: try absorbing into a single other term s with scalar a
    # (i.e., v1+v2 = a * term_s)
    for s in range(rank):
        if s == r1 or s == r2:
            continue
        
        alphas = []
        valid = True
        for (i,j,k), (v1, v2) in sig.items():
            denom = U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
            if abs(denom) < 1e-12:
                valid = False
                break
            a = (v1 + v2) / denom
            alphas.append(a)
        
        if not valid or not alphas:
            continue
        
        a0 = alphas[0]
        if all(abs(a - a0) < 1e-9 for a in alphas):
            # Remove both r1 and r2, adjust s
            remove = sorted([r1, r2])
            new_U = [row[:remove[0]] + row[remove[0]+1:remove[1]] + row[remove[1]+1:] for row in U]
            new_V = [row[:remove[0]] + row[remove[0]+1:remove[1]] + row[remove[1]+1:] for row in V]
            new_W = [row[:remove[0]] + row[remove[0]+1:remove[1]] + row[remove[1]+1:] for row in W]
            s_new = s
            if s > remove[0]:
                s_new -= 1
            if s > remove[1]:
                s_new -= 1
            for i in range(n*m):
                new_U[i][s_new] += a0 * U[i][s]
            for j in range(m*p):
                new_V[j][s_new] += a0 * V[j][s]
            for k in range(p*n):
                new_W[k][s_new] += a0 * W[k][s]
            return True, new_U, new_V, new_W
    
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    U, V, W = _build_base_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback: full schoolbook rank 36
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1.0
                    V_fb[j*p+k][c] = 1.0
                    W_fb[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try single-term removals
    rank = len(U[0])
    for r in range(rank):
        if time.time() > deadline - 20:
            break
        success, nU, nV, nW = _try_remove_single_term(U, V, W, n, m, p, r)
        if success:
            if _verify(nU, nV, nW, n, m, p, tol=1e-8):
                best_U, best_V, best_W = nU, nV, nW
                best_rank = len(nU[0])
                U, V, W = nU, nV, nW
                rank = len(U[0])
    
    # Try pair-term removals (only if we still have time)
    if time.time() < deadline - 30:
        rank = len(U[0])
        for r1, r2 in combinations(range(rank), 2):
            if time.time() > deadline - 20:
                break
            success, nU, nV, nW = _try_remove_pair(U, V, W, n, m, p, r1, r2)
            if success:
                if _verify(nU, nV, nW, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = nU, nV, nW
                    best_rank = len(nU[0])
                    U, V, W = nU, nV, nW
                    rank = len(U[0])
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
