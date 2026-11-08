"""FMM-inspired approach: systematic bilinear reduction for matrix multiplication tensor."""
import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

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

    # Build the 44-rank decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
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
    
    # Part 1: A1 (2x2) * B1 (2x8)
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
                return _schoolbook()
    
    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    candidate = {"U": U, "V": V, "W": W}
    
    # Try to find a rank 43 solution via randomized search
    if _elapsed_ok():
        best_rank = 44
        best_candidate = candidate
        
        # Define target tensor
        target = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    target[i, a, k] = 1
        
        def _compute_tensor(U_mat, V_mat, W_mat):
            # U: n*m x R, V: m*p x R, W: p*n x R
            # Reshape to n x m x R, m x p x R, p x n x R
            U_t = np.array(U_mat).reshape(n, m, -1)
            V_t = np.array(V_mat).reshape(m, p, -1)
            W_t = np.array(W_mat).reshape(p, n, -1)
            
            # Sum over R
            # T[i,a,k] = sum_r U[i,a,r] V[a,k,r] W[k,i,r]
            # Use einsum
            return np.einsum('iar,akr,ki r->iak'.replace(' ', ''), U_t, V_t, W_t)
        
        def _search_rank(R):
            if not _elapsed_ok():
                return None
            # Initialize with random small integers
            U_r = np.random.randint(-2, 3, size=(n*m, R))
            V_r = np.random.randint(-2, 3, size=(m*p, R))
            W_r = np.random.randint(-2, 3, size=(p*n, R))
            
            # Local search
            for _ in range(500):
                if not _elapsed_ok():
                    break
                T_calc = _compute_tensor(U_r, V_r, W_r)
                if np.array_equal(T_calc, target):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # Pick a random element to perturb
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    r_idx, c_idx = np.random.randint(n*m), np.random.randint(R)
                    U_r[r_idx, c_idx] += np.random.choice([-1, 1])
                elif mat_idx == 1:
                    r_idx, c_idx = np.random.randint(m*p), np.random.randint(R)
                    V_r[r_idx, c_idx] += np.random.choice([-1, 1])
                else:
                    r_idx, c_idx = np.random.randint(p*n), np.random.randint(R)
                    W_r[r_idx, c_idx] += np.random.choice([-1, 1])
            
            return None

        # Try rank 43
        res = _search_rank(43)
        if res:
            candidate = {"U": res[0], "V": res[1], "W": res[2]}
            if _validate(candidate):
                return candidate

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
