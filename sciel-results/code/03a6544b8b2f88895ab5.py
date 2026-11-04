import numpy as np
import random
import time
import sys
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 145

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Precompute target tensor indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def verify_exact(U, V, W, rank):
        for i, j, k in indices:
            s = 0
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            for c in range(rank):
                s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
            if s != 1:
                return False
        return True

    def verify_float(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i, j, k in indices:
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            s = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
            err = abs(s - 1.0)
            if err > max_err:
                max_err = err
            if max_err > 1e-6:
                return False
        return True

    # Search for lower rank
    # Known lower bound is 21.
    # Try ranks from 21 to best_rank - 1
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random candidates
        # Use small integers
        max_attempts = 2000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Values: -1, 0, 1
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(pn)]
            
            # Quick float check
            if verify_float(U, V, W, target_rank):
                # Exact check
                if verify_exact(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue

    return best
