"""Matrix multiplication 2x3x8 with rank-44 Strassen-based decomposition,
with an attempt at linear-algebraic compression to reduce rank below 44.

Strategy:
1. Build the standard rank-44 decomposition (4 Strassen blocks for 2x2x8 + 16 schoolbook terms for 2x1x8).
2. Attempt to find linear dependencies among the rank-1 terms by checking if any column of the combined
   factor matrices is in the span of others (over rationals). If a dependency is found, eliminate the
   redundant term to get a lower rank.
3. Validate the result exactly.
"""
import time
import numpy as np
from scipy.linalg import solve
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    n, m, p = 2, 3, 8

    def _elapsed_ok():
        return (time.time() - t0) < deadline

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
                    acc = 0
                    for c in range(rank):
                        acc += U[i * m + a][c] * V[a * p + k][c] * W[k * n + i][c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2 columns
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] += u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] += v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Attempt linear algebraic compression using numpy for speed.
    # Convert to numpy arrays for efficient operations.
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the tensor equation matrix.
    # For each (i, a, k), the equation is sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c] = 1
    # We want to find if we can reduce rank by finding linear dependencies.
    
    # A better approach: Try to find a lower rank decomposition by solving a least squares problem
    # or by checking if the current columns are linearly independent in a way that allows reduction.
    
    # Since the entries are integers, we can try to find if any column can be expressed as a linear
    # combination of others. But this is hard to do directly.
    
    # Instead, let's try a greedy approach with numpy for speed:
    # For each column j, check if removing it still allows the tensor to be decomposed.
    # This is equivalent to checking if the remaining columns can still satisfy the equations.
    
    best_rank = rank
    best_candidate = candidate
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Try removing each column one at a time
    for j in range(current_rank):
        if not _elapsed_ok():
            break
        
        # Create a copy without column j
        new_rank = current_rank - 1
        if new_rank <= 0:
            break
            
        # Remove column j
        mask = np.ones(current_rank, dtype=bool)
        mask[j] = False
        new_U = current_U[:, mask]
        new_V = current_V[:, mask]
        new_W = current_W[:, mask]
        
        # Validate using numpy
        valid = True
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = np.sum(new_U[i*m+a, :] * new_V[a*p+k, :] * new_W[k*n+i, :])
                    if abs(acc - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = new_rank
            # Convert back to list format
            new_U_list = [[0] * new_rank for _ in range(n * m)]
            new_V_list = [[0] * new_rank for _ in range(m * p)]
            new_W_list = [[0] * new_rank for _ in range(p * n)]
            
            for row in range(n * m):
                for c in range(new_rank):
                    val = new_U[row, c]
                    new_U_list[row][c] = int(round(val)) if abs(val - round(val)) < 1e-10 else val
            
            for row in range(m * p):
                for c in range(new_rank):
                    val = new_V[row, c]
                    new_V_list[row][c] = int(round(val)) if abs(val - round(val)) < 1e-10 else val
            
            for row in range(p * n):
                for c in range(new_rank):
                    val = new_W[row, c]
                    new_W_list[row][c] = int(round(val)) if abs(val - round(val)) < 1e-10 else val
            
            best_candidate = {"U": new_U_list, "V": new_V_list, "W": new_W_list}
            
            # Update for next iteration
            current_U = new_U
            current_V = new_V
            current_W = new_W
            current_rank = new_rank
            rank = new_rank
            U = new_U_list
            V = new_V_list
            W = new_W_list
            candidate = best_candidate
    
    if best_rank < 44:
        return best_candidate
    else:
        return candidate
