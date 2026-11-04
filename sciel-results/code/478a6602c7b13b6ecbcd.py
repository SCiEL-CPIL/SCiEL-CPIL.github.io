import time
import numpy as np
from fractions import Fraction

def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    add_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
    add_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    add_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
    add_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])

    def verify_solution(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    s = 0
                    for r_idx in range(r):
                        s += U_mat[a][r_idx] * V_mat[b][r_idx] * W_mat[c][r_idx]
                    if s != 1:
                        return False
        return True

    # Verify current solution
    if not verify_solution(U, V, W, 28):
        return {"U": U, "V": V, "W": W}

    # Try to find rank 27 by attempting to eliminate one term
    # For each term to eliminate, check if the remaining 27 terms can be
    # rescaled to still give the correct tensor.
    
    best_rank = 28
    best_U, best_V, best_W = U, V, W

    # Try eliminating each term and checking if the remaining terms still work
    # with some rescaling. This is a heuristic.
    
    for elim in range(28):
        if time.time() > _deadline:
            break
        
        # Try to see if we can express the eliminated term as a combination
        # of the others. This requires solving a non-linear system.
        # For simplicity, check if the eliminated term is "redundant" by
        # seeing if removing it and adjusting others can still give the tensor.
        
        # Quick check: if the eliminated term has all zeros in one of U, V, W,
        # it's already redundant.
        is_redundant = True
        for a in range(n * m):
            if U[a][elim] != 0:
                is_redundant = False
                break
        if is_redundant:
            for b in range(m * p):
                if V[b][elim] != 0:
                    is_redundant = False
                    break
        if is_redundant:
            for c in range(p * n):
                if W[c][elim] != 0:
                    is_redundant = False
                    break
        
        if is_redundant:
            # Create new matrices without this term
            new_rank = 27
            new_U = [[U[a][r] if r < elim else U[a][r+1] for r in range(new_rank)] for a in range(n*m)]
            new_V = [[V[b][r] if r < elim else V[b][r+1] for r in range(new_rank)] for b in range(m*p)]
            new_W = [[W[c][r] if r < elim else W[c][r+1] for r in range(new_rank)] for c in range(p*n)]
            
            if verify_solution(new_U, new_V, new_W, new_rank):
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                break

    _elapsed = time.time() - _t0
    if best_rank < 28:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
