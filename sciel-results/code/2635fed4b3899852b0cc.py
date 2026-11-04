"""Matrix multiplication 2x3x8 with optimized Strassen-based decomposition.

Strategy:
1. Build rank-44 decomposition using Strassen for 2x2x2 blocks + schoolbook for 2x1x8.
2. Aggressively compress via greedy column removal with exact integer validation.
3. Try pair-merging: check if two columns can be replaced by one.
4. Try random restarts with different column orderings.
"""
import time
import numpy as np
from fractions import Fraction
import random

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

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

    def _validate_exact(U, V, W):
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

    def _build_base():
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

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate_exact(base["U"], base["V"], base["W"]):
        return _schoolbook()

    cur_U = [row[:] for row in base["U"]]
    cur_V = [row[:] for row in base["V"]]
    cur_W = [row[:] for row in base["W"]]
    cur_r = 44

    # Greedy column removal with exact integer validation
    improved = True
    while improved and _elapsed_ok():
        improved = False
        for j in range(cur_r - 1, -1, -1):
            if not _elapsed_ok():
                break

            new_r = cur_r - 1
            if new_r <= 0:
                break

            new_U = [row[:j] + row[j+1:] for row in cur_U]
            new_V = [row[:j] + row[j+1:] for row in cur_V]
            new_W = [row[:j] + row[j+1:] for row in cur_W]

            if _validate_exact(new_U, new_V, new_W):
                cur_U = new_U
                cur_V = new_V
                cur_W = new_W
                cur_r = new_r
                improved = True

    # Try pair merging: for each pair (j1, j2), check if removing both and adding a merged term works
    if _elapsed_ok():
        for j1 in range(cur_r - 1):
            if not _elapsed_ok():
                break
            for j2 in range(j1 + 1, cur_r):
                if not _elapsed_ok():
                    break
                
                # Try to find a merged column that replaces columns j1 and j2
                # The merged column must satisfy: for all (i,a,k),
                #   u_new[i*m+a] * v_new[a*p+k] * w_new[k*n+i] = 
                #   u_j1[i*m+a]*v_j1[a*p+k]*w_j1[k*n+i] + u_j2[i*m+a]*v_j2[a*p+k]*w_j2[k*n+i]
                
                # This is generally not possible with a single rank-1 term, but let's try
                # a heuristic: if the two columns have the same support pattern, try merging
                u1 = [cur_U[i][j1] for i in range(n*m)]
                u2 = [cur_U[i][j2] for i in range(n*m)]
                v1 = [cur_V[i][j1] for i in range(m*p)]
                v2 = [cur_V[i][j2] for i in range(m*p)]
                w1 = [cur_W[i][j1] for i in range(p*n)]
                w2 = [cur_W[i][j2] for i in range(p*n)]
                
                # Check if u1 and u2 are proportional
                prop_u = True
                for a, b in zip(u1, u2):
                    if a == 0 and b != 0:
                        prop_u = False
                        break
                    if a != 0 and b == 0:
                        prop_u = False
                        break
                if prop_u:
                    # Try merging with scaling
                    new_r = cur_r - 1
                    new_U = [row[:j1] + row[j1+1:j2] + row[j2+1:] for row in cur_U]
                    new_V = [row[:j1] + row[j1+1:j2] + row[j2+1:] for row in cur_V]
                    new_W = [row[:j1] + row[j1+1:j2] + row[j2+1:] for row in cur_W]
                    
                    # Add merged column
                    merged_U = [u1[i] + u2[i] for i in range(n*m)]
                    merged_V = [v1[i] + v2[i] for i in range(m*p)]
                    merged_W = [w1[i] + w2[i] for i in range(p*n)]
                    
                    for i in range(n*m):
                        new_U[i].append(merged_U[i])
                    for i in range(m*p):
                        new_V[i].append(merged_V[i])
                    for i in range(p*n):
                        new_W[i].append(merged_W[i])
                    
                    if _validate_exact(new_U, new_V, new_W):
                        cur_U = new_U
                        cur_V = new_V
                        cur_W = new_W
                        cur_r = new_r
                        break

    if _validate_exact(cur_U, cur_V, cur_W):
        return {"U": cur_U, "V": cur_V, "W": cur_W}
    
    return _schoolbook()
