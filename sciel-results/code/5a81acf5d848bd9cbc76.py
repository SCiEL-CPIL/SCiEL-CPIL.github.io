import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    total_entries = n * m * p

    # Generate all 2x2x2 cubes
    cubes = []
    masks = []
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                mask = 0
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            idx = (i + di) * m * p + (j + dj) * p + (k + dk)
                            mask |= (1 << idx)
                cubes.append((i, j, k))
                masks.append(mask)
                
    num_cubes = len(cubes)
    
    # Branch and Bound to find max disjoint cubes
    max_count = 0
    best_selection = []
    
    # Order cubes by number of conflicts to prune better?
    # Or just use a simple heuristic.
    # Let's use a simple recursive search.
    
    def search(idx, used_mask, current_count, selection):
        nonlocal max_count, best_selection
        if time.time() > deadline - 5.0:
            return
            
        if idx == num_cubes:
            if current_count > max_count:
                max_count = current_count
                best_selection = selection[:]
            return
            
        # Pruning
        # Max possible additional cubes
        # Simple bound: remaining cubes
        if current_count + (num_cubes - idx) <= max_count:
            return
            
        # Option 1: Skip
        search(idx + 1, used_mask, current_count, selection)
        
        # Option 2: Take
        if (used_mask & masks[idx]) == 0:
            selection.append(idx)
            search(idx + 1, used_mask | masks[idx], current_count + 1, selection)
            selection.pop()

    # To speed up, we can sort cubes by how "central" they are or number of conflicts.
    # For now, just run it. 48 is small.
    search(0, 0, 0, [])
    
    # If search was cut short, we might have a suboptimal max_count.
    # We can run a few random greedy tries as a fallback or to find a good initial bound.
    # But branch and bound should be fast.
    
    # Construct decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i, j, k):
        a11 = i * m + j
        a12 = i * m + j + 1
        a21 = (i + 1) * m + j
        a22 = (i + 1) * m + j + 1
        
        b11 = j * p + k
        b12 = j * p + k + 1
        b21 = (j + 1) * p + k
        b22 = (j + 1) * p + k + 1
        
        c11 = k * n + i
        c12 = (k + 1) * n + i
        c21 = k * n + (i + 1)
        c22 = (k + 1) * n + (i + 1)
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    used_mask = 0
    for idx in best_selection:
        i, j, k = cubes[idx]
        add_strassen_block(i, j, k)
        used_mask |= masks[idx]

    # Add remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                if not (used_mask & (1 << idx)):
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
