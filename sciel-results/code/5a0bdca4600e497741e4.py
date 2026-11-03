import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # All 2x2x2 sub-blocks
    def all_blocks(n, m, p):
        blocks = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    blocks.append((i0, i0 + 1, j0, j0 + 1, k0, k0 + 1))
        return blocks
    
    all_blks = all_blocks(n, m, p)
    
    # Greedy: pick blocks that maximize uncovered entries, avoiding duplicates
    # Use multiple random orderings and pick the best
    
    import random
    best_saving = -1
    best_selected = []
    
    # Use multiple random seeds
    for seed in range(200):
        if time.time() > deadline - 60:
            break
        rng = random.Random(seed)
        covered_tmp = set()
        selected_tmp = []
        blocks_tmp = list(all_blks)
        rng.shuffle(blocks_tmp)
        
        for blk in blocks_tmp:
            i0, i1, j0, j1, k0, k1 = blk
            entries = set()
            for i in (i0, i1):
                for j in (j0, j1):
                    for k in (k0, k1):
                        entries.add((i, j, k))
            
            # Check for overlap
            if entries & covered_tmp:
                continue
            
            covered_tmp.update(entries)
            selected_tmp.append(blk)
        
        if len(selected_tmp) > len(best_selected):
            best_selected = selected_tmp
            best_saving = len(selected_tmp)
    
    # Now use best_selected blocks
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i0, i1, j0, j1, k0, k1 in best_selected:
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
        
        # Global indices
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    # Add schoolbook for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    # Convert to dense matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
