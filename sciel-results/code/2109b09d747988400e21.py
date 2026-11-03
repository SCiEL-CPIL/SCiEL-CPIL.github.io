import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                all_blocks.append((i0, i1, j0, j1, k0, k1))
                
    num_blocks = len(all_blocks)
    
    # Build conflict masks: blocks conflict if they share any (i,j,k) position
    # Two 2x2x2 blocks conflict if their i-ranges overlap AND j-ranges overlap AND k-ranges overlap
    conflict_mask = [0] * num_blocks
    
    for i in range(num_blocks):
        i0, i1, j0, j1, k0, k1 = all_blocks[i]
        for j in range(i + 1, num_blocks):
            i0_j, i1_j, j0_j, j1_j, k0_j, k1_j = all_blocks[j]
            # Check if they share any position
            i_overlap = max(i0, i0_j) <= min(i1, i1_j)
            j_overlap = max(j0, j0_j) <= min(j1, j1_j)
            k_overlap = max(k0, k0_j) <= min(k1, k1_j)
            if i_overlap and j_overlap and k_overlap:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Find maximum independent set using branch and bound
    best_set = []
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    # Precompute degree for each node
    degrees = [popcount(conflict_mask[i]) for i in range(num_blocks)]
    
    def backtrack(available_mask, current_set, current_size):
        nonlocal best_set, best_size
        if time.time() > deadline - 5:
            return
            
        # Upper bound: current size + max possible additional
        remaining = popcount(available_mask)
        if current_size + remaining <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        # Choose the node with maximum degree in the remaining graph
        max_deg = -1
        best_idx = -1
        for idx in range(num_blocks):
            if (available_mask >> idx) & 1:
                # Count neighbors in available set
                deg = popcount(conflict_mask[idx] & available_mask)
                if deg > max_deg:
                    max_deg = deg
                    best_idx = idx
                    
        if best_idx == -1:
            return
            
        bit = 1 << best_idx
        remaining_mask = available_mask & ~bit
        
        # Option 1: Include best_idx
        new_available = remaining_mask & ~conflict_mask[best_idx]
        current_set.append(best_idx)
        backtrack(new_available, current_set, current_size + 1)
        current_set.pop()
        
        # Option 2: Exclude best_idx
        backtrack(remaining_mask, current_set, current_size)
        
    backtrack((1 << num_blocks) - 1, [], 0)
    
    selected_indices = best_set
    
    # Build the decomposition
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    for idx in selected_indices:
        i0, i1, j0, j1, k0, k1 = all_blocks[idx]
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
        
        # Strassen 7 terms for 2x2x2 block
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
        
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    # Add schoolbook for remaining uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
                
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
            
    # Verify correctness
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
