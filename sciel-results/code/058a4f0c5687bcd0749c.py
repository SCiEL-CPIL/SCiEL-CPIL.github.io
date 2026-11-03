import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Generate all 2x2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1) for k in (k0, k1))
                all_blocks.append((i0, i1, j0, j1, k0, k1, entries))
    
    num_blocks = len(all_blocks)
    
    # Build conflict graph using bitmasks for speed
    conflict_mask = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][6] & all_blocks[j][6]:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    best_set = []
    best_size = 0
    time_check_counter = [0]
    
    def get_bit_count(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def backtrack(available_mask, current_set):
        nonlocal best_set, best_size
        time_check_counter[0] += 1
        if time_check_counter[0] % 50 == 0 and time.time() > deadline:
            return
        
        avail_count = get_bit_count(available_mask)
        if len(current_set) + avail_count <= best_size:
            return
        
        if available_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Choose the most constrained available block
        max_conf = -1
        best_idx = -1
        mask = available_mask
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            mask ^= lsb
            conf_in_avail = get_bit_count(conflict_mask[idx] & available_mask) - 1
            if conf_in_avail > max_conf:
                max_conf = conf_in_avail
                best_idx = idx
        
        if best_idx < 0:
            return
        
        remaining_mask = available_mask & ~(1 << best_idx)
        
        # Option 1: include block best_idx
        current_set.append(best_idx)
        new_available = remaining_mask & ~conflict_mask[best_idx]
        backtrack(new_available, current_set)
        current_set.pop()
        
        # Option 2: exclude block best_idx
        backtrack(remaining_mask, current_set)
        
    backtrack((1 << num_blocks) - 1, [])
    
    selected_indices = best_set
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    if not selected_indices:
        for i in range(n):
            for j in range(m):
                for k in range(p):
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    for idx in selected_indices:
        i0, i1, j0, j1, k0, k1, _ = all_blocks[idx]
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
        
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
        
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    # NEW: Try to pack 2x2x1, 2x1x2, 1x2x2 blocks into remaining space
    # These have rank 2 and cover 4 terms each, saving 2 per block
    remaining_terms = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    remaining_terms.add((i, j, k))
    
    # Generate all possible 2x2x1, 2x1x2, 1x2x2 blocks in remaining space
    small_blocks = []
    
    # 2x2x1 blocks (i0,i1,j0,j1,k)
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k in range(p):
                i1, j1 = i0 + 1, j0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1))
                if entries.issubset(remaining_terms):
                    small_blocks.append(('2x2x1', i0, i1, j0, j1, k, k, entries))
    
    # 2x1x2 blocks (i0,i1,j,k0,k1)
    for i0 in range(n - 1):
        for j in range(m):
            for k0 in range(p - 1):
                i1, k1 = i0 + 1, k0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for k in (k0, k1))
                if entries.issubset(remaining_terms):
                    small_blocks.append(('2x1x2', i0, i1, j, j, k0, k1, entries))
    
    # 1x2x2 blocks (i,j0,j1,k0,k1)
    for i in range(n):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                j1, k1 = j0 + 1, k0 + 1
                entries = frozenset((i, j, k) for j in (j0, j1) for k in (k0, k1))
                if entries.issubset(remaining_terms):
                    small_blocks.append(('1x2x2', i, i, j0, j1, k0, k1, entries))
    
    # Build conflict graph for small blocks
    num_small = len(small_blocks)
    small_conflict = [0] * num_small
    for i in range(num_small):
        for j in range(i + 1, num_small):
            if small_blocks[i][7] & small_blocks[j][7]:
                small_conflict[i] |= (1 << j)
                small_conflict[j] |= (1 << i)
    
    # Greedy selection of non-conflicting small blocks
    selected_small = []
    available_small = (1 << num_small) - 1
    
    while available_small and time.time() < deadline:
        # Find block with minimum conflicts among available
        best_idx = -1
        min_conf = float('inf')
        mask = available_small
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            mask ^= lsb
            conf = get_bit_count(small_conflict[idx] & available_small)
            if conf < min_conf:
                min_conf = conf
                best_idx = idx
        
        if best_idx < 0:
            break
        
        selected_small.append(best_idx)
        available_small &= ~(1 << best_idx)
        available_small &= ~small_conflict[best_idx]
    
    # Add selected small blocks to decomposition
    for idx in selected_small:
        shape_type, i0, i1, j0, j1, k0, k1, _ = small_blocks[idx]
        
        if shape_type == '2x2x1':
            # Fixed k, varying i and j
            a00 = i0 * m + j0
            a01 = i0 * m + j1
            a10 = i1 * m + j0
            a11 = i1 * m + j1
            
            b00 = j0 * p + k0
            b01 = j1 * p + k0
            
            c00 = k0 * n + i0
            c01 = k0 * n + i1
            
            # 2x2x1 rank-2 decomposition
            # Term (i0,j0,k): U=a00, V=b00, W=c00
            # Term (i0,j1,k): U=a01, V=b01, W=c00
            # Term (i1,j0,k): U=a10, V=b00, W=c01
            # Term (i1,j1,k): U=a11, V=b01, W=c01
            
            # Use rank-2: two terms
            # U1: a00, a11 ; V1: b00, b01 ; W1: c00, c01
            U_cols.append({a00: 1, a11: 1})
            V_cols.append({b00: 1, b01: 1})
            W_cols.append({c00: 1, c01: 1})
            
            # U2: a01, a10 ; V2: b01, b00 ; W2: c00, c01 (with signs)
            U_cols.append({a01: 1, a10: 1})
            V_cols.append({b01: -1, b00: -1})
            W_cols.append({c00: 1, c01: 1})
            
            for i in (i0, i1):
                for j in (j0, j1):
                    covered.add((i, j, k0))
        
        elif shape_type == '2x1x2':
            # Fixed j, varying i and k
            a00 = i0 * m + j0
            a10 = i1 * m + j0
            
            b00 = j0 * p + k0
            b01 = j0 * p + k1
            
            c00 = k0 * n + i0
            c10 = k1 * n + i0
            c01 = k0 * n + i1
            c11 = k1 * n + i1
            
            # 2x1x2 rank-2 decomposition
            # Term (i0,j,k0): U=a00, V=b00, W=c00
            # Term (i0,j,k1): U=a00, V=b01, W=c10
            # Term (i1,j,k0): U=a10, V=b00, W=c01
            # Term (i1,j,k1): U=a10, V=b01, W=c11
            
            U_cols.append({a00: 1, a10: 1})
            V_cols.append({b00: 1})
            W_cols.append({c00: 1, c11: 1})
            
            U_cols.append({a00: 1})
            V_cols.append({b01: 1})
            W_cols.append({c10: 1, c01: -1})
            
            for i in (i0, i1):
                for k in (k0, k1):
                    covered.add((i, j0, k))
        
        elif shape_type == '1x2x2':
            # Fixed i, varying j and k
            a00 = i0 * m + j0
            a01 = i0 * m + j1
            
            b00 = j0 * p + k0
            b01 = j0 * p + k1
            b10 = j1 * p + k0
            b11 = j1 * p + k1
            
            c00 = k0 * n + i0
            c01 = k1 * n + i0
            
            # 1x2x2 rank-2 decomposition
            # Term (i,j0,k0): U=a00, V=b00, W=c00
            # Term (i,j0,k1): U=a00, V=b01, W=c01
            # Term (i,j1,k0): U=a01, V=b10, W=c00
            # Term (i,j1,k1): U=a01, V=b11, W=c01
            
            U_cols.append({a00: 1})
            V_cols.append({b00: 1, b01: 1})
            W_cols.append({c00: 1, c01: 1})
            
            U_cols.append({a01: 1})
            V_cols.append({b10: 1, b11: 1})
            W_cols.append({c00: 1, c01: -1})
            
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i0, j, k))
    
    # Add remaining uncovered terms as schoolbook
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
            
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
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
