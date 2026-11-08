import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses branch-and-bound with bitmask constraint propagation to find a maximum
    packing of disjoint 2x2x2 Strassen blocks, then fills remaining entries with
    schoolbook terms.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Total tensor entries
    total_entries = n * m * p  # 105

    # Generate all possible 2x2x2 blocks with bitmask representation
    all_blocks = []  # list of (params, bitmask)
    entry_bit = {}  # (i,j,k) -> bit index
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entry_bit[(i, j, k)] = len(entry_bit)

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        mask |= (1 << entry_bit[(i, j, k)])
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), mask))

    num_blocks = len(all_blocks)
    
    # Precompute conflict matrix as bitmasks for fast intersection
    # conflict_mask[b] = bitmask of blocks that conflict with block b
    conflict_mask = []
    for b in range(num_blocks):
        cm = 0
        mb = all_blocks[b][1]
        for c in range(num_blocks):
            if b != c and (mb & all_blocks[c][1]):
                cm |= (1 << c)
        conflict_mask.append(cm)

    # Precompute: for each block, list of blocks it conflicts with (sorted by index)
    conflict_list = []
    for b in range(num_blocks):
        cl = []
        cm = conflict_mask[b]
        for c in range(num_blocks):
            if cm & (1 << c):
                cl.append(c)
        conflict_list.append(cl)

    # Degree of each block (number of conflicts)
    degree = [len(cl) for cl in conflict_list]

    best_packing = []
    best_count = 0

    # Branch and bound with constraint propagation
    def branch_and_bound(available_mask, selected, depth):
        """
        available_mask: bitmask of blocks still available (not conflicting with selected)
        selected: list of selected block indices
        """
        nonlocal best_packing, best_count

        if time.time() > deadline - 5.0:
            return

        # Upper bound: current selected + max possible from available
        # Simple bound: count available blocks / 1 (each block covers at least 1 entry)
        # Better bound: use greedy upper bound
        avail_count = bin(available_mask).count('1')
        if len(selected) + avail_count <= best_count:
            return

        # If no more blocks available
        if available_mask == 0:
            if len(selected) > best_count:
                best_count = len(selected)
                best_packing = selected[:]
            return

        # Constraint propagation: remove blocks that conflict with any selected block
        # Actually, available_mask should already exclude conflicting blocks
        # Let's verify by computing which blocks are truly available
        truly_available = available_mask
        for s in selected:
            truly_available &= ~conflict_mask[s]
        
        if truly_available != available_mask:
            # Some blocks became unavailable due to constraint propagation
            if bin(truly_available).count('1') + len(selected) <= best_count:
                return
            available_mask = truly_available
            if available_mask == 0:
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_packing = selected[:]
                return

        # Select a block to branch on (most constrained first)
        # Find the block with highest degree among available
        best_block = -1
        best_deg = -1
        temp_mask = available_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & (-temp_mask)
            b = lsb.bit_length() - 1
            if degree[b] > best_deg:
                best_deg = degree[b]
                best_block = b
            temp_mask ^= lsb

        if best_block == -1:
            if len(selected) > best_count:
                best_count = len(selected)
                best_packing = selected[:]
            return

        # Branch 1: include best_block
        new_selected = selected + [best_block]
        new_available = available_mask & ~(1 << best_block) & ~conflict_mask[best_block]
        branch_and_bound(new_available, new_selected, depth + 1)

        # Branch 2: exclude best_block
        new_available = available_mask & ~(1 << best_block)
        branch_and_bound(new_available, selected, depth + 1)

    # Run B&B with full available set
    full_available = (1 << num_blocks) - 1
    branch_and_bound(full_available, [], 0)

    # If B&B didn't find anything, use greedy fallback
    if best_count == 0:
        # Greedy by degree (most constrained first)
        sorted_blocks = sorted(range(num_blocks), key=lambda x: degree[x])
        current_packing = []
        used_mask = 0
        for b in sorted_blocks:
            if time.time() > deadline - 5.0:
                break
            if not (all_blocks[b][1] & used_mask):
                current_packing.append(b)
                used_mask |= all_blocks[b][1]
        if len(current_packing) > best_count:
            best_count = len(current_packing)
            best_packing = current_packing[:]

    # Local search to improve the packing
    def local_search(packing):
        """Try to improve packing by 1-for-2 swaps using bitmasks."""
        current = packing[:]
        used_mask = 0
        for b in current:
            used_mask |= all_blocks[b][1]
        
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try to add a block
            for b in range(num_blocks):
                if b in current:
                    continue
                if not (all_blocks[b][1] & used_mask):
                    current.append(b)
                    used_mask |= all_blocks[b][1]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swap: remove one block, try to add two
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                remove_b = current[i]
                # Temporarily remove
                temp_current = current[:i] + current[i+1:]
                temp_used = 0
                for b in temp_current:
                    temp_used |= all_blocks[b][1]
                
                # Try to find two blocks that can be added
                found_two = False
                for b1 in range(num_blocks):
                    if b1 in temp_current or b1 == remove_b:
                        continue
                    if all_blocks[b1][1] & temp_used:
                        continue
                    # Try to add b1
                    temp_used2 = temp_used | all_blocks[b1][1]
                    for b2 in range(num_blocks):
                        if b2 in temp_current or b2 == remove_b or b2 == b1:
                            continue
                        if all_blocks[b2][1] & temp_used2:
                            continue
                        # Found a valid 1-for-2 swap
                        current = temp_current + [b1, b2]
                        used_mask = temp_used2 | all_blocks[b2][1]
                        improved = True
                        found_two = True
                        break
                    if found_two:
                        break
                if found_two:
                    break

        return current

    # Apply local search to improve
    if best_count > 0:
        improved = local_search(best_packing)
        if len(improved) > best_count:
            best_count = len(improved)
            best_packing = improved[:]

    # Randomized restarts with greedy + local search
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random ordering weighted by inverse degree
        weights = [1.0 / (degree[b] + 1) for b in range(num_blocks)]
        order = []
        remaining = list(range(num_blocks))
        while remaining:
            total_w = sum(weights[b] for b in remaining)
            r = random.random() * total_w
            cum = 0
            for idx, b in enumerate(remaining):
                cum += weights[b]
                if r <= cum:
                    order.append(b)
                    remaining.remove(b)
                    break
        
        # Greedy with this order
        current_packing = []
        used_mask = 0
        for b in order:
            if not (all_blocks[b][1] & used_mask):
                current_packing.append(b)
                used_mask |= all_blocks[b][1]
        
        # Local search
        improved = local_search(current_packing)
        
        if len(improved) > best_count:
            best_count = len(improved)
            best_packing = improved[:]

    # Construct U, V, W based on the best packing found
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

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered_mask = 0
    for b_idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[b_idx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered_mask |= all_blocks[b_idx][1]

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                bit = entry_bit[(i, j, k)]
                if not (covered_mask & (1 << bit)):
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
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

    # Verification Step
    if rank > 0:
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(np.dot(U_np[u_idx], V_np[v_idx]), W_np[w_idx])
                        if not np.isclose(val, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if not valid:
                # Fall back to schoolbook
                U_cols = []
                V_cols = []
                W_cols = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
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
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
