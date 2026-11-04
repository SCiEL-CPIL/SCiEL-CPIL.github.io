import time
import random

DIMENSIONS = [3, 5, 7]

def _try_border_rank_perturbation(n, m, p, base_rank, base_U, base_V, base_W, deadline):
    """
    Attempt to reduce rank via border-rank perturbation:
    1. Start from a valid rank-r decomposition
    2. Perturb with small symbolic epsilon
    3. Try to find a rank-(r-1) decomposition that is exact in the limit
    This explores whether the tensor has border rank < rank.
    """
    if time.time() > deadline - 10.0:
        return None
    
    rank = base_rank
    U = [row[:] for row in base_U]
    V = [row[:] for row in base_V]
    W = [row[:] for row in base_W]
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    best_rank = rank
    best_decomp = (U, V, W)
    
    for c1 in range(rank):
        if time.time() > deadline - 5.0:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() > deadline - 5.0:
                break
            
            merged = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = U[i*m+j][c1] * V[j*p+k][c1] * W[k*n+i][c1]
                        val += U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2]
                        if abs(val) > 1e-10:
                            merged[(i,j,k)] = val
            
            if not merged:
                new_U = [row[:c1] + row[c1+1:] for row in U]
                new_V = [row[:c1] + row[c1+1:] for row in V]
                new_W = [row[:c1] + row[c1+1:] for row in W]
                if c2 > c1:
                    new_U = [row[:c1] + row[c1+1:] for row in new_U]
                    new_V = [row[:c1] + row[c1+1:] for row in new_V]
                    new_W = [row[:c1] + row[c1+1:] for row in new_W]
                elif c2 < c1:
                    new_U = [row[:c2] + row[c2+1:] for row in new_U]
                    new_V = [row[:c2] + row[c2+1:] for row in new_V]
                    new_W = [row[:c2] + row[c2+1:] for row in new_W]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0 if (i,j,k) in merged else 0.0
                            actual = 0.0
                            for c in range(len(new_U[0])):
                                actual += new_U[i*m+j][c] * new_V[j*p+k][c] * new_W[k*n+i][c]
                            if abs(actual - expected) > 1e-8:
                                valid = False
                                break
                    if not valid:
                        break
                if valid and len(new_U[0]) < best_rank:
                    best_rank = len(new_U[0])
                    best_decomp = (new_U, new_V, new_W)
                continue
            
            i_vals = set()
            j_vals = set()
            k_vals = set()
            for (i,j,k) in merged:
                i_vals.add(i)
                j_vals.add(j)
                k_vals.add(k)
            
            if len(i_vals) > 1 and len(j_vals) > 1 and len(k_vals) > 1:
                ref_i = min(i_vals)
                ref_j = min(j_vals)
                ref_k = min(k_vals)
                
                if (ref_i, ref_j, ref_k) not in merged:
                    continue
                
                ref_val = merged[(ref_i, ref_j, ref_k)]
                if abs(ref_val) < 1e-10:
                    continue
                
                U_prime = {}
                V_prime = {}
                W_prime = {}
                
                for i in i_vals:
                    if (i, ref_j, ref_k) in merged:
                        U_prime[i] = merged[(i, ref_j, ref_k)] / ref_val
                    else:
                        found = False
                        for j in j_vals:
                            for k in k_vals:
                                if (i, j, k) in merged and (ref_i, j, k) in merged:
                                    if abs(merged[(ref_i, j, k)]) > 1e-10:
                                        U_prime[i] = merged[(i, j, k)] / merged[(ref_i, j, k)]
                                        found = True
                                        break
                            if found:
                                break
                        if not found:
                            U_prime[i] = 0.0
                
                for j in j_vals:
                    if (ref_i, j, ref_k) in merged:
                        V_prime[j] = merged[(ref_i, j, ref_k)] / ref_val
                    else:
                        found = False
                        for i in i_vals:
                            for k in k_vals:
                                if (i, j, k) in merged and (i, ref_j, k) in merged:
                                    if abs(merged[(i, ref_j, k)]) > 1e-10:
                                        V_prime[j] = merged[(i, j, k)] / merged[(i, ref_j, k)]
                                        found = True
                                        break
                            if found:
                                break
                        if not found:
                            V_prime[j] = 0.0
                
                for k in k_vals:
                    if (ref_i, ref_j, k) in merged:
                        W_prime[k] = merged[(ref_i, ref_j, k)] / ref_val
                    else:
                        found = False
                        for i in i_vals:
                            for j in j_vals:
                                if (i, j, k) in merged and (i, j, ref_k) in merged:
                                    if abs(merged[(i, j, ref_k)]) > 1e-10:
                                        W_prime[k] = merged[(i, j, k)] / merged[(i, j, ref_k)]
                                        found = True
                                        break
                            if found:
                                break
                        if not found:
                            W_prime[k] = 0.0
                
                is_rank1 = True
                for i in i_vals:
                    for j in j_vals:
                        for k in k_vals:
                            predicted = U_prime.get(i, 0.0) * V_prime.get(j, 0.0) * W_prime.get(k, 0.0)
                            actual = merged.get((i,j,k), 0.0)
                            if abs(predicted - actual) > 1e-8:
                                is_rank1 = False
                                break
                        if not is_rank1:
                            break
                    if not is_rank1:
                        break
                
                if is_rank1:
                    new_U = [row[:c1] + row[c2+1:] for row in U]
                    new_V = [row[:c1] + row[c2+1:] for row in V]
                    new_W = [row[:c1] + row[c2+1:] for row in W]
                    
                    for i in range(n):
                        for j in range(m):
                            new_U[i*m+j][c1] = U_prime.get(i, 0.0)
                    for j in range(m):
                        for k in range(p):
                            new_V[j*p+k][c1] = V_prime.get(j, 0.0)
                    for k in range(p):
                        for i in range(n):
                            new_W[k*n+i][c1] = W_prime.get(k, 0.0)
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                actual = 0.0
                                for c in range(len(new_U[0])):
                                    actual += new_U[i*m+j][c] * new_V[j*p+k][c] * new_W[k*n+i][c]
                                expected = 1.0 if (i,j,k) in merged else 0.0
                                if abs(actual - expected) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid and len(new_U[0]) < best_rank:
                        best_rank = len(new_U[0])
                        best_decomp = (new_U, new_V, new_W)
    
    if best_rank < rank:
        return best_decomp
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    
    Optimized search:
    1. Precompute all blocks with bitmasks and precomputed priority scores
    2. Fast greedy with precomputed priorities (no per-iteration recomputation)
    3. Efficient local search using incremental mask updates
    4. More trials due to faster per-trial cost
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    total_triples = n * m * p  # 105

    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k

    # Precompute all possible 2x2x2 blocks
    all_blocks = []  # list of (params, bitmask, tuple_of_indices)
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.append(idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, tuple(indices)))

    num_all_blocks = len(all_blocks)

    # Count how many blocks cover each triple
    triple_block_count = [0] * total_triples
    for _, _, indices in all_blocks:
        for idx in indices:
            triple_block_count[idx] += 1

    # Precompute priority score for each block (sum of 1/count for its indices)
    block_priority = []
    for _, _, indices in all_blocks:
        pri = sum(1.0 / triple_block_count[idx] for idx in indices)
        block_priority.append(pri)

    best_num_blocks = 0
    best_packing = []

    def greedy_random():
        """Fast greedy with random ordering."""
        order = list(range(num_all_blocks))
        random.shuffle(order)
        used_mask = 0
        packing = []
        for bi in order:
            bm = all_blocks[bi][1]
            if (bm & used_mask) == 0:
                packing.append(bi)
                used_mask |= bm
        return packing

    def greedy_priority():
        """Greedy selecting by highest priority among available blocks."""
        available = list(range(num_all_blocks))
        used_mask = 0
        packing = []
        while available:
            best_bi = -1
            best_pri = -1
            for bi in available:
                bm = all_blocks[bi][1]
                if (bm & used_mask) == 0:
                    if block_priority[bi] > best_pri:
                        best_pri = block_priority[bi]
                        best_bi = bi
            if best_bi == -1:
                break
            packing.append(best_bi)
            used_mask |= all_blocks[best_bi][1]
            available.remove(best_bi)
        return packing

    def greedy_priority_random():
        """Greedy with priority but random tie-breaking."""
        available = list(range(num_all_blocks))
        random.shuffle(available)
        used_mask = 0
        packing = []
        while available:
            best_bi = -1
            best_pri = -1
            for bi in available:
                bm = all_blocks[bi][1]
                if (bm & used_mask) == 0:
                    if block_priority[bi] > best_pri:
                        best_pri = block_priority[bi]
                        best_bi = bi
            if best_bi == -1:
                break
            packing.append(best_bi)
            used_mask |= all_blocks[best_bi][1]
            available.remove(best_bi)
        return packing

    def local_search(packing, max_iterations=30):
        """Efficient local search: try replacing one block to gain another."""
        if not packing:
            return packing
        
        current_packing = list(packing)
        used_mask = 0
        for bi in current_packing:
            used_mask |= all_blocks[bi][1]
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            for remove_idx in range(len(current_packing)):
                removed_bi = current_packing[remove_idx]
                mask_without = used_mask & ~all_blocks[removed_bi][1]
                
                for bi_add1 in range(num_all_blocks):
                    if bi_add1 in current_packing:
                        continue
                    bm1 = all_blocks[bi_add1][1]
                    if (bm1 & mask_without) != 0:
                        continue
                    for bi_add2 in range(bi_add1 + 1, num_all_blocks):
                        if bi_add2 in current_packing:
                            continue
                        bm2 = all_blocks[bi_add2][1]
                        if (bm2 & mask_without) != 0:
                            continue
                        if (bm1 & bm2) != 0:
                            continue
                        new_packing = current_packing[:remove_idx] + current_packing[remove_idx+1:] + [bi_add1, bi_add2]
                        current_packing = new_packing
                        used_mask = 0
                        for bi in current_packing:
                            used_mask |= all_blocks[bi][1]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current_packing

    # Run many trials
    trial_count = 0
    max_trials = 5000
    for t in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        trial_count += 1
        
        if t % 4 == 0:
            packing = greedy_random()
        elif t % 4 == 1:
            packing = greedy_priority()
        elif t % 4 == 2:
            packing = greedy_priority_random()
        else:
            packing = greedy_random()
            packing = local_search(packing, max_iterations=15)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            
            if best_num_blocks >= 14:
                break

    # Systematic approach: sort by conflict count
    if time.time() < deadline - 15.0:
        conflict_counts = [0] * num_all_blocks
        for i in range(num_all_blocks):
            bm_i = all_blocks[i][1]
            for j in range(i + 1, num_all_blocks):
                if (bm_i & all_blocks[j][1]) != 0:
                    conflict_counts[i] += 1
                    conflict_counts[j] += 1
        
        sorted_indices = sorted(range(num_all_blocks), key=lambda x: conflict_counts[x])
        
        used_mask = 0
        packing = []
        for bi in sorted_indices:
            bm = all_blocks[bi][1]
            if (bm & used_mask) == 0:
                packing.append(bi)
                used_mask |= bm
        
        packing = local_search(packing, max_iterations=25)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing

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
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bi in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[bi][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for idx in all_blocks[bi][2]:
            covered.add(idx)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                idx = triple_to_idx(i, j, k)
                if idx not in covered:
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

    if time.time() < deadline - 10.0:
        improved = _try_border_rank_perturbation(n, m, p, rank, U, V, W, deadline)
        if improved is not None:
            U, V, W = improved
            rank = len(U[0])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
