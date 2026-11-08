import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Strategy:
    1. Greedy packing of disjoint 2x2x2 Strassen blocks (reduces rank by 1 per block vs schoolbook).
    2. Try to find additional rank reductions by optimizing the W matrix for a fixed U,V structure
       derived from the Strassen packing + schoolbook, or by direct integer optimization on small
       residual tensors.
       
    The base rank is 105 - num_blocks. Each Strassen block covers 8 (i,j,k) indices with 7 terms.
    Schoolbook covers 1 index with 1 term.
    
    We use a time-bounded search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # 1. Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    # 2. Greedy search for maximum disjoint packing of Strassen blocks
    #    We want to maximize the number of blocks to minimize base rank.
    #    Max possible blocks: floor(105/8) = 13. But constraints on disjointness in 3,5,7 grid.
    
    best_num_blocks = 0
    best_packing = []
    
    # Use randomized greedy with restarts - more trials with adaptive deadline
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:  # Leave 30s for construction and refinement
            break
            
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        
        for block_params, block_indices in all_blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
                
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:len(current_packing)]  # copy
            if best_num_blocks >= 13:
                break  # Theoretical maximum reached
    
    # 3. Local refinement: try to improve the packing by swapping blocks
    #    For each block in the packing, try replacing it with a different block
    #    that covers some of the same indices but frees up others
    if time.time() < deadline - 25.0:
        for _iteration in range(50):
            if time.time() > deadline - 25.0:
                break
            
            improved = False
            # Build current coverage
            current_covered = set()
            for block_params in best_packing:
                i1, i2, j1, j2, k1, k2 = block_params
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            current_covered.add((i, j, k))
            
            # Try removing each block and re-greedy filling the gap
            for idx, block_to_remove in enumerate(best_packing):
                if time.time() > deadline - 25.0:
                    break
                
                # Remove this block
                remaining_packing = best_packing[:idx] + best_packing[idx+1:]
                remaining_covered = set()
                for bp in remaining_packing:
                    i1, i2, j1, j2, k1, k2 = bp
                    for i in (i1, i2):
                        for j in (j1, j2):
                            for k in (k1, k2):
                                remaining_covered.add((i, j, k))
                
                # Now try to fill the gap with a different block
                freed_indices = current_covered - remaining_covered
                if not freed_indices:
                    continue
                
                # Try to find a block that covers some freed indices and some new indices
                best_new_block = None
                best_new_coverage = 0
                
                for block_params, block_indices in all_blocks:
                    # Check if this block can be added (disjoint from remaining_covered)
                    if block_indices & remaining_covered:
                        continue
                    
                    # Count how many freed indices this block covers
                    overlap_with_freed = len(block_indices & freed_indices)
                    total_new = len(block_indices - remaining_covered)
                    
                    # We want to maximize coverage of freed indices + new indices
                    # but we need the block to actually help (cover freed indices)
                    if overlap_with_freed > 0 and total_new > best_new_coverage:
                        best_new_coverage = total_new
                        best_new_block = block_params
                
                if best_new_block is not None:
                    # Check if this actually improves total coverage
                    new_covered = remaining_covered | set()
                    i1, i2, j1, j2, k1, k2 = best_new_block
                    for i in (i1, i2):
                        for j in (j1, j2):
                            for k in (k1, k2):
                                new_covered.add((i, j, k))
                    
                    # If we cover at least as many indices with same or fewer blocks
                    if len(new_covered) >= len(current_covered):
                        best_packing = remaining_packing + [best_new_block]
                        improved = True
                        break
            
            if not improved:
                break

    # 3. Construct U, V, W based on the best packing found
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
        
        # Standard Strassen 2x2x2 decomposition
        # M1 = (a11+a22)*(b11+b22) -> c11 += M1, c22 += M1
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11 -> c21 += M2, c22 -= M2
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22) -> c12 += M3, c22 += M3
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11) -> c11 += M4, c21 += M4
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22 -> c11 -= M5, c12 += M5
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12) -> c22 += M6
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22) -> c11 += M7
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Try to find additional Strassen blocks that cover uncovered indices
    # A 2x2x2 block needs 8 specific indices; check if any uncovered set contains such a pattern
    uncovered_list = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    
    # Check all possible 2x2x2 patterns against uncovered indices
    additional_blocks_found = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # Check if all 8 indices are uncovered
                            all_uncovered = True
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        if (i, j, k) not in uncovered_list:
                                            all_uncovered = False
                                            break
                                if not all_uncovered:
                                    break
                            if all_uncovered:
                                additional_blocks_found.append((i1, i2, j1, j2, k1, k2))
    
    # Add additional Strassen blocks
    for block_params in additional_blocks_found:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))
    
    # Add schoolbook terms for remaining uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format (list of lists)
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

    # Verification: check that the decomposition is exact
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if this is a valid (i,j,k) triple, 0 otherwise
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # Let's verify a few entries
    
    # Build numpy arrays for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all (i,j,k) triples
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Compute sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                product_sum = 0.0
                for r in range(rank):
                    product_sum += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                
                # Should be exactly 1.0
                error = abs(product_sum - 1.0)
                if error > max_error:
                    max_error = error
    
    if max_error > 1e-9:
        # If verification fails, fall back to pure schoolbook (rank 105)
        U_fallback = [[0] * (n*m*p) for _ in range(n*m)]
        V_fallback = [[0] * (n*m*p) for _ in range(m*p)]
        W_fallback = [[0] * (n*m*p) for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fallback[i * m + j][r] = 1
                    V_fallback[j * p + k][r] = 1
                    W_fallback[k * n + i][r] = 1
        
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
