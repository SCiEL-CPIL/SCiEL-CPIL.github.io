import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Strategy:
    1. Find maximum disjoint packing of 2x2x2 Strassen blocks using backtracking + randomized greedy.
    2. For the residual (uncovered) indices, try to find additional rank reductions by
       checking if any subset of residual indices can be decomposed at lower rank than schoolbook.
       
    Rank = 7 * num_blocks + residual_rank
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    total_indices = n * m * p  # 105

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
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | {(i, j, k)}
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    # 2. Find maximum disjoint packing
    best_num_blocks = 0
    best_packing = []
    
    # First, try randomized greedy with many restarts
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
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
            best_packing = current_packing[:]
            if best_num_blocks >= 13:
                break

    # If we haven't found 13, try a backtracking approach to see if 13 is possible
    if best_num_blocks < 13 and time.time() < deadline - 60.0:
        # Try to find a packing of 13 blocks via backtracking
        # This is a set packing problem. We'll use a limited backtracking search.
        def backtrack(idx, used, current):
            if time.time() > deadline - 60.0:
                return None
            if len(current) + (len(all_blocks) - idx) <= best_num_blocks:
                return None
            if idx >= len(all_blocks):
                return None
            params, indices = all_blocks[idx]
            # Try including this block
            if not (indices & used):
                result = backtrack(idx + 1, used | set(indices), current + [params])
                if result is not None:
                    return result
            # Try excluding this block
            result = backtrack(idx + 1, used, current)
            if result is not None:
                return result
            return None
        
        # Shuffle to help backtracking
        random.shuffle(all_blocks)
        result = backtrack(0, set(), [])
        if result is not None and len(result) > best_num_blocks:
            best_num_blocks = len(result)
            best_packing = result

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

    # Get uncovered indices
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered.append((i, j, k))

    # 4. Try to reduce rank of residual terms
    # For small numbers of uncovered indices, try to find better decompositions
    # Check if any 2x2x2 sub-block exists in the uncovered set
    residual_rank = len(uncovered)  # Start with schoolbook
    
    # Try to find Strassen blocks within the uncovered set
    def can_form_strassen_block(indices_set):
        """Check if a 2x2x2 Strassen block can be formed from the given index set."""
        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                for j1 in range(m - 1):
                    for j2 in range(j1 + 1, m):
                        for k1 in range(p - 1):
                            for k2 in range(k1 + 1, p):
                                needed = set()
                                for i in (i1, i2):
                                    for j in (j1, j2):
                                        for k in (k1, k2):
                                            needed.add((i, j, k))
                                if needed <= indices_set:
                                    return (i1, i2, j1, j2, k1, k2)
        return None

    # Greedily find Strassen blocks in the uncovered set
    remaining_uncovered = set(uncovered)
    additional_blocks = []
    while True:
        if time.time() > deadline - 20.0:
            break
        block = can_form_strassen_block(remaining_uncovered)
        if block is None:
            break
        additional_blocks.append(block)
        for i in (block[0], block[1]):
            for j in (block[2], block[3]):
                for k in (block[4], block[5]):
                    remaining_uncovered.discard((i, j, k))
    
    # Add additional Strassen blocks found in residual
    for block_params in additional_blocks:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)

    # Add schoolbook terms for truly uncovered indices
    for i, j, k in remaining_uncovered:
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
