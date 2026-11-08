import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses optimal packing of Strassen-style blocks + schoolbook for rest.
    Searches over block configurations to minimize total rank.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

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
        """Add a 2x2x2 Strassen block (7 terms)."""
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

    def add_schoolbook(i, j, k):
        add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    def build_decomposition(blocks):
        """
        Build decomposition from a list of blocks.
        Each block is (i1, i2, j1, j2, k1, k2) for a 2x2x2 Strassen block.
        Returns (U, V, W, rank) or None if invalid.
        """
        nonlocal U_cols, V_cols, W_cols
        U_cols = []
        V_cols = []
        W_cols = []
        
        covered = set()
        for block in blocks:
            i1, i2, j1, j2, k1, k2 = block
            add_strassen_block(i1, i2, j1, j2, k1, k2)
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        covered.add((i, j, k))
        
        # Add schoolbook for uncovered
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        add_schoolbook(i, j, k)
        
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
        
        return U, V, W, rank

    # Generate all possible 2x2x2 Strassen blocks
    # i pairs: (0,1), (1,2)
    # j pairs: (0,1), (1,2), (2,3), (3,4), (4,5)
    # k pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)
    
    i_pairs = [(0, 1), (1, 2)]
    j_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    k_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    
    all_blocks = []
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                all_blocks.append((i1, i2, j1, j2, k1, k2))
    
    # We need to find a subset of blocks that covers as many triples as possible
    # Each block covers 8 triples and costs 7 terms (saves 1 vs schoolbook)
    # But blocks can overlap in coverage, so we need to be careful
    
    # Strategy: Try all subsets of up to 8 blocks (since max 6 non-overlapping in j,k dims)
    # Actually, let's be smarter: use greedy + local search
    
    best_rank = 999
    best_blocks = None
    
    # First, try the known good configurations
    known_configs = [
        # Current: 6 blocks with alternating i
        [(0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 2, 3), (0, 1, 0, 1, 4, 5),
         (0, 1, 2, 3, 0, 1), (0, 1, 2, 3, 2, 3), (0, 1, 2, 3, 4, 5)],
        # All i=(0,1)
        [(0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 2, 3), (0, 1, 0, 1, 4, 5),
         (0, 1, 2, 3, 0, 1), (0, 1, 2, 3, 2, 3), (0, 1, 2, 3, 4, 5)],
        # Mixed i pairs
        [(0, 1, 0, 1, 0, 1), (1, 2, 0, 1, 2, 3), (0, 1, 0, 1, 4, 5),
         (1, 2, 2, 3, 0, 1), (0, 1, 2, 3, 2, 3), (1, 2, 2, 3, 4, 5)],
        [(1, 2, 0, 1, 0, 1), (0, 1, 0, 1, 2, 3), (1, 2, 0, 1, 4, 5),
         (0, 1, 2, 3, 0, 1), (1, 2, 2, 3, 2, 3), (0, 1, 2, 3, 4, 5)],
    ]
    
    for config in known_configs:
        if time.time() > deadline:
            break
        U, V, W, rank = build_decomposition(config)
        if rank < best_rank:
            best_rank = rank
            best_blocks = config
    
    # Now try to find better configurations via search
    # Each block covers 8 triples. We have 105 total triples.
    # A block "saves" 1 term if all 8 of its triples are not already covered.
    # If some are already covered, the savings are reduced.
    
    # Greedy approach: start with no blocks, add the block that maximizes new coverage
    def greedy_search(max_blocks=10):
        """Greedy search for best block combination."""
        nonlocal best_rank, best_blocks
        
        covered = set()
        blocks_used = []
        
        for _ in range(max_blocks):
            if time.time() > deadline:
                break
            
            best_block = None
            best_new_coverage = -1
            
            for block in all_blocks:
                i1, i2, j1, j2, k1, k2 = block
                new_count = 0
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            if (i, j, k) not in covered:
                                new_count += 1
                
                # Only add if it covers at least 1 new triple
                # The block costs 7 terms but saves up to 8 schoolbook terms
                # Net savings = new_count - 7 (if new_count > 7, we save terms)
                # Actually: without block, we'd use new_count schoolbook terms
                # With block, we use 7 terms for these + whatever else
                # So net change = 7 - new_count
                # We want to minimize total, so add block if 7 < new_count
                # i.e., if new_count > 7, we save (new_count - 7) terms
                
                if new_count > 7 and new_count > best_new_coverage:
                    best_new_coverage = new_count
                    best_block = block
            
            if best_block is None:
                break
            
            blocks_used.append(best_block)
            i1, i2, j1, j2, k1, k2 = best_block
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        covered.add((i, j, k))
        
        if blocks_used:
            U, V, W, rank = build_decomposition(blocks_used)
            if rank < best_rank:
                best_rank = rank
                best_blocks = blocks_used
        
        return best_blocks
    
    greedy_search()
    
    # Try local search: start from best, try swapping blocks
    if best_blocks and time.time() < deadline - 20:
        def local_search(start_blocks, iterations=50):
            nonlocal best_rank, best_blocks
            
            current_blocks = list(start_blocks)
            current_rank = best_rank
            
            for it in range(iterations):
                if time.time() > deadline - 10:
                    break
                
                improved = False
                
                # Try removing a block
                for idx in range(len(current_blocks)):
                    test_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                    if not test_blocks:
                        U, V, W, rank = build_decomposition([])
                    else:
                        U, V, W, rank = build_decomposition(test_blocks)
                    if rank < current_rank:
                        current_blocks = test_blocks
                        current_rank = rank
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try adding a block
                for block in all_blocks:
                    if block in current_blocks:
                        continue
                    test_blocks = current_blocks + [block]
                    U, V, W, rank = build_decomposition(test_blocks)
                    if rank < current_rank:
                        current_blocks = test_blocks
                        current_rank = rank
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try swapping a block
                for idx in range(len(current_blocks)):
                    for new_block in all_blocks:
                        if new_block in current_blocks:
                            continue
                        test_blocks = current_blocks[:idx] + [new_block] + current_blocks[idx+1:]
                        U, V, W, rank = build_decomposition(test_blocks)
                        if rank < current_rank:
                            current_blocks = test_blocks
                            current_rank = rank
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            if current_rank < best_rank:
                best_rank = current_rank
                best_blocks = current_blocks
        
        local_search(best_blocks)
    
    # Final build with best configuration
    if best_blocks is None:
        # Fallback: no blocks, all schoolbook
        U, V, W, rank = build_decomposition([])
    else:
        U, V, W, rank = build_decomposition(best_blocks)
    
    return {"U": U, "V": V, "W": W}
