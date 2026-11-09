import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = 128
    block_size = 16
    n_blocks = n // block_size  # 8 blocks
    
    # Precompute all non-constant 4-term progressions
    aps = []
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    # For each block, identify which APs are entirely within that block
    block_internal_aps = [[] for _ in range(n_blocks)]
    block_cross_aps = []  # APs that cross block boundaries
    
    for idx, (a, b, c, d) in enumerate(aps):
        blocks_involved = set([a // block_size, b // block_size, c // block_size, d // block_size])
        if len(blocks_involved) == 1:
            block_internal_aps[list(blocks_involved)[0]].append(idx)
        else:
            block_cross_aps.append(idx)
    
    # Precompute valid patterns for each block given K colors
    # A pattern is valid if all internal APs use >= 3 distinct colors
    valid_patterns_by_K = {}
    
    def get_valid_patterns(K):
        if K in valid_patterns_by_K:
            return valid_patterns_by_K[K]
        
        valid = []
        # For block 0, positions are 0..15
        # Internal APs within a block of 16 positions
        # AP (a, a+d, a+2d, a+3d) where all in [0,15]
        internal_aps_local = []
        for d in range(1, 6):  # max d for block of 16: 15//3=5
            for a in range(16 - 3 * d):
                internal_aps_local.append((a, a + d, a + 2 * d, a + 3 * d))
        
        # Enumerate all K^16 patterns? Too many for K>=4
        # Instead, use constraint propagation or just check on the fly
        # For small K (4), K^16 = 4^16 = 4.3B - too many
        # Better approach: generate random valid patterns or use backtracking
        
        # Use backtracking to find valid patterns
        def backtrack(pos, pattern):
            if pos == block_size:
                valid.append(tuple(pattern))
                return True
            
            for color in range(K):
                pattern[pos] = color
                # Check if any internal AP ending at pos is violated
                ok = True
                for (a, b, c, d) in internal_aps_local:
                    if d == pos:  # AP ends at current position
                        if a >= 0 and b >= 0 and c >= 0:
                            colors = [pattern[a], pattern[b], pattern[c], color]
                            if len(set(colors)) < 3:
                                ok = False
                                break
                if ok:
                    if backtrack(pos + 1, pattern):
                        # Continue to find more solutions, but limit count
                        if len(valid) >= 10000:
                            return False
                pattern[pos] = -1
            
            return len(valid) < 10000
        
        pattern = [-1] * block_size
        backtrack(0, pattern)
        valid_patterns_by_K[K] = valid
        return valid
    
    # Function to check if a full coloring is valid
    def count_bad(coloring):
        c = np.array(coloring, dtype=np.int32)
        a_colors = c[aps_arr[:, 0]]
        b_colors = c[aps_arr[:, 1]]
        c_colors = c[aps_arr[:, 2]]
        d_colors = c[aps_arr[:, 3]]
        
        abc_distinct = (a_colors != b_colors) & (a_colors != c_colors) & (b_colors != c_colors)
        abd_distinct = (a_colors != b_colors) & (a_colors != d_colors) & (b_colors != d_colors)
        acd_distinct = (a_colors != c_colors) & (a_colors != d_colors) & (c_colors != d_colors)
        bcd_distinct = (b_colors != c_colors) & (b_colors != d_colors) & (c_colors != d_colors)
        
        has_three_distinct = abc_distinct | abd_distinct | acd_distinct | bcd_distinct
        bad = ~has_three_distinct
        return np.sum(bad)
    
    # Strategy: Try K=4, 5, 6 colors
    # Use block decomposition with local optimization
    
    for K in range(4, 8):
        if time.time() > deadline:
            break
        
        found = None
        
        # Get valid patterns for this K
        valid_patterns = get_valid_patterns(K)
        
        if len(valid_patterns) == 0:
            continue
        
        # For each block, we'll assign a pattern from valid_patterns
        # Then check cross-boundary constraints
        
        # Use local search over block assignments
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Initialize: random valid pattern per block
            block_assignments = []
            for b in range(n_blocks):
                if valid_patterns:
                    block_assignments.append(random.choice(valid_patterns))
                else:
                    block_assignments.append(tuple(random.randint(0, K - 1) for _ in range(block_size)))
            
            def get_full_coloring(assignments):
                coloring = []
                for b in range(n_blocks):
                    coloring.extend(assignments[b])
                return coloring
            
            current_coloring = get_full_coloring(block_assignments)
            current_bad = count_bad(current_coloring)
            
            if current_bad == 0:
                found = current_coloring
                break
            
            # Local search: modify one block at a time
            no_improve = 0
            max_no_improve = 100
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                if current_bad == 0:
                    found = current_coloring
                    break
                
                # Pick a random block to modify
                b = random.randint(0, n_blocks - 1)
                
                # Try all valid patterns for this block
                best_pattern = block_assignments[b]
                best_bad = current_bad
                
                for pattern in valid_patterns:
                    if pattern == block_assignments[b]:
                        continue
                    
                    new_assignments = block_assignments[:]
                    new_assignments[b] = pattern
                    new_coloring = get_full_coloring(new_assignments)
                    new_bad = count_bad(new_coloring)
                    
                    if new_bad < best_bad:
                        best_bad = new_bad
                        best_pattern = pattern
                
                if best_pattern == block_assignments[b]:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
                    
                    # Random change to escape
                    if valid_patterns:
                        block_assignments[b] = random.choice(valid_patterns)
                    else:
                        block_assignments[b] = tuple(random.randint(0, K - 1) for _ in range(block_size))
                    
                    current_coloring = get_full_coloring(block_assignments)
                    current_bad = count_bad(current_coloring)
                    continue
                
                block_assignments[b] = best_pattern
                current_coloring = get_full_coloring(block_assignments)
                current_bad = best_bad
                no_improve = 0
            
            if found:
                break
        
        if found:
            return found
    
    # Fallback: direct local search
    for restart in range(100):
        if time.time() > deadline:
            break
        
        c = [random.randint(0, 7) for _ in range(n)]
        current_bad = count_bad(c)
        
        if current_bad == 0:
            return c
        
        no_improve = 0
        for step in range(20000):
            if time.time() > deadline:
                break
            
            if current_bad == 0:
                return c
            
            pos = random.randint(0, n - 1)
            old_color = c[pos]
            
            best_new_color = old_color
            best_new_bad = current_bad
            
            for new_color in range(8):
                if new_color == old_color:
                    continue
                c[pos] = new_color
                new_bad = count_bad(c)
                c[pos] = old_color
                
                if new_bad < best_new_bad:
                    best_new_bad = new_bad
                    best_new_color = new_color
            
            if best_new_color == old_color:
                no_improve += 1
                if no_improve > 500:
                    break
                new_color = random.randint(0, 7)
                if new_color == old_color:
                    continue
                c[pos] = new_color
                current_bad = count_bad(c)
            else:
                c[pos] = best_new_color
                current_bad = best_new_bad
                no_improve = 0
    
    # Final fallback: greedy
    colors = []
    for x in range(128):
        for color in range(128):
            valid = True
            for d in range(1, (x) // 3 + 1):
                if x - 3 * d < 0:
                    continue
                c0 = colors[x - 3 * d]
                c1 = colors[x - 2 * d]
                c2 = colors[x - d]
                if len({c0, c1, c2, color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    
    return colors
