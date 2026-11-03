import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for safety

    # 1. Precomputation
    points = range(v)
    k_subs = list(itertools.combinations(points, k))
    n_blocks = len(k_subs)
    
    t_subs = list(itertools.combinations(points, t))
    n_tsubs = len(t_subs)
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    
    # block_to_tsubsets: list of lists of t-subset indices
    block_to_tsubsets = []
    # tsub_to_blocks: list of lists of block indices
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    
    for bi, ks in enumerate(k_subs):
        covered = []
        for ts in itertools.combinations(ks, t):
            idx = t_sub_index[ts]
            covered.append(idx)
            tsub_to_blocks[idx].append(bi)
        block_to_tsubsets.append(covered)
        
    # Convert tsub_to_blocks to sets for fast intersection
    tsub_to_blocks_sets = [set(lst) for lst in tsub_to_blocks]

    # 2. Greedy Initialization
    def greedy_init(seed, uncovered_mask=None):
        """
        Greedy construction.
        If uncovered_mask is None, covers all t-subsets.
        If uncovered_mask is a boolean array, covers only those t-subsets.
        """
        rng = random.Random(seed)
        if uncovered_mask is None:
            uncovered = np.ones(n_tsubs, dtype=bool)
        else:
            uncovered = uncovered_mask.copy()
            
        if not np.any(uncovered):
            return set()

        # Initialize gains
        gains = np.zeros(n_blocks, dtype=np.int32)
        # Only compute gain for blocks that cover at least one uncovered t-subset
        # Optimization: We can just compute for all, it's fast enough.
        for bi in range(n_blocks):
            gains[bi] = np.sum(uncovered[block_to_tsubsets[bi]])
            
        selected = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 20:
                break
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            bi = candidates[rng.randint(0, len(candidates)-1)]
            
            selected.add(bi)
            
            # Update uncovered and gains
            for ti in block_to_tsubsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    # Decrease gain for all blocks covering ti
                    for bj in tsub_to_blocks[ti]:
                        gains[bj] -= 1
            gains[bi] = -1
            
        return selected

    # Run multiple greedy restarts
    best_selected = None
    best_size = float('inf')
    
    for seed in range(30):
        if time.time() > deadline - 50:
            break
        sel = greedy_init(seed)
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = sel
            
    if best_selected is None:
        best_selected = greedy_init(0)

    # 3. Reduction and Optimization
    def reduce_and_optimize(selected):
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in selected:
            cover_count[block_to_tsubsets[bi]] += 1
            
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            
            # --- Deletion ---
            for bi in list(selected):
                if np.min(cover_count[block_to_tsubsets[bi]]) > 1:
                    cover_count[block_to_tsubsets[bi]] -= 1
                    selected.discard(bi)
                    changed = True
            
            if not selected:
                break
                
            # --- 2-for-1 Replacement ---
            sel_list = list(selected)
            n_sel = len(sel_list)
            if n_sel < 2:
                continue
                
            # Try random pairs
            for _ in range(200):
                if time.time() > deadline - 20:
                    break
                i, j = random.sample(range(n_sel), 2)
                b1, b2 = sel_list[i], sel_list[j]
                
                c1 = [ti for ti in block_to_tsubsets[b1] if cover_count[ti] == 1]
                c2 = [ti for ti in block_to_tsubsets[b2] if cover_count[ti] == 1]
                
                c_union = set(c1) | set(c2)
                if not c_union:
                    continue
                
                candidates = None
                for ti in c_union:
                    blocks_covering = tsub_to_blocks_sets[ti]
                    if candidates is None:
                        candidates = blocks_covering.copy()
                    else:
                        candidates &= blocks_covering
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                candidates.discard(b1)
                candidates.discard(b2)
                
                if not candidates:
                    continue
                
                b3 = next(iter(candidates))
                
                # Apply
                cover_count[block_to_tsubsets[b1]] -= 1
                cover_count[block_to_tsubsets[b2]] -= 1
                cover_count[block_to_tsubsets[b3]] += 1
                
                selected.discard(b1)
                selected.discard(b2)
                selected.add(b3)
                changed = True
                break # Restart loop
                
        return selected

    selected = reduce_and_optimize(best_selected)
    
    # 4. Stochastic Local Search with Kicks
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for bi in selected:
        cover_count[block_to_tsubsets[bi]] += 1
    
    n_uncovered = np.sum(cover_count == 0)
    current_cost = len(selected) + 100 * n_uncovered
    best_cost = current_cost
    best_selected = set(selected)
    
    temperature = 50.0
    cooling_rate = 0.995
    
    steps = 0
    while time.time() < deadline - 10:
        steps += 1
        
        # Kick Move (20% chance)
        if random.random() < 0.2:
            # Remove 2 random blocks
            if len(selected) >= 2:
                b1, b2 = random.sample(list(selected), 2)
                cover_count[block_to_tsubsets[b1]] -= 1
                cover_count[block_to_tsubsets[b2]] -= 1
                selected.discard(b1)
                selected.discard(b2)
                
                # Identify newly uncovered
                newly_uncovered = np.where(cover_count == 0)[0]
                
                if len(newly_uncovered) > 0:
                    # Create mask for greedy
                    mask = np.zeros(n_tsubs, dtype=bool)
                    mask[newly_uncovered] = True
                    
                    # Greedy fill
                    new_blocks = greedy_init(random.randint(0, 10000), mask)
                    
                    # Add new blocks
                    for bi in new_blocks:
                        cover_count[block_to_tsubsets[bi]] += 1
                        selected.add(bi)
                        
                    # Reduce
                    selected = reduce_and_optimize(selected)
                    
                    # Recalculate cost
                    n_uncovered = np.sum(cover_count == 0)
                    current_cost = len(selected) + 100 * n_uncovered
                else:
                    # No new uncovered, just reduced size
                    n_uncovered = 0
                    current_cost = len(selected)
                    
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected)
                    
        else:
            # Single Flip
            bi = random.randint(0, n_blocks - 1)
            
            if bi in selected:
                # Remove
                delta_blocks = -1
                newly_uncovered = 0
                for ti in block_to_tsubsets[bi]:
                    if cover_count[ti] == 1:
                        newly_uncovered += 1
                delta_uncovered = newly_uncovered
                delta_cost = delta_blocks + 100 * delta_uncovered
                
                if delta_cost < 0 or (delta_cost >= 0 and random.random() < np.exp(-delta_cost / temperature)):
                    cover_count[block_to_tsubsets[bi]] -= 1
                    selected.discard(bi)
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected)
            else:
                # Add
                delta_blocks = 1
                newly_covered = 0
                for ti in block_to_tsubsets[bi]:
                    if cover_count[ti] == 0:
                        newly_covered += 1
                delta_uncovered = -newly_covered
                delta_cost = delta_blocks + 100 * delta_uncovered
                
                if delta_cost < 0 or (delta_cost >= 0 and random.random() < np.exp(-delta_cost / temperature)):
                    cover_count[block_to_tsubsets[bi]] += 1
                    selected.add(bi)
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected)
                        
        temperature *= cooling_rate
        if temperature < 1.0:
            temperature = 1.0

    # Final Polish
    selected = reduce_and_optimize(best_selected)
    
    # Final Validation
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for bi in selected:
        cover_count[block_to_tsubsets[bi]] += 1
    if np.any(cover_count == 0):
        selected = best_selected
        
    return [list(k_subs[bi]) for bi in sorted(selected)]
