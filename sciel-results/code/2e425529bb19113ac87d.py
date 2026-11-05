import itertools
import time
import random
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute point-to-subsets mapping for pricing
point_to_subsets = [[] for _ in range(V)]
for sub_idx, sub in enumerate(subsets):
    for p in sub:
        point_to_subsets[p].append(sub_idx)

# Create lookup for block tuples to indices
block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    # 1. Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    # Greedy
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Use a heuristic to speed up: pick block with max new coverage
        # Randomize ties to avoid bias
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Column Generation Pricing
    # We use a pricing step that finds the "most efficient" 7-block
    # based on dual prices (inverse coverage counts) to guide perturbation
    
    def pricing_step(current_counts):
        """
        Find the 7-block that maximizes sum of dual prices.
        Dual price for subset s = 1/max(1, count[s]) if count[s] > 0, else large value.
        Uses a greedy heuristic: iteratively add the point that gives best marginal gain.
        """
        # Compute dual prices
        dual_prices = [0.0] * N_SUBSETS
        for i in range(N_SUBSETS):
            if current_counts[i] == 0:
                dual_prices[i] = 100.0  # High priority for uncovered
            elif current_counts[i] == 1:
                dual_prices[i] = 10.0   # Medium priority for barely covered
            else:
                dual_prices[i] = 1.0 / current_counts[i]
        
        # Greedy point selection
        selected_points = []
        covered_subsets = set()
        
        for _ in range(K):
            best_point = -1
            best_gain = -1
            for p in range(V):
                if p in selected_points:
                    continue
                # Calculate gain from adding point p
                gain = 0.0
                for sub_idx in point_to_subsets[p]:
                    if sub_idx not in covered_subsets:
                        gain += dual_prices[sub_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1:
                break
            
            selected_points.append(best_point)
            # Update covered subsets
            for sub_idx in point_to_subsets[best_point]:
                # Check if all other points in this subset are already selected
                sub = subsets[sub_idx]
                all_selected = True
                for p in sub:
                    if p != best_point and p not in selected_points:
                        all_selected = False
                        break
                if all_selected:
                    covered_subsets.add(sub_idx)
        
        if len(selected_points) < K:
            return None
        
        # Convert to block index
        block_tuple = tuple(sorted(selected_points))
        # Find index of this block
        block_idx = block_to_idx.get(block_tuple)
        return block_idx
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Phase 1: Prune
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap
        if len(current_blocks) == 0:
            break
            
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Find subsets that would become uncovered
        m_i = current_masks[i]
        uncovered_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            mm ^= lsb
            
        if not uncovered_bits:
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        if len(uncovered_bits) > 4:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Use pricing step to find a good block to add
                block_idx = pricing_step(counts)
                if block_idx is not None and block_idx not in current_blocks:
                    add_block(block_idx)
                    no_improve_count = 0
                    max_no_improve = min(200, max_no_improve * 2)
                else:
                    # Fallback to random
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = min(200, max_no_improve * 2)
                continue
        else:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Use pricing step
                    block_idx = pricing_step(counts)
                    if block_idx is not None and block_idx not in current_blocks:
                        add_block(block_idx)
                        no_improve_count = 0
                        max_no_improve = min(200, max_no_improve * 2)
                    else:
                        c = random.randint(0, N_BLOCKS - 1)
                        if c not in current_blocks:
                            add_block(c)
                            no_improve_count = 0
                            max_no_improve = min(200, max_no_improve * 2)
                    continue

    # Final cleanup: ensure we return the best found
    return [list(all_blocks[i]) for i in best_blocks]
