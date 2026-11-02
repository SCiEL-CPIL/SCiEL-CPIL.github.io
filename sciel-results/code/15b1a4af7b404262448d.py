"""
Covering design C(24,7,3) solver.

Strategy:
1. Fast greedy construction using coverage array for speed.
2. Local search: iteratively remove redundant blocks using coverage counts.
3. Deadline guard to ensure we finish within 120s.
"""

import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    # Map from t-subset tuple to index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # is_uncovered[i] = True if t-subset i is not yet covered
    is_uncovered = [True] * n_t_subsets
    uncovered_count = n_t_subsets
    
    blocks = []
    
    # Fast Greedy
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find an uncovered t-subset to target
        target_idx = -1
        for i in range(n_t_subsets):
            if is_uncovered[i]:
                target_idx = i
                break
                
        if target_idx == -1:
            break
            
        target_subset = all_t_subsets[target_idx]
        block = list(target_subset)
        block_set = set(block)
        
        # Greedily add k - t points
        for _ in range(k - t):
            best_point = -1
            best_gain = -1
            
            # Calculate gain for each candidate point x
            # Gain(x) = number of pairs (u,v) in block such that {x,u,v} is uncovered
            block_pairs = list(itertools.combinations(block, 2))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for u, v_pt in block_pairs:
                    # Check if {x, u, v_pt} is uncovered
                    a, b, c = sorted((x, u, v_pt))
                    idx = t_subset_to_idx.get((a, b, c), -1)
                    if idx != -1 and is_uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block.append(best_point)
                block_set.add(best_point)
            else:
                # Fallback: pick random point not in block
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_point = random.choice(remaining)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            if is_uncovered[idx]:
                is_uncovered[idx] = False
                uncovered_count -= 1

    # Local Search: Remove redundant blocks
    def get_block_subsets(block):
        return [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
    
    # Build coverage counts
    coverage = [0] * n_t_subsets
    for block in blocks:
        for idx in get_block_subsets(block):
            coverage[idx] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_blocks = []
        for block in blocks:
            # Check if block is redundant
            redundant = True
            for idx in get_block_subsets(block):
                if coverage[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for idx in get_block_subsets(block):
                    coverage[idx] -= 1
                improved = True
            else:
                new_blocks.append(block)
        
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
        else:
            break

    return blocks
