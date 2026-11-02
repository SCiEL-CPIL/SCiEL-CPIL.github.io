import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Precompute: for each subset index, the set of points it contains
    # and for each point, which subsets contain it
    subset_to_points = {}
    point_to_subsets = [[] for _ in range(v)]
    for idx, sub in enumerate(all_subsets):
        subset_to_points[idx] = set(sub)
        for p in sub:
            point_to_subsets[p].append(idx)
    
    best_blocks = None
    best_count = float('inf')
    
    def remove_redundant(blocks):
        """Remove blocks that do not cover any unique t-subset."""
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            counts = {}
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    counts[sub] = counts.get(sub, 0) + 1
            
            redundant_indices = []
            for i, block in enumerate(blocks):
                redundant = True
                for sub in itertools.combinations(block, t):
                    if counts[sub] == 1:
                        redundant = False
                        break
                if redundant:
                    redundant_indices.append(i)
            
            if redundant_indices:
                changed = True
                for idx in sorted(redundant_indices, reverse=True):
                    blocks.pop(idx)
                    
        return blocks
    
    def greedy_construct(seed):
        """Run greedy set cover with a given random seed."""
        rng = random.Random(seed)
        uncovered = [True] * n_subsets
        blocks = []
        
        # Precompute for each point, the list of uncovered subset indices containing it
        # We'll maintain coverage counts per subset
        coverage = np.zeros(n_subsets, dtype=np.int32)
        
        # For efficiency, precompute all k-subsets of v points
        # But that's C(19,8)=75582, which is manageable
        # Instead, we'll use a different approach: for each candidate block, compute its coverage gain
        
        while True:
            if time.time() > deadline:
                return None
            
            if not any(uncovered):
                break
            
            # Find the most uncovered subset to target
            uncovered_indices = [i for i in range(n_subsets) if uncovered[i]]
            if not uncovered_indices:
                break
            
            target_idx = rng.choice(uncovered_indices)
            target_sub = all_subsets[target_idx]
            
            # Build a block containing target_sub
            block_set = set(target_sub)
            
            while len(block_set) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                
                # Compute gain for each candidate
                best_gain = -1
                best_candidates = []
                for x in candidates:
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x covers a new subset
                    for rest in itertools.combinations(block_set, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        # Find index of new_sub in all_subsets
                        # This is slow; let's use a dict
                        if new_sub in subset_index:
                            idx = subset_index[new_sub]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_gain == 0:
                    x = rng.choice(candidates)
                else:
                    x = rng.choice(best_candidates)
                block_set.add(x)
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Update coverage
            for sub in itertools.combinations(block, t):
                idx = subset_index[sub]
                if uncovered[idx]:
                    coverage[idx] += 1
                    if coverage[idx] >= 1:
                        uncovered[idx] = False
        
        if any(uncovered):
            return None
        
        return blocks
    
    # Create a lookup from tuple to index for fast access
    subset_index = {sub: idx for idx, sub in enumerate(all_subsets)}
    
    # Run multiple greedy restarts with different seeds
    seed = 0
    while time.time() < deadline:
        blocks = greedy_construct(seed)
        seed += 1
        if blocks is None:
            continue
        
        blocks = remove_redundant(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Try local search: 1-exchange
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                n = len(blocks)
                # Coverage count
                cov_count = {}
                for b in blocks:
                    for sub in itertools.combinations(b, t):
                        cov_count[sub] = cov_count.get(sub, 0) + 1
                
                # Try removing each block
                for i in range(n):
                    if time.time() > deadline - 3:
                        break
                    # Check if block i is redundant
                    redundant = True
                    for sub in itertools.combinations(blocks[i], t):
                        if cov_count[sub] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        improved = True
                        break
                
                if not improved:
                    # Try 1-exchange: replace block i with a new block
                    # This is expensive; do a limited version
                    if time.time() < deadline - 10:
                        # Try to find a better block to replace each block
                        # For each block, see if we can find a replacement that covers more unique subsets
                        pass
    
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered_set = set(all_subsets)
        blocks = []
        while uncovered_set:
            target_subset = min(uncovered_set)
            block = list(target_subset)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                gains = []
                for x in candidates:
                    count = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered_set:
                            count += 1
                    gains.append(count)
                max_gain = max(gains)
                x = candidates[gains.index(max_gain)]
                block.append(x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered_set.discard(sub)
        blocks = remove_redundant(blocks)
        best_blocks = blocks

    return best_blocks
