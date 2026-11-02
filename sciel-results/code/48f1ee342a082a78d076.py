import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # Precompute all triplets (t-subsets)
    all_triplets = list(itertools.combinations(range(v), t))
    triplet_to_idx = {triplet: i for i, triplet in enumerate(all_triplets)}
    n_triplets = len(all_triplets)
    
    # For each point, precompute which triplets contain it
    point_triplets = defaultdict(list)
    for i, triplet in enumerate(all_triplets):
        for p in triplet:
            point_triplets[p].append(i)
    
    # Precompute all k-subsets and their coverage of t-subsets for vectorized greedy
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    
    # cover_matrix[i] contains the indices of t-subsets covered by all_k_subsets[i]
    # Shape: (n_k_subsets, c_k_t)
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triplet_to_idx[triple]
    
    block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}
    
    best_blocks = None
    best_size = float('inf')
    
    def vectorized_greedy_cover(seed):
        """Fast greedy using precomputed cover_matrix"""
        rng = random.Random(seed)
        uncovered = np.ones(n_triplets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Vectorized gain calculation
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks
    
    def point_greedy_run(seed_idx):
        """Point-by-point greedy starting from a specific uncovered triplet"""
        rng = random.Random(seed_idx)
        uncovered = [True] * n_triplets
        blocks = []
        
        while True:
            # Check if all covered
            if not any(uncovered):
                break
            
            # Pick a random uncovered triplet
            uncovered_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncovered_indices:
                break
            idx = rng.choice(uncovered_indices)
            seed_triplet = all_triplets[idx]
            
            # Build block containing seed_triplet
            block = list(seed_triplet)
            block_set = set(block)
            
            # Greedily add k - t points
            for _ in range(k - t):
                best_gain = -1
                best_point = -1
                best_gain_ties = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered triplets containing x and 2 points from block
                    gain = 0
                    for a, b in itertools.combinations(block, 2):
                        triplet = tuple(sorted((x, a, b)))
                        tidx = triplet_to_idx.get(triplet)
                        if tidx is not None and uncovered[tidx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_gain_ties = [x]
                    elif gain == best_gain and gain > 0:
                        best_gain_ties.append(x)
                
                if best_point != -1:
                    if len(best_gain_ties) > 1:
                        best_point = rng.choice(best_gain_ties)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tidx = triplet_to_idx.get(triplet)
                if tidx is not None:
                    uncovered[tidx] = False
        
        return blocks
    
    def reduce_blocks(blocks):
        """Remove redundant blocks using coverage counting"""
        if not blocks:
            return blocks
        
        block_sets = [set(b) for b in blocks]
        changed = True
        while changed:
            changed = False
            random.shuffle(block_sets)
            for i in range(len(block_sets)):
                if i >= len(block_sets):
                    break
                b = block_sets[i]
                is_redundant = True
                for triplet in itertools.combinations(b, t):
                    covered = False
                    for j in range(len(block_sets)):
                        if i == j:
                            continue
                        if set(triplet).issubset(block_sets[j]):
                            covered = True
                            break
                    if not covered:
                        is_redundant = False
                        break
                
                if is_redundant:
                    block_sets.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in block_sets]
    
    def vectorized_reduce(blocks):
        """Vectorized redundancy removal using coverage counts"""
        if not blocks:
            return blocks
        
        block_indices = [block_to_k_idx[b] for b in blocks]
        coverage = np.zeros(n_triplets, dtype=np.int32)
        
        for idx in block_indices:
            coverage[cover_matrix[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, idx in enumerate(block_indices):
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_block_indices = [idx for i, idx in enumerate(block_indices) if i not in remove_set]
                coverage = np.zeros(n_triplets, dtype=np.int32)
                for idx in new_block_indices:
                    coverage[cover_matrix[idx]] += 1
                block_indices = new_block_indices
                changed = True
        
        return [all_k_subsets[idx] for idx in block_indices]
    
    def local_search(blocks, max_iter=50):
        """Try to improve by removing blocks that become redundant after adding others"""
        blocks = [list(b) for b in blocks]
        block_sets = [set(b) for b in blocks]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(block_sets)):
                if time.time() > deadline:
                    break
                
                # Temporarily remove block i
                removed = block_sets.pop(i)
                
                # Find uncovered triplets
                uncovered = []
                for triplet in itertools.combinations(removed, t):
                    covered = False
                    for j in range(len(block_sets)):
                        if set(triplet).issubset(block_sets[j]):
                            covered = True
                            break
                    if not covered:
                        uncovered.append(triplet)
                
                if not uncovered:
                    # Block was redundant
                    improved = True
                    break
                
                # Try to add a new block that covers as many uncovered as possible
                best_new_block = None
                best_coverage = 0
                
                # Try random blocks containing some uncovered triplets
                for _ in range(20):
                    if not uncovered:
                        break
                    seed = random.choice(uncovered)
                    new_block = list(seed)
                    new_block_set = set(new_block)
                    
                    for _ in range(k - t):
                        best_gain = -1
                        best_point = -1
                        for x in range(v):
                            if x in new_block_set:
                                continue
                            gain = 0
                            for triplet in uncovered:
                                if x in triplet and set(triplet) & new_block_set:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = x
                        if best_point != -1:
                            new_block.append(best_point)
                            new_block_set.add(best_point)
                    
                    new_block.sort()
                    # Count how many uncovered triplets this covers
                    coverage = 0
                    for triplet in uncovered:
                        if set(triplet).issubset(new_block_set):
                            coverage += 1
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_new_block = new_block
                
                if best_new_block is not None and best_coverage >= len(uncovered):
                    # Add the new block
                    block_sets.append(set(best_new_block))
                    improved = True
                    break
                else:
                    # Restore
                    block_sets.insert(i, removed)
            
            if not improved:
                break
        
        return [list(b) for b in block_sets]
    
    # Main loop
    restart_count = 0
    while time.time() < deadline - 5:
        # Alternate between vectorized greedy and point greedy
        if restart_count % 2 == 0:
            blocks = vectorized_greedy_cover(restart_count)
            reduced_blocks = vectorized_reduce(blocks)
        else:
            seed_triplet = random.choice(all_triplets)
            seed_idx = triplet_to_idx[seed_triplet]
            blocks = point_greedy_run(seed_idx)
            reduced_blocks = reduce_blocks(blocks)
        
        # Local search
        improved_blocks = local_search(reduced_blocks)
        
        if len(improved_blocks) < best_size:
            best_size = len(improved_blocks)
            best_blocks = improved_blocks
        
        restart_count += 1
        
        # If we're close to optimal, spend more time on local search
        if best_size <= 78 and time.time() < deadline - 30:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                more_improved = local_search(best_blocks, max_iter=100)
                if len(more_improved) < best_size:
                    best_size = len(more_improved)
                    best_blocks = more_improved
    
    if best_blocks is None:
        return vectorized_greedy_cover(0)
    
    return best_blocks
