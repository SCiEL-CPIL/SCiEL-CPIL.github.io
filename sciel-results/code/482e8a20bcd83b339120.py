import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    # Precompute block_masks for pruning (fast redundancy check)
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
            
    # Precompute subset_to_blocks for fast greedy
    # subset_to_blocks[i] is a list of block indices covering subset i
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            subset_to_blocks[idx].append(b_idx)
            
    def greedy_cover(uncovered_set):
        """
        Greedily cover uncovered_set.
        Returns list of block indices added.
        """
        added = []
        uncovered = set(uncovered_set) # Copy
        
        while uncovered:
            # Pick a random uncovered subset
            target = random.choice(list(uncovered))
            
            # Pick a random block covering it
            candidates = subset_to_blocks[target]
            if not candidates:
                break # Should not happen
            
            # To improve quality, we can pick the candidate that covers the most 'uncovered' subsets
            # But that requires checking all candidates.
            # Let's try a "best of 5" heuristic.
            best_b = -1
            best_gain = -1
            
            # Sample up to 5 candidates
            sample_size = min(5, len(candidates))
            sampled = random.sample(candidates, sample_size)
            
            for b_idx in sampled:
                # Count how many uncovered subsets this block covers
                # We can use the mask
                gain = (block_masks[b_idx] & 0) # Wait, we don't have uncovered as mask here easily if we use set.
                # Let's stick to simple random pick for speed, or convert uncovered to mask?
                # Converting set to mask every step is expensive.
                
                # Alternative: Just pick random. The "best" greedy is deterministic and might get stuck in same basin.
                # Random greedy diversifies.
                pass
                
            # Let's just pick random for now, it's the "new" simple approach.
            # Actually, let's do a small optimization: Pick the one with highest "potential"
            # Potential = number of subsets in block that are in 'uncovered'
            # We can approximate by just picking random.
            
            b_idx = random.choice(candidates)
            added.append(b_idx)
            
            # Update uncovered
            # Get all subsets in this block
            # We can iterate the mask
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in uncovered:
                    uncovered.remove(idx)
                m ^= lsb
                
        return added

    def prune(blocks_indices):
        """
        Remove redundant blocks.
        """
        if not blocks_indices:
            return []
        
        counts = [0] * n_subsets
        
        for b_idx in blocks_indices:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        kept = []
        for b_idx in blocks_indices:
            mask = block_masks[b_idx]
            m = mask
            redundant = True
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
            
            if not redundant:
                kept.append(b_idx)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                    
        return kept

    rng = random.Random(42)
    
    # 1. Initial Greedy
    # Start with all subsets uncovered
    initial_uncovered = set(range(n_subsets))
    init_blocks = greedy_cover(initial_uncovered)
    init_blocks = prune(init_blocks)
    
    best_solution = init_blocks[:]
    best_len = len(best_solution)
    
    start_time = time.time()
    time_limit = 110
    
    # 2. Iterated Local Search
    while time.time() - start_time < time_limit:
        current_solution = best_solution[:]
        
        # Perturbation
        # Remove K random blocks
        K = rng.randint(2, 4)
        if len(current_solution) > K:
            to_remove = set(rng.sample(current_solution, K))
            current_solution = [b for b in current_solution if b not in to_remove]
            
            # Calculate uncovered
            covered_mask = 0
            for b_idx in current_solution:
                covered_mask |= block_masks[b_idx]
                
            # uncovered is indices where bit is 0 in covered_mask
            # We need to pass a set of indices to greedy_cover
            uncovered_set = set()
            # Iterate all bits? No, iterate all subsets and check mask.
            # n_subsets is 11628. This is fast.
            for i in range(n_subsets):
                if not (covered_mask & (1 << i)):
                    uncovered_set.add(i)
            
            # Re-greedy
            new_blocks = greedy_cover(uncovered_set)
            current_solution.extend(new_blocks)
            
            # Prune
            current_solution = prune(current_solution)
            
            if len(current_solution) < best_len:
                best_len = len(current_solution)
                best_solution = current_solution[:]
                
    return [list(all_blocks[i]) for i in best_solution]
