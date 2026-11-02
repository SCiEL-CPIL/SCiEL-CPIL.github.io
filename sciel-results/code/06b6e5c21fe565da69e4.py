import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Pre-compute all t-subsets as a set for O(1) lookup
    all_t_subsets = set(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for array-based operations
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_len = float('inf')
    
    def is_valid(blocks):
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        return len(covered) == n_t_subsets
    
    def compute_gains(block, uncovered_mask):
        """Compute how many uncovered t-subsets are in this block."""
        gain = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            if not uncovered_mask[idx]:
                gain += 1
        return gain
    
    def greedy_cover(rng, deadline):
        """Fast greedy construction using random pivot selection."""
        uncovered_mask = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                break
                
            # Pick a random uncovered t-subset as pivot
            uncovered_indices = np.where(uncovered_mask)[0]
            pivot_idx = rng.choice(uncovered_indices)
            pivot_ts = list(t_subset_to_idx.keys())[pivot_idx]
            # Better: use a reverse map
            # Actually, let's just iterate to find it or precompute
            # For efficiency, let's use a list
            pivot_points = set(pivot_ts)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            # Find best block containing pivot that maximizes coverage
            best_block = None
            best_gain = -1
            
            # Sample a subset of candidate blocks if too many
            n_candidates = len(list(itertools.combinations(other_points, k - t)))
            if n_candidates > 2000:
                # Random sample of candidates
                candidates = []
                for _ in range(2000):
                    combo = rng.sample(other_points, k - t)
                    candidates.append(combo)
            else:
                candidates = list(itertools.combinations(other_points, k - t))
            
            for combo in candidates:
                block = tuple(sorted(pivot_points | set(combo)))
                gain = 0
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if uncovered_mask[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                # Update uncovered mask
                for ts in itertools.combinations(best_block, t):
                    idx = t_subset_to_idx[ts]
                    uncovered_mask[idx] = False
                    
        return blocks
    
    def reduce_covering(blocks, deadline):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
            
        blocks = list(blocks)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                # Check if block i is redundant
                remaining = blocks[:i] + blocks[i+1:]
                covered = set()
                for b in remaining:
                    covered.update(itertools.combinations(b, t))
                if len(covered) == n_t_subsets:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    def local_search(blocks, deadline):
        """Randomized local search to reduce block count."""
        blocks = list(blocks)
        best_len = len(blocks)
        best_blocks = list(blocks)
        
        # Build coverage count for each t-subset
        coverage_count = np.zeros(n_t_subsets, dtype=int)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                coverage_count[t_subset_to_idx[ts]] += 1
        
        t = 100.0  # temperature for simulated annealing
        cooling = 0.999
        
        while time.time() < deadline and t > 0.1:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
            
            # Try to remove a block
            if len(blocks) > 1:
                i = random.randrange(len(blocks))
                # Check if block i is redundant
                if all(coverage_count[t_subset_to_idx[ts]] > 1 for ts in itertools.combinations(blocks[i], t)):
                    # Safe to remove
                    for ts in itertools.combinations(blocks[i], t):
                        coverage_count[t_subset_to_idx[ts]] -= 1
                    blocks.pop(i)
                    continue
            
            # Try swap: remove block i, add block j
            if len(blocks) > 1:
                i = random.randrange(len(blocks))
                # Generate a random new block
                new_block = random.sample(range(v), k)
                new_block = sorted(new_block)
                
                # Check if this swap would be valid or improve
                # Remove block i temporarily
                for ts in itertools.combinations(blocks[i], t):
                    coverage_count[t_subset_to_idx[ts]] -= 1
                
                # Check if all t-subsets still covered
                valid = True
                for ts in itertools.combinations(blocks[i], t):
                    if coverage_count[t_subset_to_idx[ts]] == 0:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    for ts in itertools.combinations(blocks[i], t):
                        coverage_count[t_subset_to_idx[ts]] += 1
                    # Accept with probability based on temperature
                    if random.random() < np.exp(-1.0 / t):
                        # Keep the invalid state? No, revert properly
                        pass
                    continue
                
                # Add new block
                for ts in itertools.combinations(new_block, t):
                    coverage_count[t_subset_to_idx[ts]] += 1
                
                # Check if we can remove another block now
                # For simplicity, just accept if valid
                blocks[i] = new_block
                
                # Check if any block is now redundant
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if all(coverage_count[t_subset_to_idx[ts]] > 1 for ts in itertools.combinations(blocks[j], t)):
                        for ts in itertools.combinations(blocks[j], t):
                            coverage_count[t_subset_to_idx[ts]] -= 1
                        blocks.pop(j)
                        break
                
                t *= cooling
        
        return best_blocks
    
    # Run multiple iterations with different random seeds
    rng = random.Random(42)
    iterations = 0
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        blocks = greedy_cover(rng, deadline - 3)
        
        if not blocks:
            continue
            
        # Reduce
        blocks = reduce_covering(blocks, deadline - 2)
        
        # Local search
        if time.time() < deadline - 1:
            blocks = local_search(blocks, deadline - 1)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        iterations += 1
        rng.seed(42 + iterations)
        
        # If we have a very good result, we can still keep trying
        # but with less time per iteration
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered_mask = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        while np.any(uncovered_mask):
            # Find first uncovered
            pivot_idx = np.where(uncovered_mask)[0][0]
            # Get the t-subset
            # Need reverse mapping
            # Let's just iterate
            for ts in all_t_subsets:
                if uncovered_mask[t_subset_to_idx[ts]]:
                    pivot_ts = ts
                    break
            
            pivot_points = set(pivot_ts)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = 0
                for ts in itertools.combinations(block, t):
                    if uncovered_mask[t_subset_to_idx[ts]]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered_mask[t_subset_to_idx[ts]] = False
        
        best_blocks = reduce_covering(blocks, deadline)

    return best_blocks
