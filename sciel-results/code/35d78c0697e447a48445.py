"""Simulated annealing approach for covering design C(24,7,3)."""
import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using simulated annealing.
    
    Strategy:
    1. Start with a greedy covering.
    2. Use simulated annealing with fast incremental updates to reduce block count.
    3. Aggressively remove redundant blocks throughout the search.
    
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    random.seed(42)
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each block (as a frozenset), the list of t-subset indices it covers
    # We'll use a cache for this
    block_cover_cache = {}
    
    def get_block_coverage(block_tuple):
        """Get the list of t-subset indices covered by a block (as a sorted tuple)."""
        if block_tuple not in block_cover_cache:
            coverage = []
            for ts in itertools.combinations(block_tuple, t):
                coverage.append(t_subset_to_idx[ts])
            block_cover_cache[block_tuple] = coverage
        return block_cover_cache[block_tuple]
    
    # Greedy covering algorithm
    def greedy_covering():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            target = min(uncovered)
            block = set(target)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(sorted(block), t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.add(best_x)
            block_list = sorted(block)
            blocks.append(block_list)
            for ts in itertools.combinations(block_list, t):
                uncovered.discard(ts)
        return blocks
    
    # Initialize with greedy
    blocks = greedy_covering()
    n_blocks = len(blocks)
    
    # Build coverage counts
    counts = [0] * n_t_subsets
    for block in blocks:
        for idx in get_block_coverage(tuple(sorted(block))):
            counts[idx] += 1
    
    current_uncovered = sum(1 for c in counts if c == 0)
    
    # SA parameters
    start_time = time.time()
    max_time = 115.0
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    best_blocks = [b[:] for b in blocks]
    best_uncovered = current_uncovered
    best_n_blocks = n_blocks
    
    # Helper: compute delta for replacing point old_p with new_p in a block
    def compute_delta_replace(block_list, old_p, new_p, counts):
        """Compute change in uncovered count when replacing old_p with new_p in block."""
        old_block_tuple = tuple(sorted(block_list))
        new_block = block_list[:]
        new_block.remove(old_p)
        new_block.append(new_p)
        new_block.sort()
        new_block_tuple = tuple(new_block)
        
        old_cov = set(get_block_coverage(old_block_tuple))
        new_cov = set(get_block_coverage(new_block_tuple))
        
        # t-subsets that lose coverage
        lost = old_cov - new_cov
        gained = new_cov - old_cov
        
        delta = 0
        for idx in lost:
            if counts[idx] == 1:
                delta += 1
        for idx in gained:
            if counts[idx] == 0:
                delta -= 1
        return delta, old_cov, new_cov
    
    # Helper: apply a replacement
    def apply_replace(bi, old_p, new_p, old_cov, new_cov, counts):
        """Apply point replacement and update counts."""
        block = blocks[bi]
        block.remove(old_p)
        block.append(new_p)
        block.sort()
        
        for idx in old_cov:
            if idx not in new_cov:
                counts[idx] -= 1
        for idx in new_cov:
            if idx not in old_cov:
                counts[idx] += 1
    
    # Helper: check if block is redundant
    def is_redundant(bi, counts):
        block = blocks[bi]
        for idx in get_block_coverage(tuple(sorted(block))):
            if counts[idx] <= 1:
                return False
        return True
    
    # Main SA loop
    step = 0
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        step += 1
        
        # Try to remove redundant blocks more frequently
        if step % 10 == 0 and n_blocks > 1:
            # Try to find and remove a redundant block
            for bi in range(n_blocks - 1, -1, -1):
                if is_redundant(bi, counts):
                    block = blocks[bi]
                    for idx in get_block_coverage(tuple(sorted(block))):
                        counts[idx] -= 1
                    blocks.pop(bi)
                    n_blocks -= 1
                    current_uncovered = sum(1 for c in counts if c == 0)
                    if current_uncovered == 0 and n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_uncovered = current_uncovered
                        best_blocks = [b[:] for b in blocks]
                    break
        
        if n_blocks < 1:
            break
        
        # Choose move type
        r = random.random()
        
        if r < 0.6:
            # Move 1: Replace a point in a random block
            bi = random.randint(0, n_blocks - 1)
            block = blocks[bi]
            old_p = random.choice(block)
            
            # Choose new point not in block
            available = [x for x in range(v) if x not in block]
            if not available:
                continue
            new_p = random.choice(available)
            
            delta, old_cov, new_cov = compute_delta_replace(block, old_p, new_p, counts)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_replace(bi, old_p, new_p, old_cov, new_cov, counts)
                current_uncovered = sum(1 for c in counts if c == 0)
                if current_uncovered == 0 and n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
                elif current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
        
        elif r < 0.85 and n_blocks >= 2:
            # Move 2: Swap points between two blocks
            bi1 = random.randint(0, n_blocks - 1)
            bi2 = random.randint(0, n_blocks - 1)
            if bi1 == bi2:
                continue
            
            block1 = blocks[bi1]
            block2 = blocks[bi2]
            
            p1 = random.choice(block1)
            p2 = random.choice(block2)
            
            if p1 in block2 or p2 in block1:
                continue
            
            # Compute delta for both blocks
            delta1, old_cov1, new_cov1 = compute_delta_replace(block1, p1, p2, counts)
            delta2, old_cov2, new_cov2 = compute_delta_replace(block2, p2, p1, counts)
            
            # Note: delta1 and delta2 are computed independently, but we need to account for
            # the fact that changes in one block affect counts used in the other
            # For simplicity, compute combined delta
            total_delta = delta1 + delta2
            
            # More accurate: compute the actual change
            # We'll just use the sum as an approximation
            if total_delta <= 0 or random.random() < math.exp(-total_delta / T):
                # Apply both changes
                apply_replace(bi1, p1, p2, old_cov1, new_cov1, counts)
                apply_replace(bi2, p2, p1, old_cov2, new_cov2, counts)
                current_uncovered = sum(1 for c in counts if c == 0)
                if current_uncovered == 0 and n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
                elif current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
        
        else:
            # Move 3: Try to remove a redundant block
            if n_blocks > 1:
                bi = random.randint(0, n_blocks - 1)
                if is_redundant(bi, counts):
                    block = blocks[bi]
                    for idx in get_block_coverage(tuple(sorted(block))):
                        counts[idx] -= 1
                    blocks.pop(bi)
                    n_blocks -= 1
                    current_uncovered = sum(1 for c in counts if c == 0)
                    if current_uncovered == 0 and n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_uncovered = current_uncovered
                        best_blocks = [b[:] for b in blocks]
        
        T *= cooling_rate
    
    # Use best solution found
    blocks = best_blocks
    
    # Final cleanup: remove all redundant blocks
    changed = True
    while changed:
        changed = False
        counts = [0] * n_t_subsets
        for block in blocks:
            for idx in get_block_coverage(tuple(sorted(block))):
                counts[idx] += 1
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            redundant = True
            for idx in get_block_coverage(tuple(sorted(block))):
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in get_block_coverage(tuple(sorted(block))):
                    counts[idx] -= 1
                blocks.pop(i)
                changed = True
                break
    
    # Ensure sorted
    for i in range(len(blocks)):
        blocks[i].sort()
    
    # Verify coverage
    final_counts = [0] * n_t_subsets
    for block in blocks:
        for idx in get_block_coverage(tuple(sorted(block))):
            final_counts[idx] += 1
    if any(c == 0 for c in final_counts):
        # Fallback to greedy
        blocks = greedy_covering()
    
    return blocks
