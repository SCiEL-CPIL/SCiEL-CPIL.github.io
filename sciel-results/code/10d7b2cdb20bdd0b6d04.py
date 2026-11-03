import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 118.0  # Safety margin within 120s limit

    points = list(range(v))
    all_t_subsets = set(itertools.combinations(points, t))
    total_t_subsets = len(all_t_subsets)
    
    # Precompute for each t-subset, the list of indices of points NOT in it
    # This helps in generating blocks containing a given t-subset
    points_set = set(points)
    
    def time_remaining():
        return deadline - time.time()

    # --- Greedy Construction ---
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        block_t_sets = []  # List of lists of t-subsets for each block

        while uncovered:
            if time_remaining() < 1.0:
                break
                
            # Optimization: If uncovered is small, we can be more aggressive
            # Pick a sample of uncovered t-subsets to evaluate
            if len(uncovered) < 100:
                samples = list(uncovered)
            else:
                # Sample up to 20 random uncovered t-subsets
                # random.sample on a set is not directly available in older versions, 
                # but list conversion is O(N). 
                # For 11628 items, list conversion is fast.
                # However, we can maintain a list of uncovered? 
                # Removing from list is O(N). 
                # Let's stick to set for membership, and sample by converting.
                # To speed up, we can just pick one random uncovered subset?
                # Standard greedy picks the block with max gain among ALL possible blocks.
                # But there are C(19,8) = 75582 blocks.
                # Evaluating all 75582 blocks every step is 75582 * 56 (t-subsets per block) ~ 4M ops.
                # 11628 / 56 avg gain ~ 200 steps. 200 * 4M = 800M ops. Too slow.
                
                # Heuristic Greedy:
                # Pick a random uncovered t-subset s.
                # Consider only blocks that contain s.
                # Number of such blocks: C(v-t, k-t) = C(14, 3) = 364.
                # This is much smaller.
                
                s = rng.choice(list(uncovered))
                s_set = set(s)
                remaining_points = [p for p in points if p not in s_set]
                
                # Generate all blocks containing s
                # k-t = 8-5 = 3
                candidates = []
                for combo in itertools.combinations(remaining_points, k - t):
                    block = tuple(sorted(list(s) + list(combo)))
                    # Calculate gain
                    gain = 0
                    for sub in itertools.combinations(block, t):
                        if sub in uncovered:
                            gain += 1
                    candidates.append((gain, block))
                
                # Sort by gain (descending)
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Pick the best one? Or top few?
                # Picking the absolute best among those covering 's' is a good heuristic.
                best_gain, best_block = candidates[0]
                
                if best_gain == 0:
                    # Should not happen if s is uncovered, but safety
                    break

                blocks.append(best_block)
                
                # Update uncovered
                new_tsets = []
                for sub in itertools.combinations(best_block, t):
                    if sub in uncovered:
                        uncovered.discard(sub)
                    new_tsets.append(sub)
                block_t_sets.append(new_tsets)

        # Complete the covering if time ran out during greedy
        while uncovered:
            if time_remaining() < 0.5:
                # Critical: if we are very close to timeout, we must ensure validity.
                # If we break here, we return an invalid covering -> Score 0.
                # We must finish.
                pass
            s = next(iter(uncovered))
            s_set = set(s)
            remaining_points = [p for p in points if p not in s_set]
            if len(remaining_points) < k - t:
                break
            # Pick a deterministic block to finish quickly
            combo = tuple(remaining_points[:k - t])
            block = tuple(sorted(list(s) + list(combo)))
            blocks.append(block)
            new_tsets = []
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    uncovered.discard(sub)
                new_tsets.append(sub)
            block_t_sets.append(new_tsets)

        return blocks, block_t_sets

    # --- Local Search / Improvement ---
    def local_search(blocks, block_t_sets, seed):
        rng = random.Random(seed)
        # Current size
        current_size = len(blocks)
        
        # Precompute coverage counts for each t-subset
        coverage_count = defaultdict(int)
        for tsets in block_t_sets:
            for ts in tsets:
                coverage_count[ts] += 1
        
        # Map: t-subset -> list of block indices covering it
        t_to_blocks = defaultdict(list)
        for i, tsets in enumerate(block_t_sets):
            for ts in tsets:
                t_to_blocks[ts].append(i)

        improved = True
        iterations = 0
        max_iterations = 50 # Limit local search time
        
        while improved and iterations < max_iterations and time_remaining() > 10.0:
            improved = False
            iterations += 1
            
            # Try to remove a redundant block
            # A block is redundant if all its t-subsets have coverage_count > 1
            removable_candidates = []
            for i in range(len(blocks)):
                is_redundant = True
                for ts in block_t_sets[i]:
                    if coverage_count[ts] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    removable_candidates.append(i)
            
            if not removable_candidates:
                break # No redundant blocks found, stop

            # Try removing one
            idx = rng.choice(removable_candidates)
            
            # Tentatively remove
            for ts in block_t_sets[idx]:
                coverage_count[ts] -= 1
                # If it becomes 0, we broke coverage? 
                # But we checked coverage_count > 1 before removal, so it should be >= 1 now.
            
            # Check if coverage is still valid (all counts >= 1)
            # Actually, by definition of redundancy check, all counts were >= 2, so after -1 they are >= 1.
            # So coverage is valid.
            
            # Commit removal
            blocks.pop(idx)
            block_t_sets.pop(idx)
            # Update t_to_blocks: remove idx from all its t-subsets' lists
            # This is O(k choose t) list updates.
            # Since we recompute or update carefully:
            for ts in block_t_sets[idx]:
                # Remove idx from t_to_blocks[ts]
                # List removal is O(N). N is small (num blocks covering a t-subset).
                try:
                    t_to_blocks[ts].remove(idx)
                except ValueError:
                    pass
            # Adjust indices in t_to_blocks for blocks after idx
            for ts in t_to_blocks:
                lst = t_to_blocks[ts]
                for j in range(len(lst)):
                    if lst[j] > idx:
                        lst[j] -= 1
            
            current_size -= 1
            improved = True

        return blocks, block_t_sets, current_size

    # --- Main Loop ---
    best_blocks = None
    best_t_sets = None
    best_size = float('inf')

    # Run multiple seeds
    # Estimate time:
    # Greedy: ~100-200 steps. Each step: 364 candidates * 56 tsets = ~20k ops.
    # 200 * 20k = 4M ops. Very fast in Python? Maybe 1-2 seconds.
    # Local Search: Very fast.
    # So we can run many seeds.
    
    num_seeds = 10
    for seed in range(num_seeds):
        if time_remaining() < 5.0:
            break
            
        blocks, t_sets = greedy_cover(seed)
        
        # Ensure validity (should be valid if loop completed)
        # If greedy broke early due to time, the "Complete" loop should have finished it.
        
        # Local Search
        if time_remaining() > 5.0:
            blocks, t_sets, size = local_search(blocks, t_sets, seed)
        else:
            size = len(blocks)
            
        if size < best_size:
            best_size = size
            best_blocks = blocks
            best_t_sets = t_sets

    # If we didn't find anything (shouldn't happen), fallback
    if best_blocks is None:
        blocks, t_sets = greedy_cover(0)
        # Force completion if invalid
        # (Assuming greedy_cover handles completion)
        best_blocks = blocks

    return [list(b) for b in best_blocks]
