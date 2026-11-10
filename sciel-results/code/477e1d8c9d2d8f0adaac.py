import itertools
import random
import time
import sys

# Global precomputation for efficiency, though v,k,t are passed.
# Since v=19, k=7, t=4 is fixed in the task description context ("This task's parameters"),
# but the function signature takes them. We should precompute inside or cache.
# Given 120s limit, precomputing C(19,7) masks/counts is fast.

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    # Precompute for each possible block (k-subset), the list of t-subset indices it covers
    # Number of blocks = C(19, 7) = 50388
    # Each block covers C(7, 4) = 35 t-subsets.
    # Total work: 50388 * 35 ~ 1.7M operations. Fast.
    
    all_blocks = list(itertools.combinations(points, k))
    block_to_t_indices = {}
    for block in all_blocks:
        block_to_t_indices[block] = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
        
    # Helper: Greedy Construction
    def greedy_construct():
        uncovered = [0] * n_t # 0 means uncovered, 1 means covered? 
        # Actually, we want to track count of uncovered? No, just boolean is enough for greedy selection.
        # But for repair, we need to know *which* are uncovered.
        # Let's use a set of uncovered indices.
        uncovered_set = set(range(n_t))
        blocks = []
        
        while uncovered_set:
            best_block = None
            best_count = -1
            
            # Optimization: Instead of checking all 50k blocks, we can pick a random uncovered t-subset
            # and only check blocks containing it?
            # Or just iterate all blocks. 50k iterations * 35 checks = 1.7M ops per greedy step.
            # If we need 150 steps, that's 255M ops. Might be too slow in Python.
            
            # Better Greedy:
            # Pick a random uncovered t-subset.
            # Only consider blocks that contain this t-subset.
            # How many blocks contain a specific t-subset?
            # We need to choose k-t points from v-t points.
            # C(19-4, 7-4) = C(15, 3) = 455 blocks.
            # This is much faster!
            
            anchor_idx = random.choice(list(uncovered_set))
            anchor_t = all_t_subsets[anchor_idx]
            anchor_set = set(anchor_t)
            
            # Find all blocks containing anchor_t
            # We can precompute a map: t_subset -> list of blocks containing it?
            # C(19,4) = 3876 t-subsets.
            # Each t-subset is in C(19-4, 7-4) = 455 blocks.
            # Total entries: 3876 * 455 ~ 1.7M.
            # This is feasible.
            
            # Let's build this map once outside the loop?
            # But it depends on v,k,t.
            
            # Let's do it inside construct but outside the greedy loop.
            # Wait, I can just generate the 455 blocks on the fly.
            # remaining_points = [p for p in points if p not in anchor_set]
            # for combo in itertools.combinations(remaining_points, k-t):
            #    block = tuple(sorted(anchor_t + combo))
            #    ...
            
            # This is O(455 * 35) per step. Very fast.
            
            candidates = []
            remaining_points = [p for p in points if p not in anchor_set]
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor_t + combo))
                # Count how many uncovered subsets this block covers
                # block_to_t_indices[block] gives the indices
                count = 0
                for idx in block_to_t_indices[block]:
                    if idx in uncovered_set:
                        count += 1
                candidates.append((count, block))
            
            # Pick best
            max_cov = max(c[0] for c in candidates)
            best_candidates = [c[1] for c in candidates if c[0] == max_cov]
            chosen_block = random.choice(best_candidates)
            
            blocks.append(chosen_block)
            
            # Update uncovered_set
            for idx in block_to_t_indices[chosen_block]:
                uncovered_set.discard(idx)
                
        return blocks

    # Helper: Minimize (Remove Redundant)
    def minimize(blocks):
        if not blocks:
            return []
        
        # Initialize counts
        counts = [0] * n_t
        for block in blocks:
            for idx in block_to_t_indices[block]:
                counts[idx] += 1
        
        changed = True
        while changed:
            changed = False
            # Randomize order to break ties/symmetries
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            new_blocks = []
            for i in indices:
                block = blocks[i]
                # Check if redundant
                redundant = True
                for idx in block_to_t_indices[block]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove
                    changed = True
                    for idx in block_to_t_indices[block]:
                        counts[idx] -= 1
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
            
        return blocks

    # Main ILS Loop
    best_blocks = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 110
    
    # Initial Solution
    curr_blocks = greedy_construct()
    curr_blocks = minimize(curr_blocks)
    best_blocks = curr_blocks
    best_size = len(curr_blocks)
    
    while time.time() - start_time < time_limit:
        # Perturb
        perturbed_blocks = list(curr_blocks)
        # Remove some random blocks
        num_remove = random.randint(1, 5)
        if len(perturbed_blocks) > num_remove:
            to_remove = random.sample(range(len(perturbed_blocks)), num_remove)
            # Remove from end to start to keep indices valid, or use set
            remove_set = set(to_remove)
            perturbed_blocks = [b for i, b in enumerate(perturbed_blocks) if i not in remove_set]
        
        # Repair
        # Calculate uncovered
        counts = [0] * n_t
        for block in perturbed_blocks:
            for idx in block_to_t_indices[block]:
                counts[idx] += 1
        
        uncovered_set = set(i for i in range(n_t) if counts[i] == 0)
        
        # Greedy Repair
        while uncovered_set:
            anchor_idx = random.choice(list(uncovered_set))
            anchor_t = all_t_subsets[anchor_idx]
            anchor_set = set(anchor_t)
            
            remaining_points = [p for p in points if p not in anchor_set]
            best_count = -1
            best_block = None
            
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor_t + combo))
                count = 0
                for idx in block_to_t_indices[block]:
                    if idx in uncovered_set:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block:
                perturbed_blocks.append(best_block)
                for idx in block_to_t_indices[best_block]:
                    uncovered_set.discard(idx)
                    counts[idx] += 1 # Update counts for consistency if needed, though we rebuild in minimize
        
        # Minimize
        perturbed_blocks = minimize(perturbed_blocks)
        
        if len(perturbed_blocks) < best_size:
            best_size = len(perturbed_blocks)
            best_blocks = perturbed_blocks
            curr_blocks = perturbed_blocks
            
    return [list(b) for b in best_blocks]
