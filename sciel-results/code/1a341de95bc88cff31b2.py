import itertools
import random
import time


def construct(v, k, t):
    all_points = list(range(v))
    total_5subsets = list(itertools.combinations(all_points, t))
    n_5subsets = len(total_5subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(total_5subsets)}
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered_count = [0] * n_5subsets  # how many blocks cover each 5-subset
        blocks = []
        
        # Precompute for each point, which 5-subsets contain it (for fast gain calc)
        # Actually, let's use a different strategy:
        # Pick uncovered 5-subsets, build blocks around them
        
        uncovered_indices = list(range(n_5subsets))
        rng.shuffle(uncovered_indices)
        
        while uncovered_indices:
            # Pick a random uncovered 5-subset to seed the block
            seed_idx = rng.choice(uncovered_indices)
            seed_subset = total_5subsets[seed_idx]
            
            block = list(seed_subset)
            rng.shuffle(block)
            block.sort()
            
            # Add k - t more points to maximize coverage
            remaining_points = [p for p in all_points if p not in set(block)]
            rng.shuffle(remaining_points)
            
            # Greedily add points that cover most uncovered 5-subsets
            for _ in range(k - t):
                best_point = None
                best_gain = -1
                for p in remaining_points:
                    # Count how many uncovered 5-subsets would be covered by adding p
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        new_5 = tuple(sorted(combo + (p,)))
                        idx = subset_to_idx.get(new_5)
                        if idx is not None and covered_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    if best_gain == 0:
                        break
                block.append(best_point)
                remaining_points.remove(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update covered counts
            for combo in itertools.combinations(block, t):
                idx = subset_to_idx[tuple(combo)]
                covered_count[idx] += 1
            
            # Remove newly covered 5-subsets from uncovered
            newly_covered = set()
            for combo in itertools.combinations(block, t):
                idx = subset_to_idx[tuple(combo)]
                if covered_count[idx] == 1:
                    newly_covered.add(idx)
            uncovered_indices = [i for i in uncovered_indices if i not in newly_covered]
        
        return blocks
    
    # Try multiple seeds and keep the best
    best_blocks = None
    best_len = float('inf')
    start_time = time.time()
    time_limit = 110  # Leave buffer
    
    seed = 42
    while time.time() - start_time < time_limit:
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1
    
    return best_blocks
