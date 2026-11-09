import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a hybrid of randomized greedy 
    strategies and local search (redundancy removal).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    # Pre-compute all t-subsets for efficiency
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    def greedy_construct(seed, strategy):
        """
        Runs a single greedy construction pass.
        strategy: 0 = Max Gain, 1 = Random Top-3, 2 = Frequency Aware
        """
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        point_counts = [0] * v
        
        while uncovered:
            # Pick a seed point from the first uncovered subset
            first_uncovered = min(uncovered)
            seed_point = first_uncovered[0]
            block = [seed_point]
            point_counts[seed_point] += 1
            
            while len(block) < k:
                # Calculate gain for each candidate point
                gains = []
                candidates = [x for x in range(v) if x not in block]
                
                for x in candidates:
                    # Gain: number of uncovered t-subsets covered by adding x
                    # A t-subset is covered if it contains x and (t-1) elements from block
                    # We need to check which (t-1) subsets of block combined with x are in uncovered
                    gain = 0
                    # Optimization: Only check (t-1) combinations of block
                    for rest in itertools.combinations(block, t - 1):
                        # Create the t-subset
                        sub = tuple(sorted(list(rest) + [x]))
                        if sub in uncovered:
                            gain += 1
                    gains.append((gain, x))
                
                if not gains:
                    break
                    
                # Sort by gain descending
                gains.sort(key=lambda item: item[0], reverse=True)
                max_gain = gains[0][0]
                
                if max_gain == 0:
                    # No point increases coverage, pick random to finish block
                    x = rng.choice(candidates)
                else:
                    if strategy == 0:
                        # Deterministic Max Gain (tie-break by index)
                        best_candidates = [x for g, x in gains if g == max_gain]
                        x = min(best_candidates)
                    elif strategy == 1:
                        # Random from top 3
                        top_k = min(3, len(gains))
                        top_candidates = [x for g, x in gains[:top_k] if g == max_gain]
                        # Ensure we only pick from those with max gain? Or just top K?
                        # Picking from top K allows escaping local optima
                        x = rng.choice(top_candidates)
                    elif strategy == 2:
                        # Frequency Aware: Prefer points with lower count if gains are close
                        # Use gain as primary, count as secondary
                        # Score = gain * 1000 - point_counts[x]
                        scored = [(g * 1000 - point_counts[x], x) for g, x in gains]
                        scored.sort(reverse=True)
                        x = scored[0][1]
                    else:
                        x = gains[0][1]
                        
                block.append(x)
                point_counts[x] += 1
                
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    def remove_redundant_blocks(blocks, v, k, t):
        """
        Removes blocks that do not cover any unique t-subset.
        """
        blocks = list(blocks)
        # Count how many blocks cover each t-subset
        subset_counts = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                subset_counts[sub] = subset_counts.get(sub, 0) + 1
                
        new_blocks = []
        for b in blocks:
            # Check if b covers any subset that is covered only by b
            is_unique = False
            for sub in itertools.combinations(b, t):
                if subset_counts.get(sub, 0) == 1:
                    is_unique = True
                    break
            if not is_unique:
                # Remove b
                for sub in itertools.combinations(b, t):
                    subset_counts[sub] -= 1
            else:
                new_blocks.append(b)
                
        # Repeat if changes were made
        if len(new_blocks) < len(blocks):
            return remove_redundant_blocks(new_blocks, v, k, t)
        return new_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple strategies and seeds
    # Strategy 0: Deterministic
    # Strategy 1: Randomized Top-3
    # Strategy 2: Frequency Aware
    
    # Run a few deterministic or quasi-deterministic passes
    # 1. Standard Greedy (Seed 0, Strategy 0)
    b = greedy_construct(0, 0)
    b = remove_redundant_blocks(b, v, k, t)
    if len(b) < best_count:
        best_count = len(b)
        best_blocks = b
        
    # 2. Frequency Aware (Seed 0, Strategy 2)
    if time.time() < deadline:
        b = greedy_construct(0, 2)
        b = remove_redundant_blocks(b, v, k, t)
        if len(b) < best_count:
            best_count = len(b)
            best_blocks = b

    # 3. Randomized passes
    # Run several randomized instances
    num_random_trials = 20
    for i in range(num_random_trials):
        if time.time() > deadline - 5.0:
            break
        b = greedy_construct(i + 100, 1)
        b = remove_redundant_blocks(b, v, k, t)
        if len(b) < best_count:
            best_count = len(b)
            best_blocks = b
            # If we get very close to known best, we might want to stop or continue
            # 151 is the known best. If we hit 155, we are close.
            
    # Final check: Ensure validity
    if best_blocks is None:
        # Fallback: Return a trivial (though large) covering
        # Just return the first greedy result
        best_blocks = greedy_construct(0, 0)
        
    # Convert tuples to lists for output
    return [list(b) for b in best_blocks]

# Test block for local verification (optional, commented out for submission)
# if __name__ == "__main__":
#     blocks = construct(19, 7, 4)
#     print(f"Blocks: {len(blocks)}")
#     # Verify
#     covered = set()
#     for b in blocks:
#         for sub in itertools.combinations(b, 4):
#             covered.add(sub)
#     all_subs = set(itertools.combinations(range(19), 4))
#     print(f"Valid: {covered == all_subs}")
