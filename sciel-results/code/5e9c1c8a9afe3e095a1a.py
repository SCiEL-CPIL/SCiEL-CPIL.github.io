"""Vectorized greedy covering with optimized gain calculation for C(v,k,t)."""
import time
import numpy as np
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subsets = list(combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For t=3, precompute for each pair (u, v) the list of indices of subsets {u, v, w}
    # pair_to_subsets[u][v] = list of subset indices
    # Or better: pair_to_thirds[u][v] = list of (w, subset_idx)
    # Since v=24, we can use a 2D array of lists or a dict.
    pair_to_thirds = {}
    for s in all_t_subsets:
        if t == 3:
            u, v, w = s
            idx = subset_to_idx[s]
            # Add to all three pairs
            pair_to_thirds.setdefault((u, v), []).append((w, idx))
            pair_to_thirds.setdefault((v, u), []).append((w, idx)) # Symmetric, but we can just use sorted keys
            pair_to_thirds.setdefault((u, w), []).append((v, idx))
            pair_to_thirds.setdefault((w, u), []).append((v, idx))
            pair_to_thirds.setdefault((v, w), []).append((u, idx))
            pair_to_thirds.setdefault((w, v), []).append((u, idx))
        else:
            # Generic fallback for t != 3 (though task is t=3)
            pass
            
    # To make lookup faster, use a dictionary with sorted tuple keys
    pair_map = {}
    for s in all_t_subsets:
        if t == 3:
            u, v, w = s
            idx = subset_to_idx[s]
            keys = [(min(u,v), max(u,v)), (min(u,w), max(u,w)), (min(v,w), max(v,w))]
            for key in keys:
                # The third point is the one not in the key
                if key == (min(u,v), max(u,v)):
                    third = w
                elif key == (min(u,w), max(u,w)):
                    third = v
                else:
                    third = u
                if key not in pair_map:
                    pair_map[key] = []
                pair_map[key].append((third, idx))

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(random_seed):
        nonlocal best_blocks, best_count
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random.RandomState()

        uncovered = np.ones(n_subsets, dtype=np.bool_)
        blocks = []

        while uncovered.any():
            if time.time() > deadline:
                break

            # Pick a random uncovered subset to seed
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[rng.randint(len(uncovered_indices))]
            seed_subset = all_t_subsets[seed_idx]
            
            block_set = set(seed_subset)
            block_points = list(seed_subset)

            # Greedily add points
            while len(block_points) < k:
                if time.time() > deadline:
                    break
                
                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block_set]
                
                # Calculate gain for each candidate
                # Gain(x) = number of uncovered subsets S such that x in S and S\{x} subset block_set
                # For t=3, S\{x} is a pair {u,v} in block_set.
                # So for each pair {u,v} in block_set, we look at pair_map[(u,v)]
                # and count how many (w, idx) have w == x and uncovered[idx] is True.
                
                # Optimization: Precompute for current block_set the list of (w, idx) for all pairs in block_set
                # This list might have duplicates if multiple pairs lead to same (w, idx)? No, each subset has unique pairs.
                # But different pairs in block_set can lead to the same subset? No, a subset {u,v,w} is determined by its pairs.
                # If block_set has {u,v,w}, pairs are (u,v)->w, (u,w)->v, (v,w)->u.
                # We are adding x. So we look for subsets {x, u, v} where {u,v} is in block_set.
                
                # Let's accumulate counts for each candidate x
                gain_counts = {x: 0 for x in candidates}
                
                # Iterate over pairs in current block
                # Number of pairs is C(len(block_points), 2)
                for i in range(len(block_points)):
                    for j in range(i+1, len(block_points)):
                        u = block_points[i]
                        v = block_points[j]
                        key = (min(u,v), max(u,v))
                        if key in pair_map:
                            for w, idx in pair_map[key]:
                                if w in gain_counts and uncovered[idx]:
                                    gain_counts[w] += 1
                
                # Find best point
                for x, gain in gain_counts.items():
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain == 0:
                    # If no gain, pick random candidate to complete block
                    if candidates:
                        best_point = candidates[rng.randint(len(candidates))]
                    else:
                        break

                block_set.add(best_point)
                block_points.append(best_point)

            # Finalize block
            if len(block_points) >= t:
                block_points.sort()
                # Check if block is already in blocks? 
                # The problem says blocks must be distinct.
                # We should avoid adding duplicate blocks.
                block_tuple = tuple(block_points)
                if block_tuple not in [tuple(b) for b in blocks]:
                    blocks.append(block_points)
                    # Update uncovered
                    for subset in combinations(block_points, t):
                        uncovered[subset_to_idx[subset]] = False

        if blocks and len(blocks) < best_count:
            # Verify
            covered = np.zeros(n_subsets, dtype=np.bool_)
            for block in blocks:
                for subset in combinations(block, t):
                    covered[subset_to_idx[subset]] = True
            if covered.all():
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
                
        return blocks

    # Run multiple restarts
    n_restarts = 100
    for i in range(n_restarts):
        if time.time() > deadline - 2.0:
            break
        greedy_cover(i)
        
    # Try one deterministic run if time permits
    if time.time() < deadline - 2.0 and best_blocks is None:
        greedy_cover(None)

    if best_blocks is None:
        # Fallback: simple greedy without randomization, just to return something valid
        # Re-run greedy_cover(0)
        greedy_cover(0)
        if best_blocks is None:
             # Emergency fallback: return empty or invalid? No, must be valid.
             # If greedy failed to cover, it's a bug. But let's assume it works.
             # If it didn't, we have a problem.
             pass

    return best_blocks
