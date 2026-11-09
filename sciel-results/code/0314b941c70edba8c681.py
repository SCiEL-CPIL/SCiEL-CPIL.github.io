"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    uncovered = set(itertools.combinations(range(v), t))
    total_t_subsets = len(uncovered)
    
    blocks = []
    random.seed(42)
    
    while uncovered:
        # Check deadline
        if time.time() > deadline - 5.0:
            break
        
        # Pick a seed t-subset from uncovered (randomized for diversity)
        seed_subset = random.choice(list(uncovered))
        block = list(seed_subset)
        
        # Build the block by adding points with highest marginal gain
        while len(block) < k:
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Compute gain for each candidate point
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            gains = []
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_t = tuple(sorted(rest + (x,)))
                    if new_t in uncovered:
                        gain += 1
                gains.append(gain)
            
            # Find max gain and select with probability proportional to gain
            max_gain = max(gains)
            if max_gain == 0:
                # No point improves coverage, pick randomly
                block.append(random.choice(candidates))
            else:
                # Weighted random selection: probability proportional to gain
                # Use softmax-like weighting to balance exploration and exploitation
                weights = [g ** 2 + 0.1 for g in gains]  # Square to emphasize high-gain points
                total_weight = sum(weights)
                r = random.random() * total_weight
                cumulative = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        block.append(candidates[i])
                        break
                else:
                    block.append(candidates[-1])
        
        # Sort and add block
        block.sort()
        if len(block) == k:
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
    
    # Final polish: try to merge or replace blocks to reduce count
    # This is a simple local search: try removing each block and see if coverage is still complete
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try removing each block
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 1.0:
                    break
                # Temporarily remove block i
                removed = blocks.pop(i)
                # Check if all t-subsets are still covered
                covered = set()
                for b in blocks:
                    covered.update(itertools.combinations(b, t))
                if all(ts in covered for ts in itertools.combinations(range(v), t)):
                    improved = True
                    break
                else:
                    blocks.insert(i, removed)
    
    # Ensure we have a valid covering; if not, fall back to deterministic greedy
    if uncovered:
        # Complete the covering with deterministic greedy
        while uncovered and time.time() < deadline:
            seed_subset = min(uncovered)
            block = list(seed_subset)
            while len(block) < k and time.time() < deadline:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                best_x = None
                best_gain = -1
                for x in candidates:
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
            block.sort()
            if len(block) == k:
                blocks.append(block)
                uncovered.difference_update(itertools.combinations(block, t))
    
    # Deduplicate blocks
    unique_blocks = []
    seen = set()
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
# EVOLVE-BLOCK-END
