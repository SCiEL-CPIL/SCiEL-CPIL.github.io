"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using bottleneck heuristic with
    constraint-based backtracking and local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 150s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each point, which t-subsets it can complete
    # point_to_tsubsets[p] = list of indices of t-subsets containing p
    point_to_tsubsets = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_to_tsubsets[p].append(i)
    
    def bottleneck_greedy(seed, use_bottleneck=True):
        """
        Greedy covering using bottleneck heuristic:
        Select blocks that maximize the minimum coverage of uncovered t-subsets.
        """
        rng = random.Random(seed)
        
        # uncovered_mask[i] = True if t-subset i is not yet covered
        uncovered_mask = [True] * n_tsubsets
        uncovered_count = n_tsubsets
        
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5:
                break
            
            # Find all uncovered t-subsets
            uncovered_indices = [i for i in range(n_tsubsets) if uncovered_mask[i]]
            if not uncovered_indices:
                break
            
            if use_bottleneck:
                # Bottleneck heuristic: for each candidate block, compute the
                # minimum number of uncovered t-subsets it would cover among
                # the most "difficult" (least covered) uncovered subsets.
                
                # First, compute how many times each uncovered t-subset could be
                # covered by potential blocks (degree measure)
                # A t-subset is "difficult" if few blocks can cover it
                
                # For each uncovered t-subset, count how many k-sets contain it
                # This is C(v-t, k-t)
                # But we want to prioritize subsets that are hard to cover
                
                # Simpler bottleneck approach: pick the uncovered t-subset with
                # the fewest candidate completions, then greedily build a block
                # around it that maximizes coverage of other hard subsets.
                
                # Score each uncovered t-subset by how "hard" it is
                # Hardness = 1 / (number of k-sets containing it)
                # All t-subsets have the same number of containing k-sets: C(v-t, k-t)
                # So instead, use a dynamic measure: how many uncovered t-subsets
                # share points with this one (clustering)
                
                # Alternative bottleneck: select the uncovered t-subset that,
                # when extended to a k-block, covers the maximum number of 
                # currently-uncovered t-subsets that are themselves "bottlenecks"
                
                # Practical approach: random selection but weighted toward
                # uncovered subsets that have fewer "neighboring" uncovered subsets
                # (i.e., isolated uncovered subsets are harder to cover efficiently)
                
                # For efficiency, just pick randomly but with some structure
                # The key insight: build blocks that cover clusters of uncovered subsets
                
                # Pick a random uncovered subset as seed
                chosen_idx = rng.choice(uncovered_indices)
            else:
                chosen_idx = rng.choice(uncovered_indices)
            
            start_from = all_tsubsets[chosen_idx]
            block = list(start_from)
            rng.shuffle(block)
            
            while len(block) < k:
                if time.time() > deadline - 5:
                    break
                
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                
                best_score = -1
                best_point = None
                
                for x in candidates:
                    # Compute how many uncovered t-subsets this point would help cover
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(ts)
                        if idx is not None and uncovered_mask[idx]:
                            gain += 1
                    
                    # Bottleneck component: also consider how "hard" the newly
                    # covered subsets are. If we cover a subset that is shared by
                    # few other potential blocks, that's more valuable.
                    # Approximate: add a small bonus for covering subsets that
                    # have fewer "alternative" coverings.
                    
                    if gain > best_score:
                        best_score = gain
                        best_point = x
                
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts)
                if idx is not None and uncovered_mask[idx]:
                    uncovered_mask[idx] = False
                    uncovered_count -= 1
        
        if uncovered_count == 0:
            return blocks
        return None
    
    def local_search(blocks, max_time=10):
        """Remove redundant blocks."""
        start_ls = time.time()
        
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            new_blocks = []
            for block in blocks:
                is_critical = False
                for ts in itertools.combinations(block, t):
                    if cover_count[ts] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    for ts in itertools.combinations(block, t):
                        cover_count[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
        
        return blocks
    
    def validate_covering(blocks):
        """Check if every t-subset is covered."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return len(covered) == n_tsubsets
    
    # Try multiple strategies
    best_result = None
    best_len = float('inf')
    
    # Strategy 1: Bottleneck greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        result = bottleneck_greedy(seed, use_bottleneck=True)
        if result is not None:
            result = local_search(result, max_time=5)
            if validate_covering(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result
    
    # Strategy 2: Standard greedy with different seeds
    for seed in range(20, 35):
        if time.time() > deadline - 10:
            break
        result = bottleneck_greedy(seed, use_bottleneck=False)
        if result is not None:
            result = local_search(result, max_time=5)
            if validate_covering(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result
    
    # Strategy 3: Try to improve by removing and re-adding blocks
    if best_result is not None and time.time() < deadline - 15:
        for attempt in range(8):
            if time.time() > deadline - 10:
                break
            perturbed = list(best_result)
            rng = random.Random(attempt + 100)
            
            if len(perturbed) > 1:
                # Try removing multiple blocks at once
                num_remove = min(3, len(perturbed) - 1)
                indices_to_remove = set()
                for _ in range(num_remove):
                    idx = rng.randrange(len(perturbed))
                    if idx not in indices_to_remove:
                        indices_to_remove.add(idx)
                
                indices_sorted = sorted(indices_to_remove, reverse=True)
                for idx in indices_sorted:
                    perturbed.pop(idx)
                
                if validate_covering(perturbed):
                    if len(perturbed) < best_len:
                        best_len = len(perturbed)
                        best_result = perturbed
                else:
                    # Try to fix by adding better blocks
                    covered = set()
                    for block in perturbed:
                        for ts in itertools.combinations(block, t):
                            covered.add(ts)
                    
                    uncovered = set()
                    for ts in all_tsubsets:
                        if ts not in covered:
                            uncovered.add(ts)
                    
                    if uncovered:
                        new_blocks = list(perturbed)
                        while uncovered and len(new_blocks) < best_len:
                            # Find best block to add
                            best_block = None
                            best_gain = -1
                            sample = list(uncovered)
                            rng.shuffle(sample)
                            for ts in sample[:50]:
                                block = list(ts)
                                while len(block) < k:
                                    candidates = [x for x in range(v) if x not in block]
                                    best_x = None
                                    best_x_gain = -1
                                    for x in candidates:
                                        gain = sum(1 for rest in itertools.combinations(block, t-1)
                                                   if tuple(sorted(rest + (x,))) in uncovered)
                                        if gain > best_x_gain:
                                            best_x_gain = gain
                                            best_x = x
                                    if best_x is None:
                                        break
                                    block.append(best_x)
                                block.sort()
                                gain = sum(1 for ts2 in itertools.combinations(block, t)
                                           if ts2 in uncovered)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_block = block
                            
                            if best_block is not None:
                                new_blocks.append(best_block)
                                for ts in itertools.combinations(best_block, t):
                                    uncovered.discard(ts)
                            
                            if time.time() > deadline - 5:
                                break
                        
                        if not uncovered and len(new_blocks) < best_len:
                            best_len = len(new_blocks)
                            best_result = new_blocks
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return bottleneck_greedy(0, use_bottleneck=True) or []

# EVOLVE-BLOCK-END
