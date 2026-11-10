import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0
    margin = 5.0
    
    def remaining_time():
        return deadline - margin - (time.time() - start_time)
    
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    
    if not uncovered:
        return []
    
    blocks = []
    
    # Precompute t-subsets for each possible k-block for faster gain calculation
    block_tsubs = {}
    
    def get_block_tsubs(block):
        """Get or compute the t-subsets contained in a block."""
        if block not in block_tsubs:
            block_tsubs[block] = set(itertools.combinations(block, t))
        return block_tsubs[block]
    
    # Track frequency of each point in selected blocks
    point_freq = np.zeros(v, dtype=np.int32)
    
    # Greedy phase with hard-subset prioritization
    while uncovered and remaining_time() > 0:
        best_gain = -1
        best_block = None
        
        # Prioritize 'hard' uncovered subsets: those with points rarely used
        # Compute difficulty score for each uncovered subset
        # Hard subsets have points with low frequency
        if len(uncovered) <= 10000:
            # For smaller sets, compute difficulty scores
            hard_scores = {}
            for sub in uncovered:
                # Score = sum of inverse frequencies (higher = harder)
                score = 0.0
                for p in sub:
                    score += 1.0 / (point_freq[p] + 1)
                hard_scores[sub] = score
            # Pick the hardest subset as seed
            seed = max(hard_scores, key=hard_scores.get)
        else:
            # For larger sets, random selection with slight bias toward hard points
            # Sample a few candidates and pick the hardest
            candidates = random.sample(list(uncovered), min(50, len(uncovered)))
            seed = max(candidates, key=lambda sub: sum(1.0 / (point_freq[p] + 1) for p in sub))
        
        seed_set = set(seed)
        remaining_points = [x for x in range(v) if x not in seed_set]
        num_to_add = k - t
        
        for add_combo in itertools.combinations(remaining_points, num_to_add):
            candidate_block = tuple(sorted(seed + add_combo))
            
            gain = 0
            for sub in itertools.combinations(candidate_block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
            elif gain == best_gain and best_block is not None:
                # Tie-breaking: prefer blocks that use less frequent points
                # This helps spread coverage more evenly
                cand_freq = sum(point_freq[p] for p in candidate_block)
                best_freq = sum(point_freq[p] for p in best_block)
                if cand_freq < best_freq:
                    best_block = candidate_block
                elif cand_freq == best_freq:
                    if candidate_block < best_block:
                        best_block = candidate_block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Update uncovered set and point frequencies
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
        for p in best_block:
            point_freq[p] += 1
    
    # Local improvement: remove redundant blocks
    if remaining_time() > 5.0:
        changed = True
        while changed and remaining_time() > 5.0:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                block_i = tuple(blocks[i])
                tsubs_i = get_block_tsubs(block_i)
                
                # Check if all t-subsets of block_i are covered by other blocks
                all_covered = True
                for sub in tsubs_i:
                    covered = False
                    for j, block_j in enumerate(blocks):
                        if j == i:
                            continue
                        if sub in get_block_tsubs(tuple(block_j)):
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break
                
                if all_covered:
                    blocks.pop(i)
                    changed = True
                    break  # Restart from beginning since indices changed
    
    # Second pass: try to improve by swapping blocks
    if remaining_time() > 10.0:
        # Try replacing each block with a better one
        for iteration in range(2):
            if remaining_time() < 10.0:
                break
            improved = True
            while improved and remaining_time() > 10.0:
                improved = False
                for i in range(len(blocks)):
                    if remaining_time() < 10.0:
                        break
                    block_i = tuple(blocks[i])
                    tsubs_i = get_block_tsubs(block_i)
                    
                    # Find t-subsets only covered by this block
                    unique_tsubs = set()
                    for sub in tsubs_i:
                        only_in_i = True
                        for j, block_j in enumerate(blocks):
                            if j == i:
                                continue
                            if sub in get_block_tsubs(tuple(block_j)):
                                only_in_i = False
                                break
                        if only_in_i:
                            unique_tsubs.add(sub)
                    
                    if not unique_tsubs:
                        continue
                    
                    # Try to find a replacement block that covers all unique_tsubs
                    # and potentially more
                    best_new_gain = -1
                    best_new_block = None
                    
                    # The replacement must cover all unique_tsubs
                    # So it must contain all points in the union of unique_tsubs? No,
                    # it must contain each t-subset in unique_tsubs
                    # This means for each sub in unique_tsubs, all points of sub must be in the block
                    
                    # Find the intersection of all points that must be in the block
                    must_have = set()
                    for sub in unique_tsubs:
                        must_have.update(sub)
                    
                    if len(must_have) > k:
                        continue  # Can't fit all required points
                    
                    # Find all k-blocks that contain must_have
                    remaining_for_block = [x for x in range(v) if x not in must_have]
                    num_to_add = k - len(must_have)
                    
                    for add_combo in itertools.combinations(remaining_for_block, num_to_add):
                        candidate_block = tuple(sorted(tuple(must_have) + add_combo))
                        
                        # Check if it covers all unique_tsubs
                        valid = True
                        for sub in unique_tsubs:
                            if not set(sub).issubset(set(candidate_block)):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Count how many currently uncovered or under-covered t-subsets it covers
                        gain = 0
                        for sub in itertools.combinations(candidate_block, t):
                            # Count how many blocks currently cover this sub
                            cover_count = 0
                            for j, block_j in enumerate(blocks):
                                if j == i:
                                    continue
                                if sub in get_block_tsubs(tuple(block_j)):
                                    cover_count += 1
                            # If cover_count == 0, this is newly covered
                            # If cover_count == 1 and j==i was covering it, removing i would uncover it
                            # So we want blocks that maintain coverage
                            if cover_count == 0:
                                gain += 1
                            elif cover_count == 1:
                                # Check if block_i was the one covering it
                                if sub in tsubs_i:
                                    gain += 1  # We're replacing the only cover
                    
                    if best_new_block is not None and best_new_gain > 0:
                        blocks[i] = list(best_new_block)
                        improved = True
    
    return blocks
