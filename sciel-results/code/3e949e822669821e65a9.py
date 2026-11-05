import itertools
import time
import random
from collections import Counter
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    n_tsub = len(all_tsubsets_set)

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial)
        if blocks is not None:
            # Phase 1: Remove redundant blocks (greedy pass)
            blocks = _remove_redundant(blocks, v, k, t, deadline)

            # Phase 2: Local search / refinement
            # We attempt to reduce the count further by replacing blocks
            blocks = _local_search(blocks, v, k, t, all_tsubsets_set, deadline, trial)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

        # Early termination if we hit the known best
        if best_len <= 76:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]


def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy set cover with weighted heuristic.
    Score for candidate x = marginal_gain / (product of weights of points in block U {x})
    Weights are initialized to 1.0.
    After adding a block, weights of points in the block are increased slightly
    to discourage future blocks from over-using them (symmetry breaking / balancing).
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))
    
    # Weights for points
    weights = [1.0] * v
    # Decay factor for weights? Or just increase?
    # If we increase, later blocks avoid heavy points.
    # This enforces a more "spread out" design.
    weight_increase = 1.1
    
    while uncovered:
        if time.time() > deadline:
            return None

        # Pick a random uncovered t-subset as the seed for the block
        # This helps diversify the search compared to always picking the first
        seed_tuple = tuple(rng.choice(list(uncovered)))
        current_block_set = set(seed_tuple)
        # Pre-compute t-1 subsets of seed to speed up first iteration?
        # Actually, the loop below handles it.

        # Greedily add points to maximize coverage of *uncovered* t-subsets
        # adjusted by point weights
        while len(current_block_set) < k:
            if time.time() > deadline:
                return None

            best_point = -1
            best_score = -1.0
            best_gain = -1
            best_tiebreak = []

            # Pre-compute (t-1) subsets of current block for fast gain calculation
            t_minus_1_subsets = list(itertools.combinations(sorted(current_block_set), t - 1))
            
            # Optimization: For each x, we want to count how many of the t-1 subsets 
            # combined with x are in `uncovered`.
            # Instead of iterating x then subsets, we can pre-compute the "potential" 
            # for each x? 
            # Let's stick to the loop but optimize the inner check.
            
            # We can pre-compute a set of all (t-1)-subsets of the *uncovered* set?
            # No, that's huge.
            
            # Let's try a different heuristic for speed:
            # Vectorized gain calculation using NumPy?
            # Uncovered is a set of tuples. Hard to vectorize directly without mapping.
            
            # Let's just keep the Python loop but ensure it's tight.
            
            for x in points:
                if x in current_block_set:
                    continue

                gain = 0
                for sub in t_minus_1_subsets:
                    # Optimization: check membership in set
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in uncovered:
                        gain += 1

                if gain == 0:
                    score = -1.0
                else:
                    # Use sum of logs for penalty to avoid overflow/underflow issues 
                    # and speed up multiplication? 
                    # Actually k=7 is small, product is fine.
                    penalty = 1.0
                    for p in current_block_set:
                        penalty *= weights[p]
                    penalty *= weights[x]
                    score = gain / penalty

                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_point = x
                    best_tiebreak = [x]
                elif score == best_score and gain > 0:
                    best_tiebreak.append(x)

            if best_point == -1:
                # If no point has positive gain, we are stuck or the cover is complete?
                # If uncovered is not empty, but no point adds coverage, it means 
                # the current block's t-1 subsets combined with any new point 
                # are already covered.
                # This can happen if the remaining uncovered subsets are "far" from the current block.
                # In this case, we should probably abandon this block construction 
                # and start a new one? Or just pick random points to fill it?
                # Picking random points might be bad because it wastes a block.
                # Better to return None and let the main loop restart with a different seed?
                # But we've already spent time on this block.
                # Let's just pick random points to fill the block. It's a valid block, 
                # even if it doesn't cover anything new. It might cover something 
                # indirectly? No, gain is 0.
                # So it's a "wasted" block.
                # It's better to abort this trial if we get stuck early?
                # But we are in the middle of constructing a block.
                # Let's just fill with randoms.
                available = [x for x in points if x not in current_block_set]
                if not available:
                    return None
                best_point = rng.choice(available)
                best_gain = 0

            current_block_set.add(best_point)

        block = tuple(sorted(current_block_set))
        blocks.append(block)

        # Update weights
        for p in block:
            weights[p] *= weight_increase

        # Remove all t-subsets covered by this new block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Iteratively removes blocks that are redundant (every t-subset in them is covered by others).
    """
    if not blocks:
        return blocks

    # Count how many times each t-subset is covered
    # We only care if count >= 1.
    # Optimization: Use a set of "critical" subsets?
    # Simple approach: Count coverage for all.
    cover_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1

    improved = list(blocks)
    changed = True

    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = improved[i]
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                # Remove block
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                improved.pop(i)
                changed = True

    return improved


def _local_search(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Optimized local search.
    1. Remove redundant blocks.
    2. Attempt "merge" moves: Replace A with A' such that A' covers U_A U U_B,
       allowing B to be removed.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)
    
    # Pre-compute t-subsets for all blocks
    block_tsubs = []
    for b in improved:
        block_tsubs.append(list(itertools.combinations(b, t)))
    
    # Build cover count
    cover_count = Counter()
    for subs in block_tsubs:
        for sub in subs:
            cover_count[sub] += 1
    
    points = list(range(v))
    no_progress_count = 0
    max_no_progress = 10
    
    while time.time() < deadline and no_progress_count < max_no_progress:
        
        # 1. Check for and remove redundant blocks
        redundant_found = False
        n = len(improved)
        for i in range(n):
            is_redundant = True
            for sub in block_tsubs[i]:
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for sub in block_tsubs[i]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]
                del improved[i]
                del block_tsubs[i]
                redundant_found = True
                break
        
        if redundant_found:
            no_progress_count = 0
            continue
        
        # 2. Try merge moves
        n = len(improved)
        if n < 2:
            break
            
        # Compute unique subsets for each block
        unique_subs = [None] * n
        for i in range(n):
            unique_subs[i] = [s for s in block_tsubs[i] if cover_count[s] == 1]
            
        # Shuffle blocks to randomize target selection
        indices = list(range(n))
        rng.shuffle(indices)
        
        move_made = False
        for b_idx in indices:
            if time.time() > deadline:
                break
            if len(unique_subs[b_idx]) == 0:
                # Should have been caught by redundancy check, but just in case
                continue
                
            b_unique = unique_subs[b_idx]
            
            # Try to find A
            # We want A such that Union(U_A, U_B) fits in k points.
            # To maximize chance, we should pick A with small U_A and "compatible" points.
            # Let's just iterate through other blocks.
            
            for a_idx in indices:
                if a_idx == b_idx:
                    continue
                if time.time() > deadline:
                    break
                    
                a_unique = unique_subs[a_idx]
                
                # Check if union of points in U_A and U_B fits in k
                union_points = set()
                for s in a_unique:
                    union_points.update(s)
                for s in b_unique:
                    union_points.update(s)
                    
                if len(union_points) <= k:
                    # Construct A'
                    a_prime = list(union_points)
                    # Pad with random points
                    while len(a_prime) < k:
                        p = rng.randint(0, v - 1)
                        if p not in a_prime:
                            a_prime.append(p)
                    a_prime = tuple(sorted(a_prime))
                    
                    # Check if A' is distinct from existing blocks?
                    # The problem says blocks must be distinct.
                    # If A' is already in the set, we can't add it?
                    # We are *replacing* A with A'.
                    # If A' == A, no change.
                    # If A' == B, we replace A with B? Then we have two B's?
                    # We need to ensure distinctness.
                    # Let's check if A' is in the current set (excluding A).
                    # If A' is in the set, we can't use this move as a simple replacement 
                    # that maintains distinctness without removing the duplicate?
                    # Actually, if A' == B, and we replace A with A', we have {B, B, ...}.
                    # We can then remove one B?
                    # This is getting complicated.
                    # Simple check: if A' is already in improved (as a tuple), skip?
                    # Or allow duplicates and let _remove_redundant handle it?
                    # The prompt says "blocks must be distinct".
                    # So we must ensure A' is not in improved \ {A}.
                    
                    is_distinct = True
                    for j in range(len(improved)):
                        if j != a_idx and improved[j] == a_prime:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                        
                    # Perform replacement
                    # Remove A's contribution
                    for sub in block_tsubs[a_idx]:
                        cover_count[sub] -= 1
                        if cover_count[sub] == 0:
                            del cover_count[sub]
                            
                    # Add A' contribution
                    new_tsubs = list(itertools.combinations(a_prime, t))
                    for sub in new_tsubs:
                        cover_count[sub] = cover_count.get(sub, 0) + 1
                    
                    # Update structures
                    improved[a_idx] = a_prime
                    block_tsubs[a_idx] = new_tsubs
                    
                    # Check if B is now redundant
                    b_redundant = True
                    for sub in block_tsubs[b_idx]:
                        if cover_count.get(sub, 0) <= 1:
                            b_redundant = False
                            break
                            
                    if b_redundant:
                        # Remove B
                        for sub in block_tsubs[b_idx]:
                            cover_count[sub] -= 1
                            if cover_count[sub] == 0:
                                del cover_count[sub]
                        del improved[b_idx]
                        del block_tsubs[b_idx]
                        move_made = True
                        break
                    else:
                        # Revert? Or keep?
                        # If we keep it, we might have improved the "slack" even if B wasn't removed.
                        # But we didn't reduce count.
                        # Let's revert to avoid wasting time on non-improving swaps?
                        # Or keep it to help future moves?
                        # Let's keep it. It might help.
                        move_made = True
                        break
                        
            if move_made:
                no_progress_count = 0
                break # Restart outer while loop to re-evaluate redundancy
                
        if not move_made:
            no_progress_count += 1
            
    return improved
