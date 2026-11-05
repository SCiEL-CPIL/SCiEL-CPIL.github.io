import itertools
import time
import random
from collections import Counter
import numpy as np
from itertools import combinations

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
            
            # Phase 3: Final polish - try to remove more redundant blocks
            blocks = _remove_redundant(blocks, v, k, t, deadline)

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
    Uses vectorized gain computation for speed.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))
    
    # Weights for points
    weights = np.ones(v, dtype=np.float64)
    weight_increase = 1.1
    
    # Pre-compute all (t-1)-subsets for each point to speed up gain calculation
    # For a given current block S, gain(x) = number of (t-1)-subsets of S 
    # such that (t-1)-subset + x is in uncovered
    # We can precompute for each x, the set of (t-1)-subsets that would complete to a t-subset with x
    # But that's still large. Instead, let's use a different approach:
    # For each x, precompute all t-subsets containing x as a sorted tuple
    # Then gain(x) = |{s in (t-1)-subsets of S : s + x in uncovered}|
    
    # Actually, let's just optimize the inner loop with better data structures
    # Convert uncovered to a set of frozensets for faster hashing? No, tuples are fine.
    
    while uncovered:
        if time.time() > deadline:
            return None

        # Pick a random uncovered t-subset as the seed for the block
        seed_tuple = tuple(rng.choice(list(uncovered)))
        current_block_set = set(seed_tuple)

        while len(current_block_set) < k:
            if time.time() > deadline:
                return None

            best_point = -1
            best_score = -1.0
            best_gain = -1
            best_tiebreak = []

            # Pre-compute (t-1) subsets of current block
            sorted_block = sorted(current_block_set)
            t_minus_1_subsets = list(combinations(sorted_block, t - 1))
            
            # For each candidate x, compute gain
            # Optimize by precomputing the (t-1)-subsets as a list of tuples
            # and checking membership in uncovered set
            
            for x in points:
                if x in current_block_set:
                    continue

                gain = 0
                for sub in t_minus_1_subsets:
                    # Create the t-subset by adding x and sorting
                    # Since sub is already sorted and x is not in sub, we can merge
                    # But for simplicity, just sort
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in uncovered:
                        gain += 1

                if gain == 0:
                    score = -1.0
                else:
                    # Compute penalty using log-sum for numerical stability
                    log_penalty = 0.0
                    for p in current_block_set:
                        log_penalty += np.log(weights[p])
                    log_penalty += np.log(weights[x])
                    score = gain / np.exp(log_penalty)

                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_point = x
                    best_tiebreak = [x]
                elif score == best_score and gain > 0:
                    best_tiebreak.append(x)

            if best_point == -1:
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
        for sub in combinations(block, t):
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
    Optimized local search with better move evaluation.
    1. Remove redundant blocks.
    2. Attempt "merge" moves: Replace A with A' such that A' covers U_A U U_B,
       allowing B to be removed.
    3. Try single-block replacement: Replace block with a new one that covers its unique subsets.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)
    
    # Pre-compute t-subsets for all blocks
    block_tsubs = []
    for b in improved:
        block_tsubs.append(list(combinations(b, t)))
    
    # Build cover count
    cover_count = Counter()
    for subs in block_tsubs:
        for sub in subs:
            cover_count[sub] += 1
    
    points = list(range(v))
    no_progress_count = 0
    max_no_progress = 15
    
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
        
        # Try merge moves first
        for b_idx in indices:
            if time.time() > deadline:
                break
            if len(unique_subs[b_idx]) == 0:
                continue
                
            b_unique = unique_subs[b_idx]
            
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
                    
                    # Check distinctness
                    is_distinct = True
                    for j in range(len(improved)):
                        if j != a_idx and improved[j] == a_prime:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                        
                    # Perform replacement
                    for sub in block_tsubs[a_idx]:
                        cover_count[sub] -= 1
                        if cover_count[sub] == 0:
                            del cover_count[sub]
                            
                    new_tsubs = list(combinations(a_prime, t))
                    for sub in new_tsubs:
                        cover_count[sub] = cover_count.get(sub, 0) + 1
                    
                    improved[a_idx] = a_prime
                    block_tsubs[a_idx] = new_tsubs
                    
                    # Check if B is now redundant
                    b_redundant = True
                    for sub in block_tsubs[b_idx]:
                        if cover_count.get(sub, 0) <= 1:
                            b_redundant = False
                            break
                            
                    if b_redundant:
                        for sub in block_tsubs[b_idx]:
                            cover_count[sub] -= 1
                            if cover_count[sub] == 0:
                                del cover_count[sub]
                        del improved[b_idx]
                        del block_tsubs[b_idx]
                        move_made = True
                        break
                    else:
                        # Keep the change even if B isn't removed - it may help future moves
                        move_made = True
                        break
                        
            if move_made:
                no_progress_count = 0
                break
        
        if not move_made:
            no_progress_count += 1
            
    # Final pass to remove any remaining redundant blocks
    improved = _remove_redundant(improved, v, k, t, deadline)
    
    return improved
