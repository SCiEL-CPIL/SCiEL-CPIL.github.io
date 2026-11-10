import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each point x, the list of t-subsets containing x
    # point_tsubs[x] = list of indices of t-subsets that contain x
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for x in ts:
            point_tsubs[x].append(i)
    
    # Precompute for each pair of points (x,y), the list of t-subsets containing both
    # This helps with gain computation when adding a point to a block
    # Actually, let's think about this differently.
    
    # For a block B (set of points), the gain of adding point x is:
    # number of t-subsets S such that x in S, S \ {x} subset of B, and S is uncovered
    # = number of (t-1)-subsets U of B such that U ∪ {x} is a t-subset and is uncovered
    
    # Let's precompute for each point x and each subset size up to t-1,
    # but that's too much. Instead, let's compute gains on the fly but efficiently.
    
    # Key optimization: for a block B with m elements, the gain of adding x is
    # sum over (t-1)-subsets U of B: is (U ∪ {x}) uncovered?
    # We can precompute for each x, a mapping from frozenset of size t-1 to index
    # Actually, let's just use the tsub_idx lookup.
    
    # Better approach: precompute for each point x, a dictionary mapping
    # frozenset of t-1 points to the index of the t-subset (that frozenset ∪ {x})
    # But that's C(18,4) per point = 3060, times 19 points = 58140 entries. Manageable.
    
    # Actually, even better: for each x, precompute a dict:
    # x_gain_dict[x][frozenset_U] = index of t-subset U ∪ {x}
    x_gain_dict = [{} for _ in range(v)]
    for x in range(v):
        for U in itertools.combinations([p for p in range(v) if p != x], t-1):
            ts = tuple(sorted(U + (x,)))
            x_gain_dict[x][frozenset(U)] = tsub_idx[ts]
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    def greedy_build(cover_count, uncovered_set, point_order, time_limit):
        """Build blocks greedily until all t-subsets are covered or time runs out."""
        blocks = []
        while len(uncovered_set) > 0:
            if time.time() > time_limit:
                break
            
            # Pick a random uncovered t-subset to seed the block
            seed_idx = next(iter(uncovered_set))
            seed = list(all_tsubsets[seed_idx])
            block = set(seed)
            
            # Greedily add points to maximize gain
            while len(block) < k:
                if time.time() > time_limit:
                    break
                
                m = len(block)
                if m < t - 1:
                    # Can't form any (t-1)-subset yet, add random point
                    candidates = [x for x in point_order if x not in block]
                    if not candidates:
                        break
                    block.add(candidates[0])
                    continue
                
                # Compute gain for each candidate point
                best_gain = -1
                best_point = None
                second_best_gain = -1
                second_best_point = None
                
                block_list = list(block)
                (t_minus_1_subsets) = list(itertools.combinations(block_list, t-1))
                
                for x in point_order:
                    if x in block:
                        continue
                    
                    gain = 0
                    for U in (t_minus_1_subsets):
                        key = frozenset(U)
                        idx = x_gain_dict[x].get(key, -1)
                        if idx != -1 and idx in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        second_best_gain = best_gain
                        second_best_point = best_point
                        best_gain = gain
                        best_point = x
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = x
                
                if best_point is None:
                    # No point improves coverage; pick any
                    candidates = [x for x in point_order if x not in block]
                    if not candidates:
                        break
                    block.add(candidates[0])
                else:
                    block.add(best_point)
            
            if len(block) < k:
                continue
            
            block_list = sorted(block)
            blocks.append(block_list)
            
            # Update coverage
            for ts in itertools.combinations(block_list, t):
                idx = tsub_idx[ts]
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
                    cover_count[idx] += 1
                else:
                    cover_count[idx] += 1
        
        return blocks
    
    def local_search_2exchange(blocks, cover_count, uncovered_set, time_limit, rng):
        """Try 2-exchange moves: replace one block with another to reduce block count."""
        blocks_set = set(tuple(b) for b in blocks)
        blocks_list = list(blocks)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove a block and see if we can still cover everything
            # Actually, 2-exchange: swap block i with a new block
            # Simpler: try to remove each block; if uncovered subsets appear,
            # try to find a single block that covers all of them (plus maybe more)
            
            for i in range(len(blocks_list)):
                if time.time() > time_limit:
                    break
                
                # Tentatively remove block i
                removed_block = blocks_list[i]
                newly_uncovered = set()
                for ts in itertools.combinations(removed_block, t):
                    idx = tsub_idx[ts]
                    if cover_count[idx] == 1:
                        newly_uncovered.add(idx)
                
                if not newly_uncovered:
                    # Block i is redundant! Remove it.
                    for ts in itertools.combinations(removed_block, t):
                        idx = tsub_idx[ts]
                        cover_count[idx] -= 1
                        if idx in uncovered_set:
                            # Shouldn't happen if cover_count was 1
                            pass
                    del blocks_list[i]
                    improved = True
                    break  # Restart the loop since list changed
                
                # Try to find a single block that covers all newly_uncovered subsets
                # The new block must contain all points in each newly_uncovered t-subset
                # That means it must contain the union of all points in newly_uncovered
                required_points = set()
                for idx in newly_uncovered:
                    required_points.update(all_tsubsets[idx])
                
                if len(required_points) > k:
                    continue  # Can't fit in one block
                
                # Find a block of size k containing required_points that maximizes new coverage
                remaining_points = [p for p in range(v) if p not in required_points]
                need = k - len(required_points)
                
                best_new_block = None
                best_new_gain = 0
                
                # Try random subsets of remaining_points of size 'need'
                num_tries = min(50, len(remaining_points))
                for _ in range(num_tries):
                    chosen = rng.sample(remaining_points, need) if need > 0 else []
                    candidate = sorted(required_points | set(chosen))
                    if tuple(candidate) in blocks_set:
                        continue
                    
                    # Compute how many currently uncovered (or only-covered-by-removed) subsets this covers
                    gain = 0
                    for ts in itertools.combinations(candidate, t):
                        idx = tsub_idx[ts]
                        if idx in newly_uncovered:
                            gain += 1
                    
                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new_block = candidate
                
                if best_new_block is not None and best_new_gain == len(newly_uncovered):
                    # Great! Replace block i with best_new_block
                    # First, undo the removal (we didn't actually remove yet in this branch)
                    # Apply the new block
                    for ts in itertools.combinations(best_new_block, t):
                        idx = tsub_idx[ts]
                        if idx in newly_uncovered:
                            # It was going to be uncovered, now it's covered by new block
                            # cover_count stays the same (was 1 from removed, now 1 from new)
                            pass
                        else:
                            cover_count[idx] += 1
                            if cover_count[idx] == 1:
                                # Was uncovered, now covered
                                if idx in uncovered_set:
                                    uncovered_set.discard(idx)
                    
                    # Remove old block's coverage
                    for ts in itertools.combinations(removed_block, t):
                        idx = tsub_idx[ts]
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            uncovered_set.add(idx)
                    
                    blocks_list[i] = best_new_block
                    improved = True
                    break  # Restart
        
        return blocks_list
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random point order
        point_order = list(range(v))
        rng.shuffle(point_order)
        
        cover_count = [0] * num_tsubsets
        uncovered_set = set(range(num_tsubsets))
        
        # Build initial covering
        local_time_limit = deadline - 5.0  # Leave 5s for local search
        blocks = greedy_build(cover_count, uncovered_set, point_order, local_time_limit)
        
        if len(uncovered_set) > 0:
            # Incomplete covering; try to finish quickly
            # Just add blocks for remaining uncovered subsets
            while len(uncovered_set) > 0 and time.time() < deadline - 2.0:
                seed_idx = next(iter(uncovered_set))
                seed = list(all_tsubsets[seed_idx])
                block = set(seed)
                while len(block) < k:
                    candidates = [x for x in point_order if x not in block]
                    if not candidates:
                        break
                    block.add(candidates[0])
                if len(block) < k:
                    break
                block_list = sorted(block)
                blocks.append(block_list)
                for ts in itertools.combinations(block_list, t):
                    idx = tsub_idx[ts]
                    if idx in uncovered_set:
                        uncovered_set.remove(idx)
                    cover_count[idx] += 1
        
        if len(uncovered_set) > 0:
            continue  # Incomplete, skip this restart
        
        # Local search
        if time.time() < deadline - 1.0:
            blocks = local_search_2exchange(blocks, cover_count, uncovered_set, deadline - 1.0, rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        if best_count <= 264:
            break
    
    if best_blocks is None:
        return []
    
    return best_blocks
