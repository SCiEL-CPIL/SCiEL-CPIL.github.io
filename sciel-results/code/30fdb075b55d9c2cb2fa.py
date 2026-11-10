"""Bitset-accelerated greedy covering construction for C(24,7,3)."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 145  # safety margin below 150s

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}

    # For each block (k-subset), precompute its coverage mask as a bitset
    # Using Python integers as bitsets (arbitrary precision)
    # mask[b] has bit i set if t-subset i is covered by block b

    # Precompute for each t-subset index, which points it contains
    # all_t[i] is a tuple of t points

    # For greedy: we need to compute, for a partial block B and candidate x,
    # the number of newly covered t-subsets.
    # A t-subset is newly covered by B∪{x} (but not by B) iff:
    #   it contains x, and (t-1) of its other elements are in B,
    #   and it was not already covered by B.
    # For t=3: t-subsets newly covered = those containing x and 2 elements from B.
    # That's C(|B|, t-1) = C(|B|, 2) candidates when t=3.

    # Precompute: for each pair of points (a,b), the list of t-subset indices
    # that contain both a and b. For t=3, this is just the 3-subsets {a,b,c} for c in 0..v-1, c!=a,b.
    # More generally: for each (t-1)-subset, list of t-subsets extending it.

    t_minus_1 = t - 1
    # Map from (t-1)-tuple to list of t-subset indices
    ext_map = defaultdict(list)
    for ts in all_t:
        for i in range(t):
            sub = ts[:i] + ts[i+1:]
            ext_map[sub].append(t_index[ts])

    def greedy_cover(rng, time_limit):
        """Greedy set cover: repeatedly pick the block covering most uncovered t-subsets."""
        uncovered_mask = (1 << n_t) - 1  # all bits set
        blocks = []
        
        while uncovered_mask and time.time() < time_limit:
            # Pick a seed: find an uncovered t-subset
            # For speed, find first set bit
            idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            
            # Start with the t-subset's points
            seed = set(all_t[idx])
            
            # Greedily grow the block to size k
            block = seed
            for _ in range(k - t):
                if time.time() > time_limit:
                    break
                best_gain = -1
                best_x = None
                sorted_block = sorted(block)
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain: number of uncovered t-subsets containing x and (t-1) elements from block
                    gain = 0
                    for A in itertools.combinations(sorted_block, t_minus_1):
                        ts_tuple = tuple(sorted(A + (x,)))
                        if ts_tuple in t_index:
                            ti = t_index[ts_tuple]
                            if (uncovered_mask >> ti) & 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None and rng.random() < 0.5:
                        best_x = x
                if best_x is None or best_gain <= 0:
                    break
                block.add(best_x)
            
            if len(block) != k:
                # Pad with random points if needed
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.add(x)
            
            block = sorted(block)
            blocks.append(block)
            
            # Update uncovered mask
            for ts in itertools.combinations(block, t):
                ti = t_index[ts]
                uncovered_mask &= ~(1 << ti)
        
        return blocks

    def is_valid(blocks):
        covered = [False] * n_t
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered[t_index[ts]] = True
        for i in range(n_t):
            if not covered[i]:
                return False
        return True

    def remove_redundant(blocks, time_limit):
        """Iteratively remove blocks that are not needed."""
        n = len(blocks)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                # Check if block i is redundant
                candidate = blocks[:i] + blocks[i+1:]
                if is_valid(candidate):
                    blocks = candidate
                    n -= 1
                    changed = True
                    break  # restart from beginning
        return blocks

    def local_search_replace(blocks, time_limit):
        """Try to replace pairs of blocks with a single block."""
        n = len(blocks)
        improved = True
        while improved and time.time() < time_limit - 5:
            improved = False
            for i in range(n):
                if time.time() > time_limit - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit - 5:
                        break
                    # Find t-subsets only covered by block i or j
                    bi_set = set(blocks[i])
                    bj_set = set(blocks[j])
                    other_blocks = [blocks[m] for m in range(n) if m != i and m != j]
                    
                    required = set()
                    for ts in itertools.combinations(blocks[i], t):
                        ts_set = set(ts)
                        if not any(ts_set.issubset(set(ob)) for ob in other_blocks):
                            required.add(ts)
                    for ts in itertools.combinations(blocks[j], t):
                        ts_set = set(ts)
                        if not any(ts_set.issubset(set(ob)) for ob in other_blocks):
                            required.add(ts)
                    
                    if not required:
                        # Both blocks are redundant (should have been caught by remove_redundant)
                        continue
                    
                    req_points = set()
                    for ts in required:
                        req_points.update(ts)
                    
                    if len(req_points) > k:
                        continue
                    
                    # Try to build a block containing req_points
                    new_block = set(req_points)
                    for attempt in range(30):
                        trial = set(new_block)
                        while len(trial) < k:
                            candidates = [x for x in range(v) if x not in trial]
                            if not candidates:
                                break
                            trial.add(rng_choice(candidates, i * 1000 + j * 7 + attempt))
                        
                        # Check if trial covers all required
                        ok = all(set(ts).issubset(trial) for ts in required)
                        if ok:
                            candidate = blocks[:i] + blocks[i+1:j] + blocks[j+1:] + [sorted(trial)]
                            if is_valid(candidate):
                                blocks = candidate
                                n = len(blocks)
                                improved = True
                                break
                    if improved:
                        break
        return blocks

    def rng_choice(candidates, seed_val):
        return candidates[seed_val % len(candidates)]

    # Main: run multiple randomized greedy, keep best
    best_result = None
    best_len = float('inf')
    
    n_trials = 8
    for trial in range(n_trials):
        if time.time() > deadline - 25:
            break
        rng = random.Random(trial * 7919 + 12345)
        result = greedy_cover(rng, deadline - 20)
        if is_valid(result):
            # Remove redundant blocks
            result = remove_redundant(result, deadline - 15)
            if len(result) < best_len:
                best_len = len(result)
                best_result = result
    
    # Local search on best
    if best_result is not None and time.time() < deadline - 10:
        improved = local_search_replace(best_result, deadline - 5)
        if is_valid(improved):
            best_result = improved
            best_result = remove_redundant(best_result, deadline - 2)
    
    if best_result is None:
        rng = random.Random(0)
        best_result = greedy_cover(rng, deadline)
    
    return best_result
