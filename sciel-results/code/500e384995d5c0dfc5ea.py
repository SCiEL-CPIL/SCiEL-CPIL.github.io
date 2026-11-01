"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None


def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = _global_deadline if _global_deadline else float('inf')
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed (randomization helps)
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are completed by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: iteratively remove redundant
    blocks (those whose removal doesn't leave any t-subset uncovered)."""
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage: for each t-subset, count how many blocks cover it
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Check if removing block bi leaves any t-subset with count 1
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                removed_block = blocks.pop(bi)
                # Decrement counts for all t-subsets in removed block
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
    return blocks


def _pairwise_merge_search(blocks, v, k, t, deadline, rng=None):
    """Aggressive local search: try to replace pairs of blocks with a single block
    that covers the union of their unique t-subsets. Also try single-block removals."""
    if rng is None:
        rng = random.Random()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # First pass: try removing redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                return blocks
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        
        if improved or time.time() >= deadline:
            continue
        
        # Second pass: try merging two blocks into one
        # This is expensive, so only do it occasionally
        if rng.random() < 0.5:
            bi = rng.randrange(len(blocks))
            bj = rng.randrange(len(blocks))
            if bi == bj:
                continue
            
            # Check if union of t-subsets from blocks bi and bj can be covered by one block
            b_i = set(blocks[bi])
            b_j = set(blocks[bj])
            
            # Try to find a k-subset that covers t-subsets from both blocks
            # Start with the union, try to reduce to k elements
            candidate = sorted(b_i | b_j)
            if len(candidate) <= k:
                candidate_block = candidate
            else:
                # Try greedy reduction: keep elements that cover the most unique t-subsets
                # Unique t-subsets are those covered by bi or bj but not by any other block
                unique_ts = set()
                for ts in itertools.combinations(blocks[bi], t):
                    ts = tuple(sorted(ts))
                    if coverage_count.get(ts, 0) == 1:
                        unique_ts.add(ts)
                for ts in itertools.combinations(blocks[bj], t):
                    ts = tuple(sorted(ts))
                    if coverage_count.get(ts, 0) == 1:
                        unique_ts.add(ts)
                
                if not unique_ts:
                    # No unique t-subsets, so any k-subset of the union works
                    candidate_block = list(rng.sample(candidate, k))
                else:
                    # Greedily build a block that covers the unique t-subsets
                    candidate_block = []
                    candidate_set = set()
                    remaining_unique = set(unique_ts)
                    
                    while len(candidate_block) < k and remaining_unique:
                        # Find the element that covers the most remaining unique t-subsets
                        best_x = -1
                        best_gain = -1
                        for x in candidate:
                            if x in candidate_set:
                                continue
                            gain = 0
                            for ts in remaining_unique:
                                if x in ts:
                                    # Check if all other elements of ts are in candidate_set or can be added
                                    # Simplified: just count if x is in ts
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x < 0 or best_gain <= 0:
                            break
                        candidate_block.append(best_x)
                        candidate_set.add(best_x)
                        # Remove t-subsets that are now fully covered
                        new_remaining = set()
                        for ts in remaining_unique:
                            if all(x in candidate_set for x in ts):
                                pass  # covered
                            else:
                                new_remaining.add(ts)
                        remaining_unique = new_remaining
                    
                    # Fill remaining slots with random elements from candidate
                    while len(candidate_block) < k:
                        available = [x for x in candidate if x not in candidate_set]
                        if not available:
                            break
                        x = rng.choice(available)
                        candidate_block.append(x)
                        candidate_set.add(x)
            
            # Check if this candidate block covers all t-subsets that were ONLY
            # covered by blocks bi or bj
            candidate_set = set(candidate_block)
            candidate_ts = set()
            for ts in itertools.combinations(candidate_block, t):
                candidate_ts.add(tuple(sorted(ts)))
            
            # Check if removing bi and bj and adding candidate_block is valid
            safe = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1 and ts not in candidate_ts:
                    safe = False
                    break
            if safe:
                for ts in itertools.combinations(blocks[bj], t):
                    ts = tuple(sorted(ts))
                    if coverage_count.get(ts, 0) <= 1 and ts not in candidate_ts:
                        safe = False
                        break
            
            if safe:
                # Perform the merge
                # Remove bi and bj
                for ts in itertools.combinations(blocks[bi], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                for ts in itertools.combinations(blocks[bj], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                
                # Remove the blocks
                blocks.pop(max(bi, bj))
                blocks.pop(min(bi, bj))
                
                # Add the new block
                blocks.append(candidate_block)
                for ts in itertools.combinations(candidate_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] = coverage_count.get(ts, 0) + 1
                
                improved = True
                break
    
    return blocks


def _triple_merge_search(blocks, v, k, t, deadline, rng=None):
    """Try to replace three blocks with one or two blocks."""
    if rng is None:
        rng = random.Random()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # First pass: try removing redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                return blocks
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        
        if improved or time.time() >= deadline:
            continue
        
        # Try merging three blocks
        if len(blocks) >= 3 and rng.random() < 0.3:
            bi, bj, bk = rng.sample(range(len(blocks)), 3)
            
            b_i = set(blocks[bi])
            b_j = set(blocks[bj])
            b_k = set(blocks[bk])
            
            # Try to find a k-subset that covers all unique t-subsets from these three blocks
            unique_ts = set()
            for idx in [bi, bj, bk]:
                for ts in itertools.combinations(blocks[idx], t):
                    ts = tuple(sorted(ts))
                    if coverage_count.get(ts, 0) == 1:
                        unique_ts.add(ts)
            
            if not unique_ts:
                # No unique t-subsets, so we can safely merge
                candidate = sorted(b_i | b_j | b_k)
                if len(candidate) <= k:
                    candidate_block = candidate
                else:
                    candidate_block = list(rng.sample(candidate, k))
                
                # Check if this is safe
                safe = True
                for idx in [bi, bj, bk]:
                    for ts in itertools.combinations(blocks[idx], t):
                        ts = tuple(sorted(ts))
                        if coverage_count.get(ts, 0) <= 1 and ts not in set(itertools.combinations(candidate_block, t)):
                            safe = False
                            break
                    if not safe:
                        break
                
                if safe:
                    # Perform the merge
                    for idx in [bi, bj, bk]:
                        for ts in itertools.combinations(blocks[idx], t):
                            ts = tuple(sorted(ts))
                            coverage_count[ts] -= 1
                            if coverage_count[ts] <= 0:
                                del coverage_count[ts]
                    
                    # Remove the blocks
                    for idx in sorted([bi, bj, bk], reverse=True):
                        blocks.pop(idx)
                    
                    # Add the new block
                    blocks.append(candidate_block)
                    for ts in itertools.combinations(candidate_block, t):
                        ts = tuple(sorted(ts))
                        coverage_count[ts] = coverage_count.get(ts, 0) + 1
                    
                    improved = True
                    break
    
    return blocks


def _simulated_annealing(blocks, v, k, t, deadline, rng=None):
    """Simulated annealing: try random mutations (swap, add, remove blocks) and accept
    based on energy (number of uncovered t-subsets + penalty for block count)."""
    if rng is None:
        rng = random.Random()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    
    # Energy function: number of uncovered t-subsets + 0.1 * number of blocks
    def energy(blocks, coverage_count):
        uncovered = 0
        for ts, count in coverage_count.items():
            if count == 0:
                uncovered += 1
        return uncovered + 0.1 * len(blocks)
    
    current_energy = energy(blocks, coverage_count)
    best_energy = current_energy
    best_blocks = [list(b) for b in blocks]
    
    temperature = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline and temperature > 0.1:
        # Random mutation
        mutation = rng.random()
        
        if mutation < 0.5:
            # Swap a random point in a random block
            if not blocks:
                break
            bi = rng.randrange(len(blocks))
            old_x = blocks[bi][rng.randrange(len(blocks[bi]))]
            new_x = rng.randrange(v)
            if new_x not in blocks[bi]:
                blocks[bi][blocks[bi].index(old_x)] = new_x
                blocks[bi].sort()
                
                # Update coverage
                for ts in itertools.combinations([old_x] + [x for x in blocks[bi] if x != old_x], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                for ts in itertools.combinations(blocks[bi], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] = coverage_count.get(ts, 0) + 1
                
                new_energy = energy(blocks, coverage_count)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / temperature):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in blocks]
                else:
                    # Revert
                    blocks[bi][blocks[bi].index(new_x)] = old_x
                    blocks[bi].sort()
                    for ts in itertools.combinations([new_x] + [x for x in blocks[bi] if x != new_x], t):
                        ts = tuple(sorted(ts))
                        coverage_count[ts] -= 1
                        if coverage_count[ts] <= 0:
                            del coverage_count[ts]
                    for ts in itertools.combinations(blocks[bi], t):
                        ts = tuple(sorted(ts))
                        coverage_count[ts] = coverage_count.get(ts, 0) + 1
        elif mutation < 0.7:
            # Remove a random block
            if len(blocks) <= 1:
                break
            bi = rng.randrange(len(blocks))
            removed_block = blocks.pop(bi)
            for ts in itertools.combinations(removed_block, t):
                ts = tuple(sorted(ts))
                coverage_count[ts] -= 1
                if coverage_count[ts] <= 0:
                    del coverage_count[ts]
            
            new_energy = energy(blocks, coverage_count)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = [list(b) for b in blocks]
            else:
                # Revert
                blocks.insert(bi, removed_block)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] = coverage_count.get(ts, 0) + 1
        else:
            # Add a random block
            block = sorted(rng.sample(range(v), k))
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                ts = tuple(sorted(ts))
                coverage_count[ts] = coverage_count.get(ts, 0) + 1
            
            new_energy = energy(blocks, coverage_count)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = [list(b) for b in blocks]
            else:
                # Revert
                blocks.pop()
                for ts in itertools.combinations(block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
        
        temperature *= cooling_rate
    
    return best_blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 110  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # First greedy run
    blocks = _greedy_cover(v, k, t, rng, _global_deadline)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 60 and restarts < 5:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 60)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count (redundancy removal)
    if time.time() < _global_deadline - 20:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 20)

    # Phase 3: Pairwise merge search (more aggressive local search)
    if time.time() < _global_deadline - 10:
        best_blocks = _pairwise_merge_search(best_blocks, v, k, t, _global_deadline - 10, rng)

    # Phase 4: Triple merge search
    if time.time() < _global_deadline - 5:
        best_blocks = _triple_merge_search(best_blocks, v, k, t, _global_deadline - 5, rng)

    # Phase 5: Simulated annealing
    if time.time() < _global_deadline - 2:
        best_blocks = _simulated_annealing(best_blocks, v, k, t, _global_deadline - 2, rng)

    return best_blocks
# EVOLVE-BLOCK-END
