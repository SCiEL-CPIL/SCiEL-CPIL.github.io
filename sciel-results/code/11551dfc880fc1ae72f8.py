"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute the list of t-subset indices it belongs to
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(idx)

    # Precompute for each pair of points, the t-subsets they form with a third point
    # This speeds up greedy by avoiding repeated tuple creation
    pair_tsubs = {}
    for i in range(v):
        for j in range(i + 1, v):
            pair_tsubs[(i, j)] = []
            for x in range(v):
                if x != i and x != j:
                    ts = tuple(sorted((i, j, x)))
                    idx = tsub_to_idx[ts]
                    pair_tsubs[(i, j)].append(idx)

    # Greedy covering with reverse indexing - optimized version
    def greedy_cover(seed=None):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        uncovered = np.ones(n_tsubs, dtype=np.int32)  # 1 if uncovered
        blocks = []

        while uncovered.any():
            # Pick an uncovered t-subset (randomize among uncovered for diversity)
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[local_rng.randint(0, len(uncovered_indices) - 1)]
            start_tsub = all_tsubs[start_idx]

            block = list(start_tsub)
            block_set = set(block)

            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # New t-subsets covered by adding x
                    new_tsubs = 0
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_to_idx[ts]
                        if uncovered[idx]:
                            new_tsubs += 1
                    if new_tsubs > best_gain:
                        best_gain = new_tsubs
                        best_point = x

                if best_point == -1:
                    # Pick any remaining point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = remaining[local_rng.randint(0, len(remaining) - 1)]
                    else:
                        break

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(tuple(block))

            # Update uncovered
            for combo in itertools.combinations(block, t):
                idx = tsub_to_idx[combo]
                uncovered[idx] = 0

        return [list(b) for b in blocks]

    # Local search: try to remove blocks - optimized
    def local_search_remove(blocks, seed=None):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        blocks = [list(b) for b in blocks]
        improved = True
        while improved:
            improved = False
            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for block in blocks:
                for combo in itertools.combinations(block, t):
                    idx = tsub_to_idx[combo]
                    cover_count[idx] += 1

            # Try removing each block if it doesn't break coverage
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                # Check if all t-subsets in this block have cover_count > 1
                can_remove = True
                for combo in itertools.combinations(block, t):
                    idx = tsub_to_idx[combo]
                    if cover_count[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for combo in itertools.combinations(block, t):
                        idx = tsub_to_idx[combo]
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    improved = True

        return blocks

    # Max-SAT based local search: treat block selection as weighted Max-SAT
    def maxsat_local_search(blocks, seed=None, max_iters=200000):
        """
        Formulate as Max-SAT:
        - Each block is a boolean variable (selected or not)
        - Objective: minimize number of selected blocks (weight 1 per block)
        - Constraints: each t-subset must be covered by at least one selected block
        
        Use stochastic local search (WalkSAT-style) to minimize cost while maintaining coverage.
        """
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        
        # Build data structures
        # For each t-subset, list of blocks that cover it
        tsub_covering_blocks = [[] for _ in range(n_tsubs)]
        block_tsub_indices = []
        
        for bi, block in enumerate(blocks):
            idxs = []
            for combo in itertools.combinations(block, t):
                idx = tsub_to_idx[combo]
                idxs.append(idx)
                tsub_covering_blocks[idx].append(bi)
            block_tsub_indices.append(idxs)
        
        # Track which blocks are selected (start with all selected)
        selected = np.ones(n_blocks, dtype=bool)
        
        # Coverage count for each t-subset
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for idx in range(n_tsubs):
            cover_count[idx] = len(tsub_covering_blocks[idx])
        
        # Cost = number of selected blocks
        current_cost = n_blocks
        best_cost = n_blocks
        best_selected = selected.copy()
        
        # Weight for penalty on uncovered t-subsets (should be large)
        penalty_weight = 1000
        
        def compute_cost():
            """Compute total cost: number of selected blocks + penalty * uncovered_tsubs"""
            cost = np.sum(selected)
            uncovered = np.sum(cover_count == 0)
            return cost + penalty_weight * uncovered
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Try to deselect a random selected block
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0:
                break
                
            bi = selected_indices[local_rng.randint(0, len(selected_indices) - 1)]
            
            # Check if deselecting this block would leave any t-subset uncovered
            can_deselect = True
            for tidx in block_tsub_indices[bi]:
                if cover_count[tidx] == 1:
                    # This block is the only one covering this t-subset
                    can_deselect = False
                    break
            
            if can_deselect:
                # Deselect the block
                selected[bi] = False
                for tidx in block_tsub_indices[bi]:
                    cover_count[tidx] -= 1
                current_cost -= 1
                
                new_cost = compute_cost()
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_selected = selected.copy()
        
        # Extract best solution
        result_blocks = [blocks[i] for i in range(n_blocks) if best_selected[i]]
        return result_blocks

    # Simulated annealing: 1-for-1 exchanges
    def simulated_annealing(blocks, seed=None, max_iters=50000):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)

        # Build coverage count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            for combo in itertools.combinations(block, t):
                idx = tsub_to_idx[combo]
                cover_count[idx] += 1

        # Precompute for each block the list of t-subset indices it covers
        block_tsub_indices = []
        for block in blocks:
            idxs = []
            for combo in itertools.combinations(block, t):
                idxs.append(tsub_to_idx[combo])
            block_tsub_indices.append(idxs)

        current_cost = n_blocks
        best_blocks = [b[:] for b in blocks]
        best_cost = n_blocks

        temperature = 1.0
        cooling = 0.9995

        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            # Try: replace a random point in a random block with a different point
            i = local_rng.randint(0, n_blocks - 1)
            block = blocks[i]
            pos = local_rng.randint(0, k - 1)
            old_point = block[pos]

            # Choose a new point not already in the block
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                continue
            new_point = candidates[local_rng.randint(0, len(candidates) - 1)]

            # Compute change in coverage
            # Remove old point's contribution, add new point's contribution
            delta_cover = {}  # tsub_idx -> change in cover count

            # T-subsets involving old_point that are in this block
            for combo in itertools.combinations(block, t):
                if old_point in combo:
                    idx = tsub_to_idx[combo]
                    delta_cover[idx] = delta_cover.get(idx, 0) - 1

            # T-subsets involving new_point with the other t-1 points in the block
            other_points = [x for x in block if x != old_point]
            for combo in itertools.combinations(other_points, t - 1):
                ts = tuple(sorted(combo + (new_point,)))
                idx = tsub_to_idx[ts]
                delta_cover[idx] = delta_cover.get(idx, 0) + 1

            # Check validity: no t-subset should go to 0
            valid = True
            for idx, delta in delta_cover.items():
                if cover_count[idx] + delta < 1:
                    valid = False
                    break

            if valid:
                # Accept the move
                block[pos] = new_point
                block.sort()
                for idx, delta in delta_cover.items():
                    cover_count[idx] += delta
                # Update block_tsub_indices
                new_idxs = []
                for combo in itertools.combinations(block, t):
                    new_idxs.append(tsub_to_idx[combo])
                block_tsub_indices[i] = new_idxs

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [b[:] for b in blocks]

            temperature *= cooling

        return best_blocks

    # Run multiple greedy variants with different strategies
    def greedy_cover_varied(seed=None, strategy='random'):
        """Greedy with different strategies for diversity."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks = []

        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break

            if strategy == 'random':
                start_idx = uncovered_indices[local_rng.randint(0, len(uncovered_indices) - 1)]
            elif strategy == 'first':
                start_idx = uncovered_indices[0]
            elif strategy == 'most_constrained':
                # Pick t-subset that appears in fewest potential blocks
                # Approximate by picking randomly for now
                start_idx = uncovered_indices[local_rng.randint(0, len(uncovered_indices) - 1)]
            else:
                start_idx = uncovered_indices[local_rng.randint(0, len(uncovered_indices) - 1)]

            start_tsub = all_tsubs[start_idx]
            block = list(start_tsub)
            block_set = set(block)

            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    new_tsubs = 0
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_to_idx[ts]
                        if uncovered[idx]:
                            new_tsubs += 1
                    if new_tsubs > best_gain:
                        best_gain = new_tsubs
                        best_point = x

                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = remaining[local_rng.randint(0, len(remaining) - 1)]
                    else:
                        break

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(tuple(block))

            for combo in itertools.combinations(block, t):
                idx = tsub_to_idx[combo]
                uncovered[idx] = 0

        return [list(b) for b in blocks]

    # Run greedy with multiple strategies
    initial_blocks = greedy_cover_varied(seed=42, strategy='random')
    initial_count = len(initial_blocks)

    # Local search to remove redundant blocks
    blocks = local_search_remove(initial_blocks, seed=43)
    best_count = len(blocks)
    best_result = [b[:] for b in blocks]

    # Max-SAT local search
    if time.time() < deadline - 15.0:
        ms_blocks = maxsat_local_search(blocks, seed=44, max_iters=100000)
        if len(ms_blocks) < best_count:
            best_count = len(ms_blocks)
            best_result = [b[:] for b in ms_blocks]

    # Simulated annealing
    if time.time() < deadline - 10.0:
        sa_blocks = simulated_annealing(blocks, seed=45, max_iters=100000)
        if len(sa_blocks) < best_count:
            best_count = len(sa_blocks)
            best_result = [b[:] for b in sa_blocks]

    # Try more greedy runs with different seeds and strategies if time permits
    seeds_strategies = [
        (1, 'random'), (2, 'first'), (3, 'random'), (5, 'random'),
        (7, 'first'), (11, 'random'), (13, 'random'), (17, 'first'),
        (19, 'random'), (23, 'random'), (29, 'first'), (31, 'random')
    ]
    
    for seed, strategy in seeds_strategies:
        if time.time() > deadline - 15.0:
            break
        g_blocks = greedy_cover_varied(seed=seed, strategy=strategy)
        g_blocks = local_search_remove(g_blocks, seed=seed + 100)
        if len(g_blocks) < best_count:
            best_count = len(g_blocks)
            best_result = [b[:] for b in g_blocks]
            if time.time() < deadline - 10.0:
                ms_blocks = maxsat_local_search(g_blocks, seed=seed + 200, max_iters=50000)
                if len(ms_blocks) < best_count:
                    best_count = len(ms_blocks)
                    best_result = [b[:] for b in ms_blocks]
                if time.time() < deadline - 5.0:
                    sa_blocks = simulated_annealing(g_blocks, seed=seed + 300, max_iters=30000)
                    if len(sa_blocks) < best_count:
                        best_count = len(sa_blocks)
                        best_result = [b[:] for b in sa_blocks]

    return best_result
# EVOLVE-BLOCK-END
