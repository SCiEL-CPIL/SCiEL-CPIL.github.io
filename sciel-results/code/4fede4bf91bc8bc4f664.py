import itertools
import time
import random
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    def ucb_greedy_construct(v, k, t, seed=0, exploration_const=1.4):
        """
        UCB-based block construction.
        Each candidate point is an 'arm' with a reward = marginal coverage gain.
        We track pull counts and cumulative rewards per point.
        UCB = mean_reward + sqrt(2 * exploration_const * ln(total_pulls) / count)
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered and time.time() < deadline - 30:
            # Pick a starting triplet
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            # UCB state: for each point, track total reward and number of pulls
            point_rewards = {p: 0.0 for p in range(v) if p not in block_set}
            point_pulls = {p: 0 for p in range(v) if p not in block_set}
            total_pulls = 0
            
            while len(block) < k and time.time() < deadline - 30:
                # Calculate gain for each candidate point (this is the "reward" if we pull that arm)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                best_point = -1
                best_ucb = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    # Calculate actual gain (reward) for this point
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    
                    # Update UCB state (simulate pulling this arm)
                    point_rewards[p] += gain
                    point_pulls[p] += 1
                    total_pulls += 1
                    
                    # Calculate UCB value
                    if point_pulls[p] == 0:
                        ucb_val = float('inf')
                    else:
                        mean_reward = point_rewards[p] / point_pulls[p]
                        if total_pulls > 0 and point_pulls[p] > 0:
                            exploration_term = math.sqrt(2 * exploration_const * math.log(total_pulls + 1) / point_pulls[p])
                        else:
                            exploration_term = 0
                        ucb_val = mean_reward + exploration_term
                    
                    if ucb_val > best_ucb:
                        best_ucb = ucb_val
                        best_point = p
                
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                continue
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    def greedy_construct(v, k, t, seed=0, bias=0.0):
        """
        Original greedy construction for comparison/fallback.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered and time.time() < deadline - 30:
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k and time.time() < deadline - 30:
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # Phase 1: UCB-based construction with various exploration constants
    best_blocks = None
    best_count = float('inf')
    
    # Try different exploration constants (bandit hyperparameter)
    exploration_consts = [1.0, 1.2, 1.4, 1.6, 2.0]
    for ec in exploration_consts:
        for seed in range(8):
            if time.time() > deadline - 40:
                break
            blocks = ucb_greedy_construct(v, k, t, seed, exploration_const=ec)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        if time.time() > deadline - 40:
            break

    # Phase 2: Original greedy as fallback/comparison
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)

    # Local Search Optimization
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = set()
            for triplet in itertools.combinations(b1, t):
                ids_b1.add(get_id(triplet))
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = set()
                for triplet in itertools.combinations(b2, t):
                    ids_b2.add(get_id(triplet))
                
                critical_points = set()
                
                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    covered_by_b1 = 0
                    covered_by_b2 = 1
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    new_block = list(critical_points)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True

    current_blocks, _ = try_remove_redundant(current_blocks)

    return current_blocks

if __name__ == "__main__":
    pass
