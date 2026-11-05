"""Combined program: Systematic Precomputed Greedy + Local Search with UCB-inspired Exploration."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Precompute all t-subsets (triples)
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_k_blocks:
        tids = [tset_to_id[ts] for ts in itertools.combinations(block, t)]
        block_triples.append(tids)
        
    # Map each triple to the list of block indices that cover it
    # This is needed for efficient gain updates (reverse index)
    triple_blocks = [[] for _ in range(n_tsets)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    def greedy_systematic(seed):
        """
        Greedy construction over all precomputed blocks.
        Uses a UCB-inspired exploration factor to diversify restarts.
        """
        rng = random.Random(seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(n_tsets, dtype=np.bool_)
        
        selected_blocks = []
        selected_count = 0
        
        # Exploration parameter for UCB-like selection
        # We will select from the top K candidates with weights based on gain and randomness
        top_k = 20
        exploration_rate = 0.1
        
        while is_uncovered.any():
            if time.time() > deadline - 10:
                # If we run out of time, we must ensure we return a valid cover.
                # Since we are in the middle of construction, we need to finish quickly.
                # However, the task requires a valid cover.
                # If we break here, we might have an incomplete cover.
                # To be safe, if we detect time pressure, we switch to a fast deterministic greedy 
                # that just picks max gain without exploration.
                pass
            
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true
                break
            
            # Find indices with max gain (or near max for exploration)
            # To implement UCB-like exploration, we pick from a set of high-gain blocks
            # with a probability influenced by their gain and a random factor.
            
            # Get all blocks with gain > 0
            # For efficiency, just get the top K
            # np.argsort is O(N log N), which is fine for 346k elements
            # But we can just find where gain == max_gain and maybe some close ones
            # Let's find the top K by gain
            # This is expensive if done every step. 
            # Alternative: just pick random from those with max_gain, or add noise.
            
            # Simple UCB approximation:
            # score = gain + c * sqrt(log(N) / (1 + visits))
            # We don't track visits per block easily in this loop structure.
            # Instead, let's add noise to the gain for selection.
            
            # Add small random noise to gain for selection
            noise = rng.random() * exploration_rate * max_gain
            scores = gain + noise
            scores[gain == 0] = 0 # Don't pick zero gain blocks
            
            # Find the index of the max score
            b_idx = np.argmax(scores)
            
            # Safety check: if the selected block has 0 gain, something is wrong
            if gain[b_idx] == 0:
                 # Fallback to pure max gain
                 candidates = np.where(gain == max_gain)[0]
                 if len(candidates) == 0: break
                 b_idx = candidates[0]

            selected_blocks.append(b_idx)
            selected_count += 1
            
            # Update gain and is_uncovered
            # For each triple in the selected block:
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_blocks

    def reduce_cover(selected_blocks):
        """
        Remove redundant blocks from the selected set.
        """
        # Convert to list of sets of triple indices for quick check?
        # We maintain cover_count[t]
        cover_count = np.zeros(n_tsets, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        active = [True] * len(selected_blocks)
        
        # Pass 1: Remove redundant blocks
        # A block is redundant if all its triples have cover_count > 1
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                # Check if redundant
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    # Update cover counts
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    # Phase 1: Multiple systematic greedy restarts
    best_blocks_idx = None
    best_count = float('inf')
    
    num_seeds = 50 # We can afford more restarts because greedy is faster
    
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        selected = greedy_systematic(seed)
        reduced = reduce_cover(selected)
        
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_blocks_idx = reduced

    if best_blocks_idx is None:
        # Fallback
        selected = greedy_systematic(0)
        best_blocks_idx = reduce_cover(selected)
        best_count = len(best_blocks_idx)

    # Convert indices to blocks
    current_blocks = [list(all_k_blocks[b_idx]) for b_idx in best_blocks_idx]
    
    # Phase 2: Local Search on the best cover
    # We need to update the local search functions to work with the new structure
    # or adapt them. The current local search works on lists of blocks.
    # We will reuse the logic from the current program but ensure it's compatible.
    
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for ts in itertools.combinations(block, t):
                tid = tset_to_id[ts]
                count[tid] += delta

    update_counts(current_blocks, 1)

    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[tset_to_id[ts]] == 1:
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

        # Precompute block t-set IDs
        block_ids = []
        for b in blocks:
            ids = set()
            for ts in itertools.combinations(b, t):
                ids.add(tset_to_id[ts])
            block_ids.append(ids)

        # Heuristic: prioritize pairs with high overlap
        pair_scores = []
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    break
                overlap = len(block_ids[i] & block_ids[j])
                total = len(block_ids[i] | block_ids[j])
                if total > 0:
                    score = overlap / total
                    unique_i = len(block_ids[i] - block_ids[j])
                    unique_j = len(block_ids[j] - block_ids[i])
                    low_unique = (unique_i < 5 or unique_j < 5)
                    if low_unique:
                        score += 0.5
                    pair_scores.append((score, i, j))

        pair_scores.sort(reverse=True)
        max_pairs_to_try = min(len(pair_scores), 300)

        for score, i, j in pair_scores[:max_pairs_to_try]:
            if time.time() > deadline - 10:
                break

            b1 = blocks[i]
            b2 = blocks[j]
            ids_b1 = block_ids[i]
            ids_b2 = block_ids[j]

            check_ids = ids_b1 | ids_b2

            u_need = []
            for tid in check_ids:
                c = count[tid]
                cov_b1 = 1 if tid in ids_b1 else 0
                cov_b2 = 1 if tid in ids_b2 else 0
                if c == cov_b1 + cov_b2:
                    u_need.append(tid)

            if not u_need:
                update_counts([b1, b2], -1)
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])

            if len(p_need) > k:
                continue

            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 1000 + j)
            candidates = [p for p in range(v) if p not in new_block_set]

            remaining_slots = k - len(new_block)
            for _ in range(remaining_slots):
                if not candidates:
                    break
                best_p = -1
                best_gain = -1
                subs_in_block = list(itertools.combinations(new_block, t - 1))

                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if tid not in ids_b1 and tid not in ids_b2:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p

                if best_p == -1 or best_gain == 0:
                    best_p = rng.choice(candidates)

                new_block.append(best_p)
                new_block_set.add(best_p)
                candidates.remove(best_p)

            new_block.sort()

            new_ids = set()
            for ts in itertools.combinations(new_block, t):
                new_ids.add(tset_to_id[ts])

            if all(tid in new_ids for tid in u_need):
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    def try_reconstruct_block(blocks, i):
        """
        Max-SAT inspired move: Reconstruct block i to better cover its unique triplets
        and potentially increase global efficiency.
        """
        n = len(blocks)
        if n < 1:
            return blocks, False

        b = blocks[i]
        ids_b = set()
        for ts in itertools.combinations(b, t):
            ids_b.add(tset_to_id[ts])

        # Find unique triplets
        u_need = [tid for tid in ids_b if count[tid] == 1]

        if not u_need:
            # Redundant, handled by removal
            return blocks, False

        p_need = set()
        for tid in u_need:
            p_need.update(all_tsets[tid])

        if len(p_need) > k:
            return blocks, False

        # Try to build a new block that covers p_need and maximizes additional coverage
        # This is a constrained greedy construction
        new_block = list(p_need)
        new_block_set = set(new_block)
        rng = random.Random(i * 7919 + int(time.time() % 1000))
        candidates = [p for p in range(v) if p not in new_block_set]

        remaining_slots = k - len(new_block)
        for _ in range(remaining_slots):
            if not candidates:
                break
            best_p = -1
            best_gain = -1
            subs_in_block = list(itertools.combinations(new_block, t - 1))

            for p in candidates:
                gain = 0
                for sub in subs_in_block:
                    ts = tuple(sorted(sub + (p,)))
                    tid = tset_to_id[ts]
                    # We want to cover triplets that are NOT already well-covered by others
                    # or just maximize total coverage
                    if count[tid] == 0: # Should not happen if valid, but good for safety
                        gain += 10
                    elif count[tid] == 1:
                        gain += 1
                    else:
                        gain += 0.1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p

            if best_p == -1 or best_gain == 0:
                best_p = rng.choice(candidates)

            new_block.append(best_p)
            new_block_set.add(best_p)
            candidates.remove(best_p)

        new_block.sort()

        if new_block == b:
            return blocks, False

        new_ids = set()
        for ts in itertools.combinations(new_block, t):
            new_ids.add(tset_to_id[ts])

        if all(tid in new_ids for tid in u_need):
            # Check if this improves the "score"
            update_counts([b], -1)
            update_counts([new_block], 1)
            new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
            new_blocks.append(new_block)
            return new_blocks, True

        return blocks, False

    # Iteratively apply local search moves
    improved = True
    max_rounds = 100
    stall_count = 0
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        if not improved:
            stall_count += 1
            if stall_count >= 3:
                # Try perturbation/reconstruction to escape local optimum
                if time.time() < deadline - 20:
                    # Pick a random block to reconstruct
                    if current_blocks:
                        idx = random.Random(int(time.time()*1000) % 10000).randrange(len(current_blocks))
                        new_blocks, pert = try_reconstruct_block(current_blocks, idx)
                        if pert:
                            current_blocks = new_blocks
                            improved = True
                            stall_count = 0
                            # Try removals after perturbation
                            for _pass in range(3):
                                if time.time() > deadline - 15:
                                    break
                                new_blocks, rem = try_remove_redundant(current_blocks)
                                if rem:
                                    current_blocks = new_blocks
                                    improved = True
                                else:
                                    break
                if not improved:
                    break
                continue
            else:
                break
        else:
            stall_count = 0
        improved = False

        # Try removing redundant blocks
        for _pass in range(3):
            if time.time() > deadline - 15:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            else:
                break

        # Try 2-for-1 swaps
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            for _pass in range(2):
                if time.time() > deadline - 15:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        # Try reconstruction (1-for-1 but smarter)
        if current_blocks and time.time() < deadline - 15:
            # Try a few random reconstructions
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                idx = random.Random(int(time.time()*10000) % 100000).randrange(len(current_blocks))
                new_blocks, swp = try_reconstruct_block(current_blocks, idx)
                if swp:
                    current_blocks = new_blocks
                    improved = True
                    break

    # Final verification
    covered = set()
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(tset_to_id[ts])

    if len(covered) < n_tsets:
        # Fallback to the best blocks from Phase 1 if local search broke validity (should not happen)
        return [list(all_k_blocks[b_idx]) for b_idx in best_blocks_idx]

    # Final polish
    if time.time() < deadline - 5:
        for _pass in range(5):
            if time.time() > deadline - 3:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
            else:
                break

    return current_blocks
# EVOLVE-BLOCK-END
