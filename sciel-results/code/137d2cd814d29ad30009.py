import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute all t-subsets and k-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # Map each k-subset to the indices of the t-subsets it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = [t_sub_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(t_idxs)

    # Map each t-subset to the indices of the k-subsets that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed, strategy='standard'):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=np.uint8)
        blocks_idx = []
        used_blocks = set()
        
        # Precompute candidate lists for each t-subset for faster access
        # t_sub_to_k_idxs is already a list of lists
        
        while True:
            if time.time() >= deadline - 10.0:
                break
            
            if not uncovered.any():
                break

            if strategy == 'most_constrained':
                # Find uncovered t-subset with fewest available candidates
                best_s_idx = -1
                min_candidates = float('inf')
                # Sample a few to find a constrained one
                for _ in range(20):
                    s_idx = rng.randint(0, n_t - 1)
                    if not uncovered[s_idx]:
                        continue
                    avail = [b for b in t_sub_to_k_idxs[s_idx] if b not in used_blocks]
                    if len(avail) < min_candidates:
                        min_candidates = len(avail)
                        best_s_idx = s_idx
                    if min_candidates <= 1:
                        break
                if best_s_idx == -1:
                    # Fallback to random uncovered
                    while True:
                        s_idx = rng.randint(0, n_t - 1)
                        if uncovered[s_idx]:
                            break
                else:
                    s_idx = best_s_idx
            elif strategy == 'random':
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if uncovered[s_idx]:
                        break
            else:
                s_idx = int(np.argmax(uncovered))
                if uncovered[s_idx] == 0:
                    break

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1

            # Vectorized gain calculation
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                t_idxs = k_sub_t_idxs[b_idx]
                gain = int(uncovered[t_idxs].sum())
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            uncovered[k_sub_t_idxs[best_block_idx]] = 0

        return blocks_idx

    def reduce_blocks(blocks_idx, max_passes=10):
        if not blocks_idx:
            return blocks_idx

        blocks_idx = list(blocks_idx)
        
        for _ in range(max_passes):
            if time.time() >= deadline - 5.0:
                break
            
            # Build cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                np.add.at(cover_count, k_sub_t_idxs[b_idx], 1)

            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if not np.any(counts_for_block == 1):
                    # This block is redundant
                    changed = True
                    np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                else:
                    new_blocks.append(b_idx)
            
            blocks_idx = new_blocks
            if not changed:
                break

        return blocks_idx

    def local_search(blocks_idx, max_iter=1000):
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        n_blocks = len(blocks_idx)
        
        # Initial cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            np.add.at(cover_count, k_sub_t_idxs[b_idx], 1)

        for _ in range(max_iter):
            if time.time() >= deadline - 5.0:
                break
            
            improved = False
            
            # 1. Try deletions
            for i in range(len(blocks_idx)):
                b_idx = blocks_idx[i]
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if not np.any(counts_for_block == 1):
                    # Remove block
                    np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                    blocks_idx.pop(i)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try swaps
            # Pick a random block to try swapping
            i = random.randint(0, len(blocks_idx) - 1)
            b_idx = blocks_idx[i]
            
            # Find essential t-subsets for this block
            essential = [t_idx for t_idx in k_sub_t_idxs[b_idx] if cover_count[t_idx] == 1]
            
            if not essential:
                # This block is not essential for anything, but deletion failed? 
                # This shouldn't happen if deletion check is correct. 
                # Maybe it became essential after other changes? 
                # If essential is empty, we can just delete it? 
                # Wait, if essential is empty, it means NO t-subset in this block has count 1.
                # So it should have been deleted in step 1.
                # Why wasn't it? Maybe the loop in step 1 didn't reach it?
                # Ah, step 1 breaks on first deletion. If no deletion happened, we are here.
                # If essential is empty, we can delete it.
                np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                blocks_idx.pop(i)
                improved = True
                continue

            # Find candidates that cover all essential t-subsets
            # Intersection of candidate lists
            candidates = set(t_sub_to_k_idxs[essential[0]])
            for t_idx in essential[1:]:
                candidates &= set(t_sub_to_k_idxs[t_idx])
            
            # Remove current block and already used blocks
            candidates.discard(b_idx)
            candidates -= set(blocks_idx)
            
            if not candidates:
                continue

            # Pick a candidate that maximizes new coverage
            best_replace = -1
            best_gain = -1
            
            # Limit candidates for speed
            if len(candidates) > 50:
                candidates = set(random.sample(list(candidates), 50))
            
            essential_set = set(essential)
            
            for cand_idx in candidates:
                cand_t_idxs = k_sub_t_idxs[cand_idx]
                # Check if it covers all essential (should be true by construction)
                # But let's be safe
                if not essential_set.issubset(set(cand_t_idxs)):
                    continue
                
                # Gain: number of t-subsets covered by cand_idx that are currently uncovered
                # Or rather, number of t-subsets where count will increase from 0 to 1?
                # Actually, we are replacing b_idx with cand_idx.
                # The net change in cover_count is:
                # -1 for t-subsets in b_idx \ cand_idx
                # +1 for t-subsets in cand_idx \ b_idx
                # We want to ensure we don't uncover anything (handled by essential check)
                # We want to maximize the number of t-subsets that become covered (0->1)
                # Or minimize the number that become uncovered? (Should be 0)
                
                # Simple heuristic: maximize overlap with uncovered sets?
                # Let's just pick the one with the most unique coverage among essential?
                # Actually, any valid swap maintains validity.
                # To help future deletions, we want to increase cover counts of other blocks?
                # Or decrease redundancy?
                
                # Let's pick the one that covers the most t-subsets that are currently count 1 
                # but NOT in this block? No.
                
                # Let's just pick the first one or random.
                # Better: pick the one that minimizes the "essential" status of other blocks?
                
                # For now, random selection from valid candidates.
                best_replace = cand_idx
                break # Just pick one to keep it fast? 
                # Or pick the one with max gain in terms of "newly covered" if we were adding?
                # Since it's a swap, the total number of covered t-subsets doesn't change 
                # (it's always all of them if valid).
                # So the quality depends on the distribution of counts.
                # A good swap might allow more deletions later.
                
                # Let's try to maximize the minimum count of the essential t-subsets?
                # No, they will be count 2 after swap (1 from other, 1 from new).
                
                # Let's just take the first valid candidate.
            
            if best_replace != -1:
                # Perform swap
                np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                np.add.at(cover_count, k_sub_t_idxs[best_replace], 1)
                blocks_idx[i] = best_replace
                improved = True
        
            if not improved:
                # No improvement in this iteration, try a few more random swaps?
                # Or break?
                # Let's break to avoid infinite loop of no-ops
                break

        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    blocks_idx = greedy_cover(0, 'standard')
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Most Constrained Greedy
    blocks_idx = greedy_cover(1, 'most_constrained')
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx and len(blocks_idx) < best_len:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 3. Randomized Restarts
    for seed in range(2, 200):
        if time.time() >= deadline - 30.0:
            break
        strategy = 'most_constrained' if seed % 2 == 0 else 'random'
        blocks_idx = greedy_cover(seed, strategy)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 4. Local Search on Best
    if best_blocks and time.time() < deadline - 20.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 5. Second Local Search Pass
    if best_blocks and time.time() < deadline - 10.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 6. Final Reduction
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
