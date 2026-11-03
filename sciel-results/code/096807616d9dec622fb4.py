"""Improved covering algorithm using vectorized greedy with local search."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks and their coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build coverage: for each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_idx[ts])
        block_tsubs.append(tsubs)
    
    # Inverse: for each t-subset, list of blocks covering it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(seed=0, alpha=1.0):
        """Greedy with soft-max selection for diversity.
        
        alpha controls randomness: alpha=1 gives pure greedy,
        higher alpha gives more random selection among high-gain blocks.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # gain[i] = number of uncovered t-subsets covered by block i
        gain = np.array([np.sum(uncovered[ts]) for ts in block_tsubs], dtype=np.float64)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Find max gain
            max_gain = gain.max()
            if max_gain <= 0:
                break
            
            # Soft-max sampling: pick block with probability proportional to exp(gain/alpha)
            # among blocks with gain > 0
            active_mask = gain > 0
            if not np.any(active_mask):
                break
            
            # For efficiency, only consider blocks with gain >= max_gain - threshold
            threshold = max_gain * 0.5  # consider blocks with at least 50% of max gain
            candidates = np.where(active_mask & (gain >= threshold))[0]
            
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                # Soft-max selection
                gains = gain[candidates]
                # Normalize to avoid overflow
                gains_shifted = gains - gains.max()
                weights = np.exp(gains_shifted / alpha)
                weights /= weights.sum()
                chosen = candidates[rng.choice(len(candidates), p=weights)]
            
            blocks_used.append(all_blocks[chosen])
            
            # Update uncovered and gain
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in tsub_blocks[ts_idx]:
                        gain[bi] -= 1
            
            # Mark chosen block as unusable
            gain[chosen] = -1
        
        return blocks_used
    
    def local_search(blocks, seed=0):
        """Improve by removing redundant blocks and trying swaps."""
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        
        # Build cover count and block-tsub indices
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_idx[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        # Remove redundant blocks
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        reduced = [blocks[i] for i in range(len(blocks)) if kept[i]]
        
        # Try 1-for-2 swaps: replace two blocks with one
        n_reduced = len(reduced)
        if n_reduced > 2 and time.time() < deadline - 5:
            # Rebuild indices for reduced
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_indices = []
            for b in reduced:
                tsubs = []
                for ts in itertools.combinations(b, t):
                    idx = tsub_idx[ts]
                    tsubs.append(idx)
                    cover_count[idx] += 1
                block_tsub_indices.append(tsubs)
            
            # Try random pairs
            max_trials = min(200, n_reduced * (n_reduced - 1) // 2)
            tried = set()
            for _ in range(max_trials):
                if time.time() > deadline:
                    break
                i, j = rng.sample(range(n_reduced), 2)
                key = min(i, j) * n_reduced + max(i, j)
                if key in tried:
                    continue
                tried.add(key)
                
                # Get t-subsets that would become uncovered if we remove i and j
                # A t-subset is safe if cover_count[idx] >= 3 or (cover_count[idx] == 2 and covered by a block other than i,j)
                # Actually, we need: after removing i and j, every t-subset must still be covered.
                # So for each t-subset in block_i ∪ block_j, check if it's covered by some other block.
                
                # Build set of t-subsets in blocks i and j
                tsubs_ij = set(block_tsub_indices[i]) | set(block_tsub_indices[j])
                
                # Check if there's a single block that covers all t-subsets in tsubs_ij
                # AND that for every t-subset NOT in tsubs_ij but covered by i or j, it's still covered
                # Actually, simpler: we need a block B such that:
                # 1. B covers all t-subsets in tsubs_ij
                # 2. For every t-subset s covered by i or j but NOT by B, cover_count[s] >= 3
                #    (so removing i and j still leaves s covered)
                
                # This is expensive. Let's try a simpler heuristic:
                # Find blocks that cover many of tsubs_ij
                # For each candidate block, check coverage
                
                # Get all t-subsets covered by i or j
                all_tsubs_ij = set(block_tsub_indices[i] + block_tsub_indices[j])
                
                # For each block, count how many of all_tsubs_ij it covers
                # This is expensive, so limit candidates
                # Only consider blocks that overlap significantly with i or j
                
                # Alternative: just try all blocks (n_blocks is ~50388, might be too slow)
                # Instead, sample a subset
                n_candidates = min(5000, n_blocks)
                candidate_blocks = rng.sample(range(n_blocks), n_candidates)
                
                best_candidate = -1
                best_score = -1
                
                for cb in candidate_blocks:
                    # Skip if cb is i or j
                    if all_blocks[cb] == reduced[i] or all_blocks[cb] == reduced[j]:
                        continue
                    
                    # Count how many t-subsets in all_tsubs_ij are covered by cb
                    # And check that uncovered t-subsets (those with cover_count == 2 and covered by i or j)
                    # are still covered
                    
                    covered_by_cb = set(block_tsubs[cb])
                    # t-subsets in all_tsubs_ij covered by cb
                    covered_count = len(all_tsubs_ij & covered_by_cb)
                    
                    # t-subsets in all_tsubs_ij NOT covered by cb
                    not_covered = all_tsubs_ij - covered_by_cb
                    
                    # Check if all not_covered t-subsets have cover_count >= 3
                    # (meaning they're covered by at least 3 blocks, so removing i and j still leaves them covered)
                    ok = True
                    for ts_idx in not_covered:
                        if cover_count[ts_idx] < 3:
                            ok = False
                            break
                    
                    if ok:
                        score = covered_count
                        if score > best_score:
                            best_score = score
                            best_candidate = cb
                
                if best_candidate >= 0 and best_score > 0:
                    # Replace i and j with best_candidate
                    new_block = all_blocks[best_candidate]
                    # Remove i and j (higher index first to keep indices valid)
                    if i > j:
                        i, j = j, i
                    reduced.pop(i)
                    reduced.pop(j)
                    reduced.append(new_block)
                    n_reduced = len(reduced)
                    
                    # Rebuild cover_count and block_tsub_indices
                    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
                    block_tsub_indices = []
                    for b in reduced:
                        tsubs = []
                        for ts in itertools.combinations(b, t):
                            idx = tsub_idx[ts]
                            tsubs.append(idx)
                            cover_count[idx] += 1
                        block_tsub_indices.append(tsubs)
                    
                    # After swap, remove any new redundant blocks
                    changed = True
                    while changed:
                        changed = False
                        for idx in range(len(reduced)):
                            redundant = True
                            for ts_idx in block_tsub_indices[idx]:
                                if cover_count[ts_idx] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                for ts_idx in block_tsub_indices[idx]:
                                    cover_count[ts_idx] -= 1
                                reduced.pop(idx)
                                block_tsub_indices.pop(idx)
                                n_reduced = len(reduced)
                                changed = True
                                break
        
        return reduced
    
    # Run greedy with multiple seeds and different alpha values
    alphas = [1.0, 2.0, 5.0, 10.0]
    num_trials_per_alpha = 10
    
    for alpha in alphas:
        for seed in range(num_trials_per_alpha):
            if time.time() > deadline:
                break
            
            blocks = greedy_cover(seed + int(alpha * 100), alpha)
            if not blocks:
                continue
            
            blocks = local_search(blocks, seed)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
            
            # Early termination if we hit a very good result
            if best_len <= 140:
                break
        if best_len <= 140:
            break
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_cover(0, 1.0)
        best_blocks = local_search(best_blocks, 0)
    
    return best_blocks
