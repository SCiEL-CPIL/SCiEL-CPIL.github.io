import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all t-subsets and map to indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    # Precompute all k-subsets
    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    
    # Number of t-subsets in a k-set
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
        
    t_count = nCr(k, t)
    
    # Precompute for each k-subset, the indices of the t-subsets it contains
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    def greedy_randomized(seed, top_k=10):
        """
        Randomized greedy set cover.
        Picks a block from the top 'top_k' gainers randomly.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=bool)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gains: number of uncovered triples covered by each block
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # If max gain is 0, we are stuck or done. 
                if not np.all(covered):
                    return None
                else:
                    break
            
            # Find indices of blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > top_k:
                # Sample from top candidates to diversify
                # We pick a random subset of the max gainers
                pass
            
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Add block
            selected_blocks.append(best_idx)
            
            # Update covered
            covered[block_t_indices[best_idx]] = True
            
            if np.all(covered):
                break

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset in it is covered by at least one other block.
        """
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        
        # We iterate until no more blocks can be removed
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Count coverage for each t-subset
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_blocks = []
            removed_any = False
            
            for b_idx in current_blocks:
                # Check if this block is redundant
                subs_in_block = block_t_indices[b_idx]
                
                # If all these t-subsets have coverage_count > 1, the block is redundant
                # Note: coverage_count includes this block itself, so we need > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    # Redundant: do not add to new_blocks
                    # Decrement coverage for these subsets
                    coverage_count[subs_in_block] -= 1
                    removed_any = True
                else:
                    # Keep it
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            if not removed_any:
                break
                
        return current_blocks

    def greedy_packing(seed, lambda_max=1):
        """
        Greedy packing: select blocks such that no t-subset is covered more than lambda_max times.
        This creates a sparse structure. We then fill in the gaps with standard greedy.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=np.int32)
        selected_blocks = []
        
        # Phase 1: Packing
        # We want to add blocks that cover many *uncovered* (or under-covered) triples
        # but strictly enforce the lambda_max constraint.
        # Actually, for packing, we usually want to maximize the number of blocks subject to constraints.
        # But here we want a *covering*. So "Packing with Conflict Resolution" means:
        # Build a set of blocks that are "non-conflicting" (disjoint in terms of t-subsets? No, that's too strong).
        # The prompt suggests: "select the maximum number of 7-subsets such that no 3-subset is covered more than lambda_max times".
        # If we build such a set, it might not cover everything. We then use standard greedy to cover the rest.
        # The benefit is that the initial set is diverse and sparse.
        
        # Let's try a different interpretation:
        # Standard greedy tends to pick blocks that overlap heavily.
        # Let's try to pick blocks that cover *rare* triples.
        
        # Actually, let's stick to the standard greedy but with a "diversity" penalty.
        # Or, let's use the "Conflict Resolution" idea:
        # 1. Start with an empty set.
        # 2. Repeatedly pick a block that covers the most *uncovered* triples.
        # 3. If there are ties, pick one that minimizes the overlap with already selected blocks.
        
        # This is essentially the same as greedy_randomized but with tie-breaking.
        # Let's implement a version that tracks coverage counts and penalizes high-count triples.
        
        # For now, let's just use greedy_randomized. The "Packing" idea might be too complex to implement safely in the remaining time.
        # Let's focus on improving the reduction and local search.
        
        # Actually, let's implement a simple "conflict-aware" greedy:
        # Score = gain - alpha * overlap_with_existing
        # This encourages picking blocks that are different from previous ones.
        
        alpha = 0.1
        existing_cover_count = np.zeros(n_ts, dtype=np.int32)
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Gain: number of *uncovered* triples
            uncovered_mask = existing_cover_count == 0
            gains = (uncovered_mask[block_t_indices]).sum(axis=1)
            
            # Overlap: number of triples already covered (by any block)
            overlaps = (existing_cover_count[block_t_indices] > 0).sum(axis=1)
            
            scores = gains - alpha * overlaps
            
            max_score = np.max(scores)
            if max_score < 0:
                # No good block found? This shouldn't happen if we have uncovered triples.
                # If gains is 0 for all, we are stuck.
                if not np.any(gains > 0):
                    if np.all(existing_cover_count > 0):
                        break
                    else:
                        return None
                else:
                    # Fallback to pure gain
                    max_gain = np.max(gains)
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = candidates[rng.randint(len(candidates))]
                    selected_blocks.append(best_idx)
                    existing_cover_count[block_t_indices[best_idx]] += 1
                    if np.all(existing_cover_count > 0):
                        break
                    continue
            
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            selected_blocks.append(best_idx)
            existing_cover_count[block_t_indices[best_idx]] += 1
            
            if np.all(existing_cover_count > 0):
                break

        return selected_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts
    num_runs = 0
    while time.time() < deadline - 10.0:
        seed = num_runs
        
        # Alternate between standard greedy and conflict-aware greedy
        if seed % 2 == 0:
            blocks = greedy_randomized(seed)
        else:
            blocks = greedy_packing(seed)
        
        if blocks is not None:
            # Reduce redundancy
            reduced = reduce_blocks(blocks)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_blocks = reduced
        
        num_runs += 1
        
        # If we've run enough, break
        if num_runs > 200: 
             break

    if best_blocks is not None:
        # Convert indices back to lists of points
        return [list(all_ks[i]) for i in best_blocks]
    
    # Fallback if no valid covering found (should not happen)
    blocks = greedy_randomized(0)
    if blocks:
        return [list(all_ks[i]) for i in blocks]
        
    return []
