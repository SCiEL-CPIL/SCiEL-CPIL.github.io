import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # 3. Build block_t_indices (list of lists of t-indices for each block)
    #    and block_masks (bitmasks for fast greedy)
    #    and blocks_covering_t (inverted index: t-index -> list of block indices)
    block_t_indices = [[] for _ in range(num_k)]
    block_masks = [0] * num_k
    blocks_covering_t = [[] for _ in range(num_t)]
    
    for idx, block in enumerate(k_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            t_idx = t_index_map[combo]
            block_t_indices[idx].append(t_idx)
            mask |= (1 << t_idx)
            blocks_covering_t[t_idx].append(idx)
        block_masks[idx] = mask
        
    full_mask = (1 << num_t) - 1
    
    def _greedy_pass(seed=0):
        """Greedy set cover with optional random tie-breaking."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_idx = -1
            best_count = -1
            best_ties = []
            
            # Iterate all blocks to find max gain
            # For speed, we don't shuffle indices, but we collect ties
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_ties = [i]
                elif gain == best_count and gain > 0:
                    best_ties.append(i)
            
            if best_idx == -1 and not best_ties:
                break
                
            if best_ties:
                if len(best_ties) == 1:
                    best_idx = best_ties[0]
                else:
                    best_idx = rng.choice(best_ties)
            else:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    def _local_search(selected_indices, max_seconds=None):
        """
        Local search using subset counts and inverted index.
        State: set of selected block indices.
        Objective: minimize size.
        """
        if time.time() > deadline:
            return selected_indices
            
        if max_seconds is None:
            max_seconds = deadline - time.time() - 5.0 # leave 5s for safety
            
        end_time = time.time() + max_seconds
        
        current_set = set(selected_indices)
        best_set = set(current_set)
        best_len = len(current_set)
        
        # Initialize subset counts
        subset_counts = np.zeros(num_t, dtype=np.int32)
        for idx in current_set:
            for t_idx in block_t_indices[idx]:
                subset_counts[t_idx] += 1
                
        # Helper to update counts
        def add_block(idx):
            for t_idx in block_t_indices[idx]:
                subset_counts[t_idx] += 1
            current_set.add(idx)
            
        def remove_block(idx):
            for t_idx in block_t_indices[idx]:
                subset_counts[t_idx] -= 1
            current_set.remove(idx)
            
        # Local search loop
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Try to remove redundant blocks
            # We iterate over a copy of the list to allow modification
            for idx in list(current_set):
                if time.time() > end_time:
                    return best_set
                    
                # Check if block idx is redundant
                # Redundant if all its t-subsets have count >= 2
                redundant = True
                for t_idx in block_t_indices[idx]:
                    if subset_counts[t_idx] <= 1:
                        redundant = False
                        break
                        
                if redundant:
                    remove_block(idx)
                    improved = True
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = set(current_set)
                    break # Restart loop after modification
                
            if not improved:
                # Try 1-swap
                # Pick a random block to try removing
                candidates = list(current_set)
                random.shuffle(candidates)
                
                for idx_to_remove in candidates:
                    if time.time() > end_time:
                        return best_set
                        
                    # Find critical t-subsets (count == 1)
                    critical_ts = []
                    for t_idx in block_t_indices[idx_to_remove]:
                        if subset_counts[t_idx] == 1:
                            critical_ts.append(t_idx)
                        
                    if not critical_ts:
                        # Should have been caught in redundancy check, but just in case
                        remove_block(idx_to_remove)
                        improved = True
                        if len(current_set) < best_len:
                            best_len = len(current_set)
                            best_set = set(current_set)
                        break
                        
                    # Find candidate replacement blocks
                    # Intersection of blocks covering all critical_ts
                    # Start with the set of blocks covering the first critical t
                    # Use sets for fast intersection
                    # Note: blocks_covering_t lists are small (364 for this problem)
                    candidates_set = set(blocks_covering_t[critical_ts[0]])
                    
                    for t_idx in critical_ts[1:]:
                        candidates_set &= set(blocks_covering_t[t_idx])
                        if not candidates_set:
                            break
                            
                    if candidates_set:
                        # Filter out already selected blocks
                        valid_candidates = [c for c in candidates_set if c not in current_set]
                        if valid_candidates:
                            # Pick a random valid candidate
                            replacement = random.choice(valid_candidates)
                            # Perform swap
                            remove_block(idx_to_remove)
                            add_block(replacement)
                            improved = True
                            # Size unchanged, so best_len doesn't update here unless we find removal later
                            # But structure changed, so we break to restart redundancy check
                            break
                            
        return best_set

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Greedy + Local Search
    # Run a few variations of greedy
    attempts = 0
    while time.time() < deadline - 10.0:
        seed = attempts
        selected = _greedy_pass(seed)
        
        if selected is None:
            break
            
        if len(selected) < best_len:
            best_len = len(selected)
            # Store indices for now
            best_indices = list(selected)
            
            # Run local search on this solution
            # Allocate remaining time minus safety margin
            # If this is the first run, give it more time
            time_remaining = deadline - time.time() - 5.0
            if attempts == 0:
                ls_time = time_remaining * 0.6
            else:
                ls_time = min(10.0, time_remaining * 0.2) # Limit subsequent searches
            
            polished_indices = _local_search(best_indices, max_seconds=ls_time)
            
            if len(polished_indices) < best_len:
                best_len = len(polished_indices)
                best_indices = polished_indices
                best_blocks = [list(k_blocks[i]) for i in best_indices]
            else:
                # If polish didn't improve, keep the unpolished one if it was better?
                # No, best_len was updated with greedy len. 
                # If polish didn't improve best_len, best_blocks might still be None or old.
                # We want to keep the best valid solution.
                if best_blocks is None or len(best_indices) < len(best_blocks):
                    best_blocks = [list(k_blocks[i]) for i in best_indices]
        
        attempts += 1
        
        # If we found a very good solution, stop early?
        # Lower bound is 264. Best known 335.
        if best_len <= 340:
            break

    # Fallback
    if best_blocks is None:
        selected = _greedy_pass(0)
        if selected:
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks

# Small self-test to ensure it doesn't crash on import or trivial cases
if __name__ == "__main__":
    # Quick test with small parameters to verify logic
    # C(6, 3, 2) should be small
    res = construct(6, 3, 2)
    print(f"C(6,3,2) found {len(res)} blocks")
    # Verify coverage
    v, k, t = 6, 3, 2
    points = set(range(v))
    blocks = [set(b) for b in res]
    all_t_subsets = list(itertools.combinations(points, t))
    covered = 0
    for sub in all_t_subsets:
        sub_set = set(sub)
        for b in blocks:
            if sub_set.issubset(b):
                covered += 1
                break
    print(f"Covered {covered}/{len(all_t_subsets)}")
    assert covered == len(all_t_subsets), "Coverage failed"
    print("Test passed")
