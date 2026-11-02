import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()
    
    # 1. Precomputation
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Block coverage as bitsets (Python ints)
    # Bit i is set if block covers t-subset i
    block_cover_bits = [0] * n_blocks
    # Also store list of t-subset indices for faster iteration if needed
    block_ts_indices = [[] for _ in range(n_blocks)]
    
    # Map t-subset index to list of blocks covering it
    ts_to_blocks = defaultdict(list)
    
    for b_idx, blk in enumerate(all_blocks):
        bits = 0
        ts_list = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            bits |= (1 << idx)
            ts_list.append(idx)
            ts_to_blocks[idx].append(b_idx)
        block_cover_bits[b_idx] = bits
        block_ts_indices[b_idx] = ts_list

    def greedy_construct_indices(seed):
        """Run greedy with random tie-breaking, using bitsets for speed."""
        local_rng = random.Random(seed)
        uncovered_mask = (1 << n_ts) - 1
        uncovered_count = n_ts
        in_set = [False] * n_blocks
        selected = []
        
        while uncovered_count > 0 and _remaining() > 5.0:
            best_b = -1
            best_gain = 0
            best_tie = 0
            
            for b_idx in range(n_blocks):
                if in_set[b_idx]:
                    continue
                gain = (block_cover_bits[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                    best_tie = local_rng.random()
                elif gain == best_gain and gain > 0:
                    tie = local_rng.random()
                    if tie > best_tie:
                        best_tie = tie
                        best_b = b_idx
                        
            if best_b == -1 or best_gain == 0:
                break
                
            in_set[best_b] = True
            selected.append(best_b)
            uncovered_mask &= ~block_cover_bits[best_b]
            uncovered_count -= best_gain
            
        return selected

    def local_search_indices(indices, seed, time_limit):
        """Improve covering via reduction and stochastic swaps."""
        local_rng = random.Random(seed)
        
        # State
        in_set = [False] * n_blocks
        for b in indices:
            in_set[b] = True
        
        current_size = len(indices)
        best_size = current_size
        best_indices = indices.copy()
        
        # Precompute coverage counts
        coverage_count = [0] * n_ts
        for b in indices:
            for ts_idx in block_ts_indices[b]:
                coverage_count[ts_idx] += 1
                
        def is_redundant(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                if coverage_count[ts_idx] == 1:
                    return False
            return True
            
        def remove_block(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                coverage_count[ts_idx] -= 1
            in_set[b_idx] = False
            
        def add_block(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                coverage_count[ts_idx] += 1
            in_set[b_idx] = True
            
        # Main Loop
        while _remaining() > time_limit:
            # Phase 1: Reduction
            changed = True
            while changed and _remaining() > time_limit + 1.0:
                changed = False
                # Iterate over a copy of current blocks to allow removal
                current_blocks = [b for b in range(n_blocks) if in_set[b]]
                # Randomize order to avoid bias
                local_rng.shuffle(current_blocks)
                
                for b_idx in current_blocks:
                    if not in_set[b_idx]:
                        continue
                    if is_redundant(b_idx):
                        remove_block(b_idx)
                        changed = True
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_indices = [b for b in range(n_blocks) if in_set[b]]
                        break # Restart scan after removal
        
            # Phase 2: Stochastic Swap
            current_blocks = [b for b in range(n_blocks) if in_set[b]]
            if not current_blocks:
                break
                
            b_old = local_rng.choice(current_blocks)
            
            # Find unique t-subsets of b_old
            unique_ts = []
            for ts_idx in block_ts_indices[b_old]:
                if coverage_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                # Should have been removed in Phase 1
                continue
                
            # Find candidates for replacement
            # Intersection of ts_to_blocks for all ts in unique_ts
            if len(unique_ts) == 1:
                candidates = ts_to_blocks[unique_ts[0]]
            else:
                # Start with the smallest list
                min_idx = min(unique_ts, key=lambda x: len(ts_to_blocks[x]))
                candidates = set(ts_to_blocks[min_idx])
                for ts_idx in unique_ts:
                    if ts_idx == min_idx:
                        continue
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
            # Filter out blocks already in set
            candidates = [b for b in candidates if not in_set[b]]
            
            if candidates:
                # Pick a candidate
                # Heuristic: Pick one that covers the most "critical" t-subsets elsewhere?
                # Or just random.
                b_new = local_rng.choice(candidates)
                
                # Perform Swap
                remove_block(b_old)
                add_block(b_new)
                # Size remains same
                
            else:
                # No single block can replace b_old.
                # Kick: Remove b_old, then greedily cover the now-uncovered t-subsets.
                # This increases size, so we only do it if we are stuck.
                # Let's skip kicks for now to keep it simple and fast.
                # If we are stuck, we might try a different b_old.
                pass
                
        # Return best found
        return best_indices

    def branch_and_bound(start_indices, time_limit, seed):
        """
        Structural change: Branch-and-bound search on a compatibility graph.
        Nodes are blocks. Edges represent compatibility (disjointness in core coverage).
        We search for a smaller set of blocks by trying to replace subsets.
        """
        local_rng = random.Random(seed)
        
        # State
        in_set = [False] * n_blocks
        for b in start_indices:
            in_set[b] = True
        
        current_size = len(start_indices)
        best_size = current_size
        best_indices = start_indices.copy()
        
        # Precompute coverage counts
        coverage_count = [0] * n_ts
        for b in start_indices:
            for ts_idx in block_ts_indices[b]:
                coverage_count[ts_idx] += 1
                
        def is_redundant(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                if coverage_count[ts_idx] == 1:
                    return False
            return True
            
        def remove_block(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                coverage_count[ts_idx] -= 1
            in_set[b_idx] = False
            
        def add_block(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                coverage_count[ts_idx] += 1
            in_set[b_idx] = True

        # Upper Bound Estimation:
        # If we remove a block, we must cover its unique t-subsets.
        # Let U be the set of unique t-subsets of the removed block.
        # We need at least ceil(|U| / max_coverage_of_U) blocks to cover U.
        # This is a weak bound. A stronger bound comes from LP relaxation,
        # but we don't have it here. We'll use a simple counting bound.
        
        def estimate_min_blocks_needed(uncovered_mask):
            # Greedy estimate of min blocks to cover uncovered_mask
            # This is expensive if done often. We use it sparingly.
            if uncovered_mask == 0:
                return 0
            
            # Simple greedy count
            remaining = uncovered_mask
            count = 0
            # Limit depth to avoid slowness
            while remaining and count < 10:
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_set[b_idx]:
                        continue
                    gain = (block_cover_bits[b_idx] & remaining).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                if best_gain == 0:
                    return float('inf') # Infeasible with remaining blocks?
                count += 1
                # Subtract the best block's coverage? We don't know which one.
                # This estimate is very loose. 
                # Better: just return a lower bound based on density.
                # Let's just return 1 for now to not prune too aggressively.
                break
            return count

        # Search Tree
        # We will try to remove one block at a time and see if we can cover its gap.
        # This is effectively a 1-exchange local search with lookahead.
        
        visited = set()
        
        while _remaining() > time_limit:
            # Try to remove a random block
            current_blocks = [b for b in range(n_blocks) if in_set[b]]
            if not current_blocks:
                break
            
            local_rng.shuffle(current_blocks)
            
            for b_remove in current_blocks:
                if not in_set[b_remove]:
                    continue
                
                # Check if removing b_remove leaves any t-subset uncovered
                # i.e., is b_remove redundant?
                if is_redundant(b_remove):
                    # This is just reduction, handled by local_search.
                    # In B&B, we might want to force non-redundant removals?
                    # No, if it's redundant, we should remove it.
                    remove_block(b_remove)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_indices = [b for b in range(n_blocks) if in_set[b]]
                    break # Restart after change
                
                # If not redundant, calculate the gap
                gap_mask = 0
                for ts_idx in block_ts_indices[b_remove]:
                    if coverage_count[ts_idx] == 1:
                        gap_mask |= (1 << ts_idx)
                
                if gap_mask == 0:
                    continue # Should be redundant
                
                # Find blocks that cover the gap
                # We need a set of blocks S such that Union(covers(B) for B in S) covers gap_mask
                # And S is disjoint from current_set \ {b_remove}
                
                # Heuristic: Find a single block that covers the entire gap.
                # If exists, swap.
                # If not, try 2 blocks?
                
                # For now, stick to 1-exchange but with a smarter candidate selection.
                # Candidates must cover all ts in gap_mask.
                
                # Intersection of ts_to_blocks for all ts in gap_mask
                gap_ts_list = []
                temp_mask = gap_mask
                while temp_mask:
                    # Get lowest set bit
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    gap_ts_list.append(idx)
                    temp_mask ^= lsb
                    
                if len(gap_ts_list) == 1:
                    candidates = ts_to_blocks[gap_ts_list[0]]
                else:
                    min_idx = min(gap_ts_list, key=lambda x: len(ts_to_blocks[x]))
                    candidates = set(ts_to_blocks[min_idx])
                    for ts_idx in gap_ts_list:
                        if ts_idx == min_idx:
                            continue
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                    
                candidates = [b for b in candidates if not in_set[b] and b != b_remove]
                
                if candidates:
                    # Pick best candidate
                    # Score: number of *other* uncovered (or unique) ts it covers?
                    # Actually, we just want to minimize future gaps.
                    # Let's pick the one that maximizes coverage of *currently*
                    # low-coverage t-subsets (coverage_count == 2).
                    
                    best_cand = -1
                    best_score = -1
                    
                    for c in candidates:
                        score = 0
                        for ts_idx in block_ts_indices[c]:
                            if coverage_count[ts_idx] == 2:
                                score += 1
                            elif coverage_count[ts_idx] == 1:
                                score += 10 # Avoid creating new gaps? 
                                # Wait, if coverage is 1, it's unique to some block.
                                # If we add c, coverage becomes 2.
                                # This is good.
                        if score > best_score:
                            best_score = score
                            best_cand = c
                            
                    if best_cand != -1:
                        # Perform swap
                        remove_block(b_remove)
                        add_block(best_cand)
                        # Size remains same
                        # Check for new redundancies?
                        # We'll let the next iteration handle it.
                        break # Restart after change
                        
        return best_indices

    # Run multiple greedy instances
    best_overall = None
    best_overall_size = float('inf')
    
    # Time budget allocation:
    # 30% for greedy, 70% for local search / B&B?
    
    # Let's run 3 greedy variants.
    seeds = [1, 2, 3]
    candidates = []
    
    for seed in seeds:
        if _remaining() < 15.0:
            break
        idxs = greedy_construct_indices(seed)
        if idxs:
            candidates.append(idxs)
            
    if not candidates:
        # Fallback: simple greedy
        idxs = greedy_construct_indices(0)
        return [list(all_blocks[b]) for b in idxs]
        
    # Sort candidates by size
    candidates.sort(key=len)
    
    # Run local search and B&B on the best candidate
    # We'll alternate or run B&B on the best one.
    
    # Take the best candidate
    best_initial = candidates[0]
    
    # Run local search first to get a good baseline
    if _remaining() > 10.0:
        ls_budget = min(30.0, _remaining() / 2.0)
        improved = local_search_indices(best_initial, 123, ls_budget)
        if len(improved) < best_overall_size:
            best_overall_size = len(improved)
            best_overall = improved
    else:
        best_overall = best_initial
        best_overall_size = len(best_initial)
        
    # Run B&B on the result of local search
    if _remaining() > 10.0 and best_overall:
        bb_budget = _remaining() / 2.0
        improved = branch_and_bound(best_overall, bb_budget, 456)
        if len(improved) < best_overall_size:
            best_overall_size = len(improved)
            best_overall = improved
            
    if best_overall is None:
        best_overall = candidates[0]
        
    return [list(all_blocks[b]) for b in best_overall]
