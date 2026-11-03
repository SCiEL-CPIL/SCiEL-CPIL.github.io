# EVOLVE-BLOCK-START
import itertools
import random
import time
import heapq
import numpy as np
from collections import defaultdict

_DEADLINE = 142.0
_start_time = None

# Precomputed global structures (lazy init)
_G = {}


def _init_globals(v, k, t):
    """Precompute all structures needed for the solver."""
    if (v, k, t) in _G:
        return _G[(v, k, t)]

    import itertools
    from collections import defaultdict

    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)
    
    # Convert block_ts to a list of numpy arrays for faster indexing? 
    # Or keep as lists but use np.array for bulk operations if needed.
    # For now, keep as lists of ints for flexibility, but we will use numpy for coverage vectors.

    # For each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # Precompute bitmask for each block: which t-subsets it covers
    # n_ts for C(19,4) = 3876, fits in a Python int
    block_mask = []
    for ts_list in block_ts:
        mask = 0
        for s in ts_list:
            mask |= (1 << s)
        block_mask.append(mask)

    # Precompute bitmask for each block: which points it contains
    block_pt_mask = []
    for blk in all_blocks:
        pm = 0
        for p in blk:
            pm |= (1 << p)
        block_pt_mask.append(pm)

    # For each t-subset, bitmask of points in it
    ts_pt_mask = []
    for ts in all_ts:
        pm = 0
        for p in ts:
            pm |= (1 << p)
        ts_pt_mask.append(pm)

    # Number of t-subsets per block
    n_ts_per_block = len(block_ts[0])  # C(k,t)

    _G[(v, k, t)] = {
        'all_ts': all_ts,
        'n_ts': n_ts,
        'ts_index': ts_index,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'block_ts': block_ts,
        'ts_to_blocks': ts_to_blocks,
        'block_mask': block_mask,
        'block_pt_mask': block_pt_mask,
        'ts_pt_mask': ts_pt_mask,
        'n_ts_per_block': n_ts_per_block,
    }
    return _G[(v, k, t)]


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    G = _init_globals(v, k, t)
    all_ts = G['all_ts']
    n_ts = G['n_ts']
    ts_index = G['ts_index']
    all_blocks = G['all_blocks']
    n_blocks = G['n_blocks']
    block_ts = G['block_ts']
    ts_to_blocks = G['ts_to_blocks']
    block_mask = G['block_mask']
    block_pt_mask = G['block_pt_mask']
    ts_pt_mask = G['ts_pt_mask']
    n_ts_per_block = G['n_ts_per_block']

    rng = random.Random(12345)

    def greedy_solve(seed, max_time_frac=0.5):
        """Run one greedy pass with given random seed for tie-breaking."""
        local_rng = random.Random(seed)
        time_budget = _DEADLINE * max_time_frac
        local_start = time.time()

        uncovered_mask = (1 << n_ts) - 1  # all t-subsets uncovered
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [n_ts_per_block] * n_blocks
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update coverage using bitmask
            bm = block_mask[b_idx]
            new_uncovered = bm & uncovered_mask
            # Count newly covered
            newly_covered = bin(new_uncovered).count('1')
            uncovered_count -= newly_covered
            uncovered_mask &= ~bm

            # Update gains of other blocks containing newly covered t-subsets
            # Iterate over set bits in new_uncovered
            nu = new_uncovered
            while nu:
                s_idx = (nu & (-nu)).bit_length() - 1
                nu &= nu - 1
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        # Convert to list of blocks
        result = [list(all_blocks[b]) for b in selected]

        # Check if complete
        if uncovered_count > 0:
            # Add remaining blocks greedily (brute force over remaining)
            while uncovered_count > 0 and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = bin(block_mask[b_idx] & uncovered_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                uncovered_mask &= ~block_mask[best_b]
                uncovered_count -= best_gain

        return result

    def local_search_improve(blocks, seed, time_limit):
        """Improve a covering by removing redundant blocks and trying swaps using NumPy."""
        local_rng = random.Random(seed)
        local_start = time.time()

        if not blocks:
            return blocks

        # Build coverage counts using NumPy
        coverage = np.zeros(n_ts, dtype=np.int32)
        # Pre-compute t-subset indices for each block in the current list for speed?
        # No, we'll compute on the fly or use a helper.
        # Actually, let's just use the global block_ts if the block is from all_blocks.
        # But blocks are lists of points. We need to map block -> t-subset indices.
        
        # Optimization: Since blocks are subsets of points, and we have all_blocks,
        # we can try to find the index in all_blocks? No, blocks might be modified/swapped.
        # However, the content of a block determines its t-subsets.
        # We can compute t-subset indices for a block on the fly.
        
        def get_ts_indices(block):
            return [ts_index[ts] for ts in itertools.combinations(block, t)]

        for blk in blocks:
            indices = get_ts_indices(blk)
            coverage[indices] += 1

        # Phase 1: Remove redundant blocks (fast pass)
        changed = True
        while changed and time.time() - local_start < time_limit:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() - local_start > time_limit:
                    break
                blk = blocks[i]
                indices = get_ts_indices(blk)
                # Check if min coverage > 1
                if np.min(coverage[indices]) > 1:
                    coverage[indices] -= 1
                    blocks.pop(i)
                    changed = True
                    break

        # Phase 2: Randomized swap search - try to replace a block with another
        # that maintains coverage but potentially allows further deletions.
        attempts = 0
        max_attempts = 3000
        while (time.time() - local_start < time_limit and attempts < max_attempts
               and len(blocks) > 0):
            attempts += 1
            idx = local_rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            
            # Get indices for old block
            old_indices = get_ts_indices(old_block)
            
            # Find t-subsets with coverage exactly 1 (only covered by this block)
            # unique_ts_indices = [i for i in old_indices if coverage[i] == 1]
            unique_mask = (coverage[old_indices] == 1)
            unique_indices = old_indices[unique_mask]
            
            if len(unique_indices) == 0:
                # Fully redundant - remove
                coverage[old_indices] -= 1
                blocks.pop(idx)
                continue

            # Must find a replacement covering all unique_ts
            # Quick check: union of points in unique_ts must fit in k
            union_pts_mask = 0
            for s_idx in unique_indices:
                union_pts_mask |= ts_pt_mask[s_idx]
            if bin(union_pts_mask).count('1') > k:
                continue

            # Find best replacement: must contain all points in union_pts_mask
            best_repl = -1
            best_overlap = -1

            # Precompute: for the old block, which t-subsets have coverage > 1
            # old_covered_by_others = set(i for i in old_indices if coverage[i] > 1)
            # We want to maximize overlap with already covered sets.
            
            # Candidate blocks must contain union_pts_mask
            # We can filter all_blocks or just iterate.
            # To speed up, we can pre-compute a list of blocks for each point mask? 
            # No, too many masks.
            
            # Iterate through all_blocks
            for b_idx in range(n_blocks):
                # Must contain all points in union_pts_mask
                if (block_pt_mask[b_idx] & union_pts_mask) != union_pts_mask:
                    continue

                # Count how many t-subsets of this block are already covered by others
                # i.e., coverage[s_idx] > 0 (after removing old block, so coverage[s_idx] >= 1 in current state? 
                # No, current state includes old block.
                # If we remove old block, coverage drops by 1 for old_indices.
                # So for a candidate block, the "new" coverage for a t-subset s is:
                # current_coverage[s] - (1 if s in old_indices else 0) + (1 if s in cand_indices else 0)
                # We want to ensure new_coverage[s] >= 1 for all s.
                # And we want to maximize "redundancy" to allow future deletions?
                # Actually, the previous logic was: overlap with sets covered by others.
                
                cand_indices = block_ts[b_idx]
                
                # Check validity: 
                # For any s in old_indices, if coverage[s] == 1, it MUST be in cand_indices.
                # We already know unique_indices are the ones with coverage 1.
                # So cand must contain all unique_indices.
                
                # We can check this using bitmask or set intersection.
                # Since n_ts_per_block is small (35), we can just loop.
                valid = True
                for s_idx in unique_indices:
                    if s_idx not in cand_indices:
                        valid = False
                        break
                if not valid:
                    continue

                # Calculate "score": how many t-subsets of cand are already covered by others (in current state)
                # This helps identify blocks that are "redundant" and can be removed later.
                overlap = 0
                for s_idx in cand_indices:
                    # If s_idx is in old_indices, its effective coverage after swap is coverage[s_idx] - 1 + 1 = coverage[s_idx].
                    # If s_idx is NOT in old_indices, its effective coverage is coverage[s_idx] + 1.
                    # We want to maximize the number of s_idx where coverage[s_idx] > 1 (in current state) 
                    # AND s_idx is in cand_indices?
                    # Actually, if s_idx is in cand_indices and coverage[s_idx] > 1, it's "safe".
                    if coverage[s_idx] > 1:
                        overlap += 1

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_repl = b_idx

            if best_repl == -1:
                continue

            # Check: does the replacement actually help?
            # Previous heuristic: best_overlap >= n_ts_per_block - len(unique_ts)
            min_needed = n_ts_per_block - len(unique_indices)
            if best_overlap < min_needed:
                continue

            new_block = all_blocks[best_repl]
            new_indices = block_ts[best_repl]

            # Apply swap
            coverage[old_indices] -= 1
            coverage[new_indices] += 1

            # Verify validity
            valid = True
            for s_idx in old_indices:
                if coverage[s_idx] == 0:
                    valid = False
                    break

            if valid:
                blocks[idx] = list(new_block)
                # After swap, try to remove other now-redundant blocks
                for j in range(len(blocks) - 1, -1, -1):
                    if j == idx:
                        continue
                    blk_j = blocks[j]
                    indices_j = get_ts_indices(blk_j)
                    if np.min(coverage[indices_j]) > 1:
                        coverage[indices_j] -= 1
                        blocks.pop(j)
                        break
            else:
                # Revert
                coverage[new_indices] -= 1
                coverage[old_indices] += 1

        return blocks

    def local_search_2exchange(blocks, seed, time_limit):
        """Try 2-exchange: remove two blocks, add one block that covers all uniquely-covered t-subsets."""
        local_rng = random.Random(seed)
        local_start = time.time()

        if len(blocks) < 2:
            return blocks

        # Build coverage counts
        coverage = np.zeros(n_ts, dtype=np.int32)
        
        def get_ts_indices(block):
            return [ts_index[ts] for ts in itertools.combinations(block, t)]

        for blk in blocks:
            indices = get_ts_indices(blk)
            coverage[indices] += 1

        improved = True
        while improved and time.time() - local_start < time_limit:
            improved = False
            n = len(blocks)
            if n < 2:
                break
            
            # Randomize pair order
            pairs = list(itertools.combinations(range(n), 2))
            local_rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() - local_start > time_limit:
                    return blocks
                
                idx1, idx2 = i, j
                if idx1 == idx2:
                    continue
                blk1 = blocks[idx1]
                blk2 = blocks[idx2]
                
                indices1 = get_ts_indices(blk1)
                indices2 = get_ts_indices(blk2)
                
                # Temporarily remove both
                coverage[indices1] -= 1
                coverage[indices2] -= 1
                
                # Find t-subsets that are now uncovered
                # uncovered_ts = [s for s in indices1 if coverage[s] == 0] + [s for s in indices2 if coverage[s] == 0 and s not in indices1]
                uncovered_set = set()
                for s in indices1:
                    if coverage[s] == 0:
                        uncovered_set.add(s)
                for s in indices2:
                    if coverage[s] == 0:
                        uncovered_set.add(s)
                        
                uncovered_ts = list(uncovered_set)
                
                if len(uncovered_ts) > 0:
                    # Look for a single block that covers all uncovered_ts
                    # Quick check: union of points must fit in k
                    union_pts_mask = 0
                    for s_idx in uncovered_ts:
                        union_pts_mask |= ts_pt_mask[s_idx]
                    if bin(union_pts_mask).count('1') > k:
                        # Revert
                        coverage[indices1] += 1
                        coverage[indices2] += 1
                        continue
                    
                    # Find candidate blocks covering all uncovered_ts
                    found = False
                    for cand in ts_to_blocks[uncovered_ts[0]]:
                        # Check if cand covers all uncovered_ts
                        cand_indices = block_ts[cand]
                        all_covered = True
                        for s_idx in uncovered_ts:
                            if s_idx not in cand_indices:
                                all_covered = False
                                break
                        if all_covered:
                            # Check that cand is not already in blocks (excluding the two we're removing)
                            cand_set = set(all_blocks[cand])
                            in_blocks = False
                            for b_idx, b in enumerate(blocks):
                                if b_idx == idx1 or b_idx == idx2:
                                    continue
                                if set(b) == cand_set:
                                    in_blocks = True
                                    break
                            if in_blocks:
                                continue
                            
                            # Valid 2->1 exchange: add cand
                            coverage[cand_indices] += 1
                            
                            # Remove the two blocks
                            # Remove higher index first
                            if idx2 > idx1:
                                blocks.pop(idx2)
                                blocks.pop(idx1)
                            else:
                                blocks.pop(idx1)
                                blocks.pop(idx2)
                            
                            blocks.append(list(all_blocks[cand]))
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Revert
                        coverage[indices1] += 1
                        coverage[indices2] += 1
                else:
                    # Both blocks were redundant (no uncovered t-subsets after removal)
                    # Remove one (the one at higher index)
                    if idx2 > idx1:
                        blocks.pop(idx2)
                    else:
                        blocks.pop(idx1)
                    improved = True
                    break
        
        return blocks

    # Multi-restart greedy
    best_result = None
    best_count = float('inf')

    # Determine number of restarts based on time
    n_restarts = 6
    # Allocate time: 45% to greedy, 25% to 1-exchange LS, 15% to 2-exchange LS, 15% to MILP polish
    total_budget = _DEADLINE - 10.0  # safety margin
    greedy_frac = 0.45
    ls1_frac = 0.25
    ls2_frac = 0.15
    milp_frac = 0.15
    time_per_greedy = total_budget * greedy_frac / n_restarts
    time_per_ls1 = total_budget * ls1_frac / n_restarts
    time_per_ls2 = total_budget * ls2_frac / n_restarts
    time_per_milp = total_budget * milp_frac

    for restart in range(n_restarts):
        if _remaining() < 15.0:
            break

        seed = 12345 + restart * 7919
        max_frac = min(0.50, time_per_greedy / _DEADLINE)

        result = greedy_solve(seed, max_time_frac=max_frac)

        if not result:
            continue

        # Check validity using bitmask
        cov_mask = 0
        for blk in result:
            blk_set = set(blk)
            for ts in itertools.combinations(blk, t):
                cov_mask |= (1 << ts_index[ts])
        if cov_mask != (1 << n_ts) - 1:
            # Invalid - skip
            continue

        # Local search 1-exchange
        ls1_time = min(time_per_ls1, _remaining() - 20.0)
        if ls1_time > 2.0:
            result = local_search_improve(result, seed + 1, ls1_time)

        # Local search 2-exchange
        ls2_time = min(time_per_ls2, _remaining() - 15.0)
        if ls2_time > 1.5:
            result = local_search_2exchange(result, seed + 2, ls2_time)

        if len(result) < best_count:
            best_count = len(result)
            best_result = result

    if best_result is None:
        # Fallback: single greedy pass
        best_result = greedy_solve(12345, max_time_frac=0.9)

    # Final verification
    final_cov = [0] * n_ts
    for blk in best_result:
        for ts in itertools.combinations(blk, t):
            final_cov[ts_index[ts]] += 1
    if not all(c > 0 for c in final_cov):
        # Emergency fix: add blocks for uncovered subsets
        unc_mask = 0
        for i in range(n_ts):
            if final_cov[i] == 0:
                unc_mask |= (1 << i)
        while unc_mask and _remaining() > 0.5:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = bin(block_mask[b_idx] & unc_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            best_result.append(list(all_blocks[best_b]))
            unc_mask &= ~block_mask[best_b]

    # MILP polish: try to remove redundant blocks using a more systematic approach
    if _remaining() > time_per_milp + 5.0:
        milp_time = min(time_per_milp, _remaining() - 5.0)
        if milp_time > 3.0:
            best_result = _milp_polish(best_result, v, k, t, G, milp_time)

    return best_result


def _milp_polish(blocks, v, k, t, G, time_limit):
    """Try to reduce the number of blocks using a MILP-based approach.
    
    Formulation: minimize sum(x_i) subject to coverage constraints.
    We use a greedy deletion heuristic informed by coverage counts.
    """
    import time
    
    all_ts = G['all_ts']
    n_ts = G['n_ts']
    ts_index = G['ts_index']
    all_blocks = G['all_blocks']
    n_blocks = G['n_blocks']
    block_ts = G['block_ts']
    ts_to_blocks = G['ts_to_blocks']
    block_mask = G['block_mask']
    n_ts_per_block = G['n_ts_per_block']
    
    local_start = time.time()
    
    if not blocks:
        return blocks
    
    # Build coverage counts
    coverage = np.zeros(n_ts, dtype=np.int32)
    
    def get_ts_indices(block):
        return [ts_index[ts] for ts in itertools.combinations(block, t)]

    for blk in blocks:
        indices = get_ts_indices(blk)
        coverage[indices] += 1
    
    # Iteratively try to remove blocks
    # Priority: blocks with most redundancy (highest min coverage among their t-subsets)
    improved = True
    while improved and time.time() - local_start < time_limit:
        improved = False
        n = len(blocks)
        if n == 0:
            break
        
        # For each block, compute its "removability score"
        # A block is removable if all its t-subsets have coverage > 1
        # Score = how much redundancy it has (higher = more redundant)
        best_idx = -1
        best_score = -1
        
        for i in range(n):
            blk = blocks[i]
            indices = get_ts_indices(blk)
            min_cov = np.min(coverage[indices])
            if min_cov > 1:
                # This block is removable
                # Score: prefer removing blocks that free up the most coverage
                score = np.sum(coverage[indices])
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        if best_idx >= 0:
            # Remove the block
            blk = blocks[best_idx]
            indices = get_ts_indices(blk)
            coverage[indices] -= 1
            blocks.pop(best_idx)
            improved = True
    
    return blocks
# EVOLVE-BLOCK-END
