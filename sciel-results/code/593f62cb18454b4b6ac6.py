import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin for 150s limit

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    # Precompute masks as integer bitmasks for fast coverage checks
    masks = [0] * num_k_blocks
    for i in range(num_k_blocks):
        mask = 0
        for idx in block_to_ts[i]:
            mask |= (1 << idx)
        masks[i] = mask

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        # Precompute C(k,t)
        def C(n, r):
            if r > n or r < 0:
                return 0
            if r == 0 or r == n:
                return 1
            r = min(r, n - r)
            result = 1
            for i in range(1, r + 1):
                result = result * (n - i + 1) // i
            return result

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Vectorized gain update
            if len(newly_covered) > 0:
                all_blocks_to_decrement = []
                for ts_idx in newly_covered:
                    all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                if all_blocks_to_decrement:
                    counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    def prune_redundant(block_indices):
        """
        Remove blocks that are redundant.
        Returns a new list of block indices.
        """
        if not block_indices:
            return []
        
        # Count coverage
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                # Check if removing idx keeps all counts >= 1
                mask = masks[idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Remove idx
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_improve(block_indices, max_iters=2000):
        """
        WalkSAT-style local search: try to remove blocks, and if stuck,
        try 1-exchange swaps to enable further removals.
        """
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        
        # Precompute coverage counts
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        # Helper to check if a block is removable
        def is_removable(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_counts[j] == 1:
                    return False
                m ^= lsb
            return True
        
        # Helper to remove a block
        def remove_block(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] -= 1
                m ^= lsb
            current_blocks.remove(idx)
            
        # Helper to add a block
        def add_block(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
            current_blocks.append(idx)

        # Phase A: Iterative removal
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            random.shuffle(current_blocks)
            
            for idx in current_blocks[:]:
                if time.time() > deadline - 3.0:
                    break
                if is_removable(idx):
                    remove_block(idx)
                    improved = True
        
        # Phase B: 1-exchange swaps to break local optima
        # Try to replace a block with one that covers its unique t-subsets
        # and allows removal of another block
        iters = 0
        while iters < max_iters and time.time() < deadline - 3.0:
            iters += 1
            if len(current_blocks) == 0:
                break
            
            i = random.randint(0, len(current_blocks) - 1)
            idx_i = current_blocks[i]
            mask_i = masks[idx_i]
            
            # Find t-subsets uniquely covered by block i
            unique_ts = []
            m = mask_i
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_counts[j] == 1:
                    unique_ts.append(j)
                m ^= lsb
            
            if len(unique_ts) == 0:
                # Block is redundant, remove it
                remove_block(idx_i)
                continue
            
            # Find candidate blocks that cover all unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [c for c in candidates if c not in set(current_blocks)]
            if not valid_candidates:
                continue
            
            new_block_idx = random.choice(valid_candidates)
            
            # Check if swap is valid: after removing idx_i and adding new_block_idx,
            # all t-subsets still covered
            # First, check if new_block_idx covers all unique_ts (it does by construction)
            # But we need to ensure no t-subset becomes uncovered
            
            # Simulate: remove idx_i, add new_block_idx
            # A t-subset becomes uncovered only if it was covered ONLY by idx_i
            # and new_block_idx does NOT cover it.
            # But by construction, new_block_idx covers all unique_ts of idx_i.
            # However, there might be t-subsets that are covered by both idx_i and
            # some other block, but NOT by new_block_idx. That's fine because
            # they're still covered by the other block.
            # The only risk is if new_block_idx introduces a t-subset that was
            # previously uncovered... but all t-subsets are already covered.
            
            # Actually, the swap is always valid if new_block_idx covers all unique_ts
            # of idx_i, because:
            # - t-subsets covered only by idx_i are covered by new_block_idx
            # - t-subsets covered by idx_i and others remain covered by others
            # - t-subsets not covered by idx_i are unaffected
            
            # Perform swap
            remove_block(idx_i)
            add_block(new_block_idx)
            
            # After swap, try to remove more blocks
            improved2 = True
            while improved2 and time.time() < deadline - 3.0:
                improved2 = False
                random.shuffle(current_blocks)
                for idx in current_blocks[:]:
                    if time.time() > deadline - 3.0:
                        break
                    if is_removable(idx):
                        remove_block(idx)
                        improved2 = True
        
        return current_blocks

    # --- Main Loop ---
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts with different strategies
    num_restarts = 200  # Limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Vary explore probability for diversity
        explore_prob = min(0.3, restart * 0.005)
        
        # 1. Greedy
        blocks = run_greedy(seed=restart, explore_prob=explore_prob)
        if blocks is None:
            continue
        
        # Convert to indices
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
        block_indices = []
        for b in blocks:
            idx = block_tuple_to_idx.get(tuple(b), -1)
            if idx >= 0:
                block_indices.append(idx)
        
        if not block_indices:
            continue
            
        # 2. Prune
        pruned = prune_redundant(block_indices)
        
        # 3. Local Search
        if len(pruned) < best_count:
            ls_blocks = local_search_improve(pruned)
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = [list(all_k_blocks[i]) for i in ls_blocks]
        else:
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_blocks = [list(all_k_blocks[i]) for i in pruned]

    if best_blocks is None:
        # Fallback: Run one deterministic greedy
        blocks = run_greedy(seed=0, explore_prob=0.0)
        if blocks is not None:
            block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
            block_indices = []
            for b in blocks:
                idx = block_tuple_to_idx.get(tuple(b), -1)
                if idx >= 0:
                    block_indices.append(idx)
            if block_indices:
                pruned = prune_redundant(block_indices)
                best_blocks = [list(all_k_blocks[i]) for i in pruned]
            else:
                best_blocks = []
        else:
            best_blocks = []

    return best_blocks
