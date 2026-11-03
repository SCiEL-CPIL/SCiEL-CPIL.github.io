import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    # tsub_mask[b]: bitset of t-subsets covered by block b
    # tsub_blocks[ts]: bitset of blocks covering t-subset ts
    # block_tsubs[b]: list of t-subset indices covered by block b
    tsub_mask = [0] * n_blocks
    tsub_blocks = [0] * n_tsubsets
    block_tsubs = [[] for _ in range(n_blocks)]

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tsub_list = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks[idx] |= (1 << b_idx)
            tsub_list.append(idx)
        tsub_mask[b_idx] = mask
        block_tsubs[b_idx] = tsub_list

    full_tsub_mask = (1 << n_tsubsets) - 1

    def greedy_cover(seed):
        """
        Greedy set cover using a heap.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected
            
            # Update gains
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with deletions, 1-for-1 swaps, and 2-for-1 replacements.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        # Calculate cover counts
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: 1-for-1 Swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                # Find tsubs uniquely covered by b
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    # Block is redundant, remove it
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates covering all unique_tsubs
                # Start with most constrained
                min_ts = min(unique_tsubs, key=lambda x: tsub_blocks[x].bit_count())
                candidates = tsub_blocks[min_ts]
                
                # Intersect with others
                for ts_idx in unique_tsubs[1:]:
                    candidates &= tsub_blocks[ts_idx]
                    if not candidates:
                        break

                if not candidates:
                    continue

                # Exclude current block and already selected blocks
                candidates &= ~selected_mask_of(block_set) # This is slow, let's use a different check
                
                # Actually, we just need to check if candidate is in block_set
                # Let's iterate through candidates
                cand_list = []
                m = candidates
                while m:
                    lsb = m & -m
                    b_idx = lsb.bit_length() - 1
                    if b_idx not in block_set:
                        cand_list.append(b_idx)
                    m ^= lsb
                
                if not cand_list:
                    continue

                rng.shuffle(cand_list)
                
                swapped = False
                for r in cand_list[:10]: # Limit candidates
                    # Try swap
                    # Remove b
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    
                    # Add r
                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1
                    
                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

            if improved:
                continue

            # Phase 3: 2-for-1 Replacements
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Try random pairs
            n_checks = min(50, len(current_blocks) * (len(current_blocks)-1) // 2)
            for _ in range(n_checks):
                if time.time() > deadline - 5.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                # Find required tsubs
                required_set = set()
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_set.add(ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_set.add(ts_idx)

                if not required_set:
                    # Both redundant?
                    block_set.remove(b1)
                    for ts_idx in block_tsubs[b1]:
                        cover_count[ts_idx] -= 1
                    block_set.remove(b2)
                    for ts_idx in block_tsubs[b2]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidate covering all required
                min_ts = min(required_set, key=lambda x: tsub_blocks[x].bit_count())
                candidates = tsub_blocks[min_ts]
                
                for ts_idx in list(required_set)[1:]:
                    candidates &= tsub_blocks[ts_idx]
                    if not candidates:
                        break

                if not candidates:
                    continue

                candidates &= ~(1 << b1)
                candidates &= ~(1 << b2)
                
                cand_list = []
                m = candidates
                while m:
                    lsb = m & -m
                    b_idx = lsb.bit_length() - 1
                    if b_idx not in block_set:
                        cand_list.append(b_idx)
                    m ^= lsb

                if not cand_list:
                    continue

                rng.shuffle(cand_list)
                r = cand_list[0]

                # Perform swap
                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1
                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1
                
                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1
                
                improved = True
                break

        return list(block_set)

    def selected_mask_of(block_set):
        mask = 0
        for b in block_set:
            mask |= (1 << b)
        return mask

    def perturb_and_repair(blocks, seed):
        """
        Perturb by removing some blocks and repair with greedy.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks
        
        block_set = set(blocks)
        
        # Remove 10-20% of blocks
        n_remove = max(1, len(block_set) // 10)
        to_remove = rng.sample(list(block_set), min(n_remove, len(block_set)))
        
        for b in to_remove:
            block_set.remove(b)
            
        # Calculate uncovered
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        uncovered_mask = 0
        for ts_idx in range(n_tsubsets):
            if cover_count[ts_idx] == 0:
                uncovered_mask |= (1 << ts_idx)
                
        if uncovered_mask == 0:
            return list(block_set)
            
        # Greedy repair
        # We need to add blocks to cover uncovered_mask
        # Use a simple greedy on the uncovered set
        # To save time, we just run a quick greedy
        # Initialize gains for repair
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            if b_idx in block_set:
                gains[b_idx] = -1
            else:
                gains[b_idx] = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks) if gains[b_idx] > 0]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
            best_block = -1
            best_gain = 0
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and b_idx not in block_set:
                    best_block = b_idx
                    best_gain = g
                    break
                    
            if best_block == -1:
                break
                
            block_set.add(best_block)
            gains[best_block] = -1
            
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                cm = tsub_blocks[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if b_idx2 not in block_set and gains[b_idx2] > 0:
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb
                
        return list(block_set)

    best_solution = None
    best_size = float('inf')

    # Run ILS
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        seed = 1000 + i
        
        # 1. Greedy
        greedy_sol = greedy_cover(seed)
        if not greedy_sol:
            continue
            
        # 2. Local Search
        ls_sol = local_search(greedy_sol, seed)
        
        # 3. Perturb and Repair
        perturbed_sol = perturb_and_repair(ls_sol, seed)
        
        # 4. Local Search again
        final_sol = local_search(perturbed_sol, seed + 100)
        
        if len(final_sol) < best_size:
            best_size = len(final_sol)
            best_solution = final_sol.copy()
            
        # Update best with ls_sol too
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol.copy()

    # Final local search on best
    if best_solution is not None and time.time() < deadline - 5.0:
        final_ls = local_search(best_solution, 9999)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_solution = final_ls

    if best_solution is None:
        best_solution = greedy_cover(0)

    return [list(all_blocks[idx]) for idx in best_solution]
