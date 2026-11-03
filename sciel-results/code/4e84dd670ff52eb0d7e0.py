import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitsets for fast set operations
    tsub_mask = [0] * n_blocks       # tsub_mask[b] has bits for tsubs covered by block b
    tsub_blocks_mask = [0] * n_tsubsets # tsub_blocks_mask[ts] has bits for blocks covering ts
    block_tsub_sets = [None] * n_blocks # Sets for local search intersection
    tsub_block_sets = [None] * n_tsubsets

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tsub_set = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
            tsub_set.add(idx)
        tsub_mask[b_idx] = mask
        block_tsub_sets[b_idx] = tsub_set

    for i in range(n_tsubsets):
        tsub_block_sets[i] = set()
        m = tsub_blocks_mask[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_block_sets[i].add(idx)
            m ^= lsb

    full_tsub_mask = (1 << n_tsubsets) - 1
    
    # Precompute conflict masks for Packing LB (independent sets)
    # Two t-subsets conflict if they are covered by the same block
    tsub_conflict_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        mask = tsub_mask[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_conflict_mask[idx] |= mask
            m ^= lsb

    def packing_lb(uncovered_mask):
        """
        Calculate lower bound using maximal independent set of uncovered t-subsets.
        Two t-subsets are independent if no single block covers both.
        """
        count = 0
        remaining = uncovered_mask
        while remaining:
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            remaining &= ~tsub_conflict_mask[idx]
        return count

    def greedy_cover(seed):
        """
        Standard greedy using heap for gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains: number of tsubs covered by each block
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
                # Stale or already selected, discard

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected
            
            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
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
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

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

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates using set intersection
                # Start with the most constrained unique tsub
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work per swap attempt
                if len(cand_list) > 20:
                    cand_list = cand_list[:20]

                swapped = False
                for r in cand_list:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checks = 50
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

                # Find tsubs that would become uncovered if we removed b1 and b2
                required_set = set()
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_set.add(ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_set.add(ts_idx)

                if not required_set:
                    # Both blocks are redundant?
                    # Try removing both
                    block_set.remove(b1)
                    for ts_idx in block_tsubs[b1]:
                        cover_count[ts_idx] -= 1
                    block_set.remove(b2)
                    for ts_idx in block_tsubs[b2]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidate blocks covering all required_set
                # Start with most constrained
                min_ts = min(required_set, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in list(required_set)[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                
                if not candidates:
                    continue

                candidates.discard(b1)
                candidates.discard(b2)
                
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                if len(cand_list) > 5:
                    cand_list = cand_list[:5]

                r = cand_list[0]

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

    def branch_and_bound(uncovered_mask, current_blocks, current_size, best_size, selected_mask):
        """
        Branch-and-bound with bitset-based lower bounds.
        """
        if time.time() > deadline - 5.0:
            return False, None

        if uncovered_mask == 0:
            return True, list(current_blocks)

        # Lower bound: Packing LB
        lb_pack = packing_lb(uncovered_mask)
        
        n_uncovered = uncovered_mask.bit_count()
        if n_uncovered == 0:
            return True, list(current_blocks)
            
        if current_size + lb_pack >= best_size:
            return False, None

        # Select the most constrained uncovered t-subset (fewest candidate blocks)
        best_ts = -1
        best_candidates = None
        best_cand_count = float('inf')
        
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            ts_idx = lsb.bit_length() - 1
            candidates = tsub_blocks_mask[ts_idx] & ~selected_mask
            cand_count = candidates.bit_count()
            if cand_count < best_cand_count:
                best_cand_count = cand_count
                best_ts = ts_idx
                best_candidates = candidates
            if cand_count == 0:
                return False, None
            m ^= lsb

        if best_ts == -1:
            return True, list(current_blocks)

        # Branch on which block to use for best_ts
        candidate_list = []
        m = best_candidates
        while m:
            lsb = m & (-m)
            b_idx = lsb.bit_length() - 1
            cover = (tsub_mask[b_idx] & uncovered_mask).bit_count()
            candidate_list.append((cover, b_idx))
            m ^= lsb
        
        # Sort by coverage descending
        candidate_list.sort(reverse=True)
        
        # Limit branching factor
        max_branch = min(8, len(candidate_list))
        
        for cover, b_idx in candidate_list[:max_branch]:
            new_blocks = current_blocks + [b_idx]
            new_uncovered = uncovered_mask & ~tsub_mask[b_idx]
            new_selected_mask = selected_mask | (1 << b_idx)
            
            found, sol = branch_and_bound(
                new_uncovered, new_blocks, current_size + 1, best_size, new_selected_mask
            )
            
            if found and sol is not None:
                return True, sol
        
        return False, None

    best_solution = None
    best_size = float('inf')

    # Run multiple restarts
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 25.0:
            break

        seed = 1000 + i
        greedy_sol = greedy_cover(seed)

        if not greedy_sol:
            continue

        # Local Search
        ls_sol = local_search(greedy_sol, seed)

        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)

        # Try B&B to potentially improve further
        # B&B from scratch is expensive, so use it sparingly or only if time permits
        if time.time() < deadline - 15.0 and len(t2o_sol) > 140:
            # Initialize B&B from scratch with the current best as upper bound
            bnb_sol = branch_and_bound(
                full_tsub_mask, [], 0, len(t2o_sol), 0
            )
            if bnb_sol[0] and bnb_sol[1] is not None:
                t2o_sol = bnb_sol[1]

        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()

    # Final pass: try more aggressive local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        final_ls = local_search(best_solution, 9999)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_solution = final_ls

        # Try 2-for-1 on the final best
        if time.time() < deadline - 5.0:
            final_t2o = two_for_one_search(best_solution, 8888)
            if len(final_t2o) < best_size:
                best_size = len(final_t2o)
                best_solution = final_t2o

    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = greedy_cover(0)

    return [list(all_blocks[idx]) for idx in best_solution]
