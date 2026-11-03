import itertools
import numpy as np
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Precompute t-subsets and k-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # For each k-subset (block), compute the bitmask of t-subsets it covers
    # n_t can be up to C(19,4)=3876, so we need multiple int64 words
    n_words = (n_t + 63) // 64

    # block_masks[j] = list of n_words int64 values representing t-subsets covered by block j
    block_masks = [np.zeros(n_words, dtype=np.int64) for _ in range(n_k)]
    # block_tlist[j] = list of t-subset indices covered by block j (for iteration)
    block_tlist = [[] for _ in range(n_k)]

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            word = i // 64
            bit = i % 64
            block_masks[j][word] |= (np.int64(1) << np.int64(bit))
            block_tlist[j].append(i)

    # Precompute for each t-subset, which blocks cover it (for SA moves)
    # t_to_blocks[i] = list of block indices covering t-subset i
    t_to_blocks = [[] for _ in range(n_t)]
    for j in range(n_k):
        for i in block_tlist[j]:
            t_to_blocks[i].append(j)

    # Also: for each block, the set of t-subset indices (as a set for fast intersection)
    # Actually block_tlist is already a list; we'll use it directly.

    def is_covered(covered_masks):
        """Check if all t-subsets are covered. covered_masks is array of n_words int64."""
        for w in range(n_words):
            if w < n_words - 1:
                if covered_masks[w] != -1:  # all bits set
                    return False
            else:
                # Last word: only n_t % 64 bits matter (or 64 if divisible)
                remaining = n_t - w * 64
                if remaining < 64:
                    mask = (np.int64(1) << np.int64(remaining)) - 1
                    if covered_masks[w] != mask:
                        return False
                else:
                    if covered_masks[w] != -1:
                        return False
        return True

    def popcount_mask(m):
        """Count bits set in a multi-word bitmask."""
        total = 0
        for w in range(n_words):
            total += np.bit_count(np.uint64(m[w]))
        return total

    # Phase 1: Greedy initialization using bitmask scoring
    def greedy_init():
        covered = np.zeros(n_words, dtype=np.int64)
        solution = []
        remaining = n_t

        # For each block, precompute its mask
        # Greedy: pick block that covers most uncovered t-subsets
        # We'll use a heap with lazy deletion

        # Initial weights: popcount of block mask
        weights = np.zeros(n_k, dtype=np.int64)
        for j in range(n_k):
            weights[j] = popcount_mask(block_masks[j])

        heap = [(-weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)

        picked = np.zeros(n_k, dtype=bool)

        while remaining > 0:
            if time.time() > deadline - 30:
                break

            neg_w, j = heapq.heappop(heap)
            w = -neg_w

            if picked[j]:
                continue
            if w > weights[j] + 1e-9:
                continue
            if w <= 0:
                break

            # Select block j
            picked[j] = True
            solution.append(j)

            # Update covered
            new_bits = 0
            for wi in range(n_words):
                new_bits |= (block_masks[j][wi] & ~covered[wi])
                covered[wi] |= block_masks[j][wi]
            new_count = popcount_mask(np.array([block_masks[j][wi] & ~covered[wi] for wi in range(n_words)], dtype=np.int64))
            # Actually, let's just count the new bits
            new_count = 0
            for wi in range(n_words):
                new_count += np.bit_count(np.uint64(block_masks[j][wi] & ~covered[wi]))
            remaining -= new_count

            # Update weights of other blocks
            for wi in range(n_words):
                newly_covered = block_masks[j][wi] & ~covered[wi]  # wait, covered already updated
                # Let me redo: before update, covered_old. After, covered_new.
                # newly_covered = block_masks[j] & ~covered_old
                # For each other block b, weight decreases by popcount(block_masks[b] & newly_covered)

            # This is O(n_k * n_words) per step which might be slow.
            # Alternative: just recompute weights lazily.
            # For now, let's just push updated weights for blocks that share t-subsets.
            # Actually, for speed, let's just use a simpler approach:
            # Don't maintain heap precisely; just scan all blocks each iteration if n_k is manageable.
            # n_k = C(19,7) = 50388. That's too many to scan each time.

            # Let's use the heap but update more carefully.
            # For each block b not yet picked, its new weight = popcount(block_masks[b] & ~covered)
            # We can update this incrementally: when we add block j, for each b,
            # weight[b] -= popcount(block_masks[b] & block_masks[j] & ~covered_before)

            # This is still O(n_k * n_words). Let's just do it.
            for b in range(n_k):
                if not picked[b]:
                    delta = 0
                    for wi in range(n_words):
                        delta += np.bit_count(np.uint64(block_masks[b][wi] & block_masks[j][wi] & ~covered[wi]))
                    # Wait, covered is already updated. We need the old covered.
                    # Let me restructure.

        # This is getting complicated. Let me use a simpler greedy:
        # Just iterate, each time pick the block with max new coverage by scanning.
        # With n_k=50388 and maybe 150 iterations, that's 7.5M popcount ops. Might be too slow.

        # Alternative: Use the heap approach but with approximate weights.
        # Let me restart with a cleaner implementation.

        return solution

    # Let me implement a cleaner greedy with bitmask
    def greedy_init_v2():
        covered = np.zeros(n_words, dtype=np.int64)
        solution = []
        remaining = n_t
        picked = np.zeros(n_k, dtype=bool)

        # Precompute: for each block, the list of t-subset indices
        # We'll maintain a "gain" array: gain[j] = number of currently uncovered t-subsets in block j
        gain = np.zeros(n_k, dtype=np.int32)
        for j in range(n_k):
            gain[j] = len(block_tlist[j])

        while remaining > 0:
            if time.time() > deadline - 25:
                break

            # Find block with max gain
            best_j = -1
            best_gain = -1
            for j in range(n_k):
                if not picked[j] and gain[j] > best_gain:
                    best_gain = gain[j]
                    best_j = j

            if best_j == -1 or best_gain <= 0:
                break

            # Pick best_j
            picked[best_j] = True
            solution.append(best_j)

            # Update covered and gains
            for i in block_tlist[best_j]:
                if not (covered[i // 64] >> (i % 64)) & 1:
                    covered[i // 64] |= (np.int64(1) << np.int64(i % 64))
                    remaining -= 1
                    # Decrease gain for all blocks that also cover t-subset i
                    for b in t_to_blocks[i]:
                        if not picked[b]:
                            gain[b] -= 1

        return solution

    # Run greedy multiple times with different random orderings for diversity
    best_solution = None
    best_size = float('inf')

    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break

        # Shuffle the order of blocks to get different greedy results
        if restart == 0:
            # Standard greedy
            sol = greedy_init_v2()
        else:
            # Perturbed greedy: add small random noise to initial gains
            covered = np.zeros(n_words, dtype=np.int64)
            solution = []
            remaining = n_t
            picked = np.zeros(n_k, dtype=bool)

            gain = np.zeros(n_k, dtype=np.float64)
            for j in range(n_k):
                gain[j] = len(block_tlist[j]) + random.uniform(0, 1)

            while remaining > 0:
                if time.time() > deadline - 40:
                    break

                best_j = -1
                best_gain = -1
                for j in range(n_k):
                    if not picked[j] and gain[j] > best_gain:
                        best_gain = gain[j]
                        best_j = j

                if best_j == -1 or best_gain <= 0:
                    break

                picked[best_j] = True
                solution.append(best_j)

                for i in block_tlist[best_j]:
                    if not (covered[i // 64] >> (i % 64)) & 1:
                        covered[i // 64] |= (np.int64(1) << np.int64(i % 64))
                        remaining -= 1
                        for b in t_to_blocks[i]:
                            if not picked[b]:
                                gain[b] -= 1

            sol = solution

        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol.copy()

    if best_solution is None:
        # Fallback: just return first k blocks (shouldn't happen)
        return [list(k_subsets[i]) for i in range(min(151, n_k))]

    # Phase 2: Simulated Annealing
    # State: current solution (list of block indices)
    # Move: remove one block, add one block (swap), or just remove (if valid)
    # Objective: minimize |solution|, subject to all t-subsets covered

    current_solution = best_solution.copy()
    current_size = len(current_solution)
    best_sa_solution = current_solution.copy()
    best_sa_size = current_size

    # Precompute cover counts for current solution
    def compute_cover_counts(sol):
        counts = np.zeros(n_t, dtype=np.int32)
        for b in sol:
            for i in block_tlist[b]:
                counts[i] += 1
        return counts

    cover_counts = compute_cover_counts(current_solution)

    # SA parameters
    T_start = 5.0
    T_end = 0.01
    cooling = 0.995
    T = T_start
    max_iters = 5000
    iters_done = 0

    random.seed(42)

    while T > T_end and iters_done < max_iters:
        if time.time() > deadline - 10:
            break

        iters_done += 1

        # Choose a move:
        # 1. Remove a block (if it doesn't break coverage)
        # 2. Swap a block with another
        move_type = random.random()

        if move_type < 0.3 and len(current_solution) > 1:
            # Try to remove a random block
            idx = random.randrange(len(current_solution))
            b_remove = current_solution[idx]

            # Check if removing it keeps all t-subsets covered
            can_remove = True
            for i in block_tlist[b_remove]:
                if cover_counts[i] == 1:
                    can_remove = False
                    break

            if can_remove:
                # Remove it
                for i in block_tlist[b_remove]:
                    cover_counts[i] -= 1
                current_solution.pop(idx)
                current_size -= 1

                if current_size < best_sa_size:
                    best_sa_size = current_size
                    best_sa_solution = current_solution.copy()
                # Keep the removal (always accept improving moves)
            else:
                # Don't remove; try a swap instead
                pass

        # Swap: remove one block, add another
        if len(current_solution) > 0:
            idx = random.randrange(len(current_solution))
            b_old = current_solution[idx]

            # Find a candidate block to add
            # Prefer blocks that cover many currently singly-covered t-subsets
            # or blocks that are "complementary"
            # Simple: pick a random block not in current solution
            # To be smarter, pick from blocks that cover at least one t-subset
            # that is currently covered only by b_old

            # Find t-subsets that would become uncovered if we remove b_old
            critical = [i for i in block_tlist[b_old] if cover_counts[i] == 1]

            if critical:
                # Need a replacement that covers all critical t-subsets
                # Find blocks that cover all critical t-subsets
                # Start with blocks covering critical[0]
                candidates = set(t_to_blocks[critical[0]])
                candidates.discard(b_old)
                for ci in critical[1:]:
                    candidates.intersection_update(set(t_to_blocks[ci]))
                    if not candidates:
                        break

                if candidates:
                    # Score candidates: prefer ones that cover many other t-subsets
                    # (to make the solution more robust)
                    best_cand = -1
                    best_score = -1
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for cand in cand_list[:20]:
                        score = 0
                        for i in block_tlist[cand]:
                            if cover_counts[i] <= 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_cand = cand

                    if best_cand != -1:
                        # Perform swap: remove b_old, add best_cand
                        # Cost change: 0 (same size)
                        # But we need to check if it's a "good" swap
                        # Accept with probability based on temperature

                        # Compute delta "energy": we want to minimize number of
                        # singly-covered t-subsets (makes future removals easier)
                        old_critical_count = sum(1 for i in range(n_t) if cover_counts[i] == 1)

                        # Simulate swap
                        for i in block_tlist[b_old]:
                            cover_counts[i] -= 1
                        for i in block_tlist[best_cand]:
                            cover_counts[i] += 1

                        new_critical_count = sum(1 for i in range(n_t) if cover_counts[i] == 1)
                        delta = new_critical_count - old_critical_count

                        # Accept if delta <= 0 or with probability exp(-delta/T)
                        if delta <= 0 or random.random() < np.exp(-delta / T):
                            current_solution[idx] = best_cand
                            # Keep the change
                        else:
                            # Revert
                            for i in block_tlist[best_cand]:
                                cover_counts[i] -= 1
                            for i in block_tlist[b_old]:
                                cover_counts[i] += 1

            else:
                # No critical t-subsets, so we can just swap with a random block
                # to explore
                if len(current_solution) < n_k:
                    # Pick a random block not in solution
                    while True:
                        cand = random.randrange(n_k)
                        if cand not in current_solution:
                            break
                    # Swap
                    current_solution[idx] = cand
                    # Update cover counts
                    for i in block_tlist[b_old]:
                        cover_counts[i] -= 1
                    for i in block_tlist[cand]:
                        cover_counts[i] += 1

        # Cooling
        T *= cooling

        # Periodic deletion pass
        if iters_done % 200 == 0:
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                new_sol = []
                for b in current_solution:
                    can_remove = True
                    for i in block_tlist[b]:
                        if cover_counts[i] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for i in block_tlist[b]:
                            cover_counts[i] -= 1
                        changed = True
                    else:
                        new_sol.append(b)
                current_solution = new_sol
                current_size = len(current_solution)
                if current_size < best_sa_size:
                    best_sa_size = current_size
                    best_sa_solution = current_solution.copy()

    # Final deletion pass
    cover_counts = compute_cover_counts(best_sa_solution)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_sol = []
        for b in best_sa_solution:
            can_remove = True
            for i in block_tlist[b]:
                if cover_counts[i] == 1:
                    can_remove = False
                    break
            if can_remove:
                for i in block_tlist[b]:
                    cover_counts[i] -= 1
                changed = True
            else:
                new_sol.append(b)
        best_sa_solution = new_sol

    # Verify and return
    final_solution = best_sa_solution
    if len(final_solution) > best_size:
        final_solution = best_solution

    return [list(k_subsets[idx]) for idx in final_solution]
