import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # 1. Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # 3. For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)

    # 4. Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i in range(n_k):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # 5. Precompute block sizes (number of t-subsets each block covers)
    block_sizes = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)

    # 5. Greedy set cover using cover_count for efficiency
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        selected = []

        for _ in range(n_k):
            if not uncovered.any():
                break
            if time.time() > deadline:
                break

            # Add small noise to break ties for randomization
            if seed > 0:
                noise = np.array([rng.random() * 0.01 for _ in range(n_k)], dtype=np.float64)
            else:
                noise = 0.0

            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))

            if scores[best_idx] < 0.5:
                break

            selected.append(best_idx)

            # Update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1

        return selected

    # 6. Efficient deletion using cover_count
    def reduce_blocks(blocks_idx, max_time=None):
        if not blocks_idx:
            return blocks_idx
        if max_time is None:
            max_time = deadline

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < max_time:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                # Check if any t-subset is only covered by this block
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    # 6b. Try to replace 2 blocks with 1 block that covers everything they covered
    def try_two_for_one(blocks_idx, cover_count, max_time=None):
        if max_time is None:
            max_time = deadline
        if len(blocks_idx) < 2:
            return blocks_idx, cover_count

        # Build a set for fast lookup
        block_set = set(blocks_idx)

        # For each pair of blocks, check if there's a single block covering their union
        # This is expensive, so only do it for promising pairs
        n_blocks = len(blocks_idx)
        improved = False
        max_pairs = min(n_blocks * (n_blocks - 1) // 2, 5000)
        rng = random.Random(42)

        # Randomly sample pairs to try
        pairs_tried = 0
        for _ in range(max_pairs):
            if time.time() > max_time:
                break
            i = rng.randint(0, n_blocks - 1)
            j = rng.randint(0, n_blocks - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            b1 = blocks_idx[i]
            b2 = blocks_idx[j]

            # Get the union of t-subsets covered by b1 and b2
            # A replacement block must cover all t-subsets that are ONLY covered by b1 or b2
            # (i.e., t-subsets where cover_count == 1 and it's in b1 or b2)
            critical_ts = set()
            for t_idx in k_sub_t_idxs[b1]:
                if cover_count[t_idx] == 1:
                    critical_ts.add(t_idx)
            for t_idx in k_sub_t_idxs[b2]:
                if cover_count[t_idx] == 1:
                    critical_ts.add(t_idx)

            if not critical_ts:
                continue

            # Find a block that covers all critical t-subsets
            # Use the reverse mapping: for each critical t-subset, which blocks cover it?
            # The intersection of these sets gives candidate blocks
            candidates = None
            for t_idx in critical_ts:
                covering = set(t_sub_to_k_idxs[t_idx])
                if candidates is None:
                    candidates = covering
                else:
                    candidates &= covering
                if not candidates:
                    break

            if not candidates:
                continue

            # Pick a candidate not in current blocks (or any)
            for cand in candidates:
                if cand not in block_set:
                    # Replace b1 and b2 with cand
                    for t_idx in k_sub_t_idxs[b1]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[b2]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[cand]:
                        cover_count[t_idx] += 1
                    new_blocks = [b for b in blocks_idx if b != b1 and b != b2]
                    new_blocks.append(cand)
                    blocks_idx = new_blocks
                    block_set = set(blocks_idx)
                    improved = True
                    break

            if improved:
                break

        return blocks_idx, cover_count

    # 7. Simulated Annealing for swap-based local search
    def simulated_annealing(initial_blocks, max_time):
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        n_uncovered = int(np.sum(cover_count == 0))
        current_energy = len(current_blocks) + 1000.0 * n_uncovered
        best_blocks = list(current_blocks)
        best_energy = current_energy

        T = 50.0
        cooling_rate = 0.99
        min_T = 1.0
        penalty = 1000.0
        rng = random.Random(12345)

        # Track which blocks are currently in use for fast membership check
        current_block_set = set(current_blocks)

        while time.time() < max_time and T > min_T:
            move_type = rng.random()

            if move_type < 0.3:
                # Move 1: Try to remove a random block
                if len(current_blocks) > 0:
                    i = rng.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[i]

                    # Count how many t-subsets would become uncovered
                    new_uncovered = 0
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            new_uncovered += 1

                    if new_uncovered == 0:
                        # Valid removal - always accept
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_block_set.discard(b_idx)
                        current_energy -= 1.0
                    else:
                        # Invalid removal - accept with probability
                        delta_E = 1.0 + penalty * new_uncovered
                        if rng.random() < np.exp(-delta_E / T):
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(i)
                            current_block_set.discard(b_idx)
                            current_energy += delta_E

            elif move_type < 0.7:
                # Move 2: Try to swap a block with another (smarter candidate selection)
                if len(current_blocks) > 0:
                    i = rng.randint(0, len(current_blocks) - 1)
                    orig_idx = current_blocks[i]

                    # Find t-subsets that are only covered by orig_idx
                    critical_for_orig = []
                    for t_idx in k_sub_t_idxs[orig_idx]:
                        if cover_count[t_idx] == 1:
                            critical_for_orig.append(t_idx)

                    if critical_for_orig:
                        # We need a replacement that covers all critical t-subsets
                        # Find candidates via intersection
                        candidates = None
                        for t_idx in critical_for_orig[:3]:  # Limit to first 3 for speed
                            covering = set(t_sub_to_k_idxs[t_idx])
                            if candidates is None:
                                candidates = covering
                            else:
                                candidates &= covering
                            if not candidates:
                                break

                        if candidates:
                            # Remove blocks already in current set
                            candidates -= current_block_set
                            if candidates:
                                new_idx = rng.choice(list(candidates))
                            else:
                                T *= cooling_rate
                                continue
                        else:
                            T *= cooling_rate
                            continue
                    else:
                        # No critical t-subsets, pick any block not in current set
                        # Sample a few random blocks and pick one not in current set
                        new_idx = None
                        for _ in range(10):
                            candidate = rng.randint(0, n_k - 1)
                            if candidate not in current_block_set:
                                new_idx = candidate
                                break
                        if new_idx is None:
                            T *= cooling_rate
                            continue

                    if new_idx == orig_idx:
                        T *= cooling_rate
                        continue

                    # Compute energy change for the swap
                    removed_uncovered = 0
                    for t_idx in k_sub_t_idxs[orig_idx]:
                        if cover_count[t_idx] == 1:
                            removed_uncovered += 1

                    added_covered = 0
                    for t_idx in k_sub_t_idxs[new_idx]:
                        if cover_count[t_idx] == 0:
                            added_covered += 1

                    delta_uncovered = removed_uncovered - added_covered
                    delta_E = penalty * delta_uncovered

                    if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                        # Perform the swap
                        for t_idx in k_sub_t_idxs[orig_idx]:
                            cover_count[t_idx] -= 1
                        for t_idx in k_sub_t_idxs[new_idx]:
                            cover_count[t_idx] += 1
                        current_blocks[i] = new_idx
                        current_block_set.discard(orig_idx)
                        current_block_set.add(new_idx)
                        current_energy += delta_E

            else:
                # Move 3: Try 2-for-1 exchange
                if len(current_blocks) >= 2:
                    current_blocks, cover_count = try_two_for_one(
                        current_blocks, cover_count, max_time
                    )
                    current_block_set = set(current_blocks)
                    current_energy = len(current_blocks) + penalty * int(np.sum(cover_count == 0))

            # Periodically try to reduce blocks
            if rng.random() < 0.05:
                new_blocks = reduce_blocks(current_blocks, max_time)
                if len(new_blocks) < len(current_blocks):
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx in new_blocks:
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] += 1
                    current_blocks = new_blocks
                    current_block_set = set(current_blocks)
                    current_energy = len(current_blocks) + penalty * int(np.sum(cover_count == 0))

            # Track best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            T *= cooling_rate

        return best_blocks

    # 8. Main algorithm
    # Step 1: Greedy cover
    selected = greedy_cover(seed=0)

    # Step 2: Delete redundant blocks
    selected = reduce_blocks(selected)

    # Step 3: Simulated Annealing to improve (use most of the time budget)
    if time.time() < deadline - 10:
        sa_time = min(100.0, deadline - time.time() - 5)
        selected = simulated_annealing(selected, start_time + sa_time)

    # Step 4: Final deletion pass
    selected = reduce_blocks(selected)

    # Step 5: Try a few more greedy restarts with different seeds if time permits
    if time.time() < deadline - 20:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            greedy_sel = greedy_cover(seed=seed)
            greedy_sel = reduce_blocks(greedy_sel)
            if len(greedy_sel) < len(selected):
                selected = greedy_sel
                # Also run SA on the better solution
                if time.time() < deadline - 10:
                    sa_time2 = min(30.0, deadline - time.time() - 5)
                    selected = simulated_annealing(selected, start_time + sa_time2)
                    selected = reduce_blocks(selected)

    # Final cleanup
    selected = reduce_blocks(selected)

    return [list(all_k_subs[i]) for i in selected]
