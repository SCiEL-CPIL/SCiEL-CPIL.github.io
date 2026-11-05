import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

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

    # 5. Greedy set cover using cover_count for efficiency
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        selected = []

        for _ in range(n_k):
            if not uncovered.any():
                break
            if time.time() > deadline - 5:
                break

            # Add small noise to break ties for randomization
            if seed > 0:
                noise = rng.random() * 0.01
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

    # 7. Repair function: given a set of blocks and uncovered mask, greedily cover remaining
    def repair(blocks_idx, uncovered_mask, seed=0):
        if not np.any(uncovered_mask):
            return list(blocks_idx)
        
        rng = random.Random(seed)
        current_blocks = list(blocks_idx)
        # Recalculate gains based on uncovered_mask
        gains = np.zeros(n_k, dtype=np.int32)
        for t_idx in np.where(uncovered_mask)[0]:
            for b_idx in t_sub_to_k_idxs[t_idx]:
                gains[b_idx] += 1
        
        while True:
            if time.time() > deadline - 5:
                break
            if not np.any(uncovered_mask):
                break
            if gains.max() == 0:
                break
            
            noise = rng.random() * 0.1
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
            
            current_blocks.append(best_idx)
            
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
        
        return current_blocks

    # 8. Simulated Annealing for swap-based local search
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

        T = 30.0
        cooling_rate = 0.985
        min_T = 0.5
        penalty = 1000.0
        rng = random.Random(12345)

        # Precompute block sizes for quick reference
        block_sizes = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)

        while time.time() < max_time and T > min_T:
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
                    current_energy -= 1.0
                else:
                    # Invalid removal - accept with probability
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E

            # Move 2: Try to swap a block with another
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                orig_idx = current_blocks[i]

                # Pick a random candidate block to swap in
                new_idx = rng.randint(0, n_k - 1)
                if new_idx == orig_idx:
                    T *= cooling_rate
                    continue

                # Check if new_idx is already in current_blocks
                if new_idx in current_blocks:
                    T *= cooling_rate
                    continue

                # Compute energy change for the swap
                # Remove orig_idx: how many t-subsets become uncovered?
                removed_uncovered = 0
                for t_idx in k_sub_t_idxs[orig_idx]:
                    if cover_count[t_idx] == 1:
                        removed_uncovered += 1

                # Add new_idx: how many newly covered t-subsets?
                added_covered = 0
                for t_idx in k_sub_t_idxs[new_idx]:
                    if cover_count[t_idx] == 0:
                        added_covered += 1

                # Net change in uncovered count
                # When we remove orig_idx first, some t-subsets may become uncovered
                # Then when we add new_idx, some may become covered again
                # This is approximate but good enough for SA

                delta_uncovered = removed_uncovered - added_covered
                delta_E = penalty * delta_uncovered  # Size stays the same in a swap

                # Actually, in a swap the size doesn't change, so:
                delta_E = penalty * delta_uncovered

                if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                    # Perform the swap
                    for t_idx in k_sub_t_idxs[orig_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_idx]:
                        cover_count[t_idx] += 1
                    current_blocks[i] = new_idx
                    current_energy += delta_E

                    # After swap, try to delete redundant blocks
                    new_blocks = reduce_blocks(current_blocks, max_time)
                    if len(new_blocks) < len(current_blocks):
                        # Update cover_count for the reduced set
                        cover_count = np.zeros(n_t, dtype=np.int32)
                        for b_idx in new_blocks:
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] += 1
                        current_blocks = new_blocks
                        current_energy = len(current_blocks) + penalty * int(np.sum(cover_count == 0))

            # Track best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            T *= cooling_rate

        return best_blocks

    # 9. Main algorithm
    # Step 1: Greedy cover
    selected = greedy_cover(seed=0)

    # Step 2: Delete redundant blocks
    selected = reduce_blocks(selected)

    best_selected = list(selected)
    best_len = len(selected)

    # Step 3: Simulated Annealing to improve
    if time.time() < deadline - 20:
        sa_time = min(50.0, deadline - time.time() - 10)
        sa_start = time.time()
        sa_selected = simulated_annealing(selected, sa_start + sa_time)
        sa_selected = reduce_blocks(sa_selected)
        if len(sa_selected) < best_len:
            best_len = len(sa_selected)
            best_selected = sa_selected

    # Step 4: Multi-restart with perturbation and repair (Inspiration 1 strategy)
    # We will try several restarts. For each:
    # 1. Start from best_selected
    # 2. Remove some random blocks (perturbation)
    # 3. Repair uncovered subsets
    # 4. Reduce
    # 5. If better, update best

    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Determine how many blocks to remove
        # Start with small removals, increase slightly
        num_remove = min(2 + r // 2, len(best_selected) - 1)
        if num_remove <= 0:
            continue
            
        rng_restart = random.Random(1000 + r)
        # Pick indices to remove
        remove_indices = rng_restart.sample(range(len(best_selected)), num_remove)
        remove_set = set(best_selected[i] for i in remove_indices)
        
        # Create perturbed set
        perturbed = [b for b in best_selected if b not in remove_set]
        
        # Calculate uncovered mask
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in perturbed:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        uncovered_mask = (cover_count == 0).astype(bool)
        
        # Repair
        repaired = repair(perturbed, uncovered_mask, seed=1000 + r)
        
        # Reduce
        reduced = reduce_blocks(repaired)
        
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_selected = reduced

    # Step 5: Try a few more greedy restarts with different seeds if time permits
    if time.time() < deadline - 15:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            greedy_sel = greedy_cover(seed=seed)
            greedy_sel = reduce_blocks(greedy_sel)
            if len(greedy_sel) < best_len:
                best_len = len(greedy_sel)
                best_selected = greedy_sel

    # Final cleanup
    best_selected = reduce_blocks(best_selected)

    return [list(all_k_subs[i]) for i in best_selected]
