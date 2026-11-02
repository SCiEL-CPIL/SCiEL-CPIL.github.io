"""Covering design C(19,8,5): bitmask-based greedy set cover with ILP removal and simulated annealing."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Precompute all k-blocks and their coverage bitmasks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << t_index_map[combo])
        block_masks[idx] = mask

    full_mask = (1 << num_t) - 1

    def _greedy_pass(seed, randomize_ties=False):
        """One greedy set cover pass with optional randomization for diversity."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        while uncovered:
            if time.time() > deadline - 3.0:
                return None

            # Find block with max coverage of uncovered subsets
            best_idx = -1
            best_count = -1
            best_indices = []

            indices = list(range(num_k))
            if randomize_ties:
                rng.shuffle(indices)

            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
                    best_indices = [i]
                elif gain == best_count and gain > 0:
                    best_indices.append(i)

            if best_idx == -1 or best_count == 0:
                break

            # Random tie-breaking for diversity
            if randomize_ties and len(best_indices) > 1:
                best_idx = rng.choice(best_indices)

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    def _remove_redundant(selected_indices):
        """Remove redundant blocks using bitmask comparisons."""
        if not selected_indices:
            return selected_indices

        n = len(selected_indices)
        kept = list(selected_indices)
        changed = True

        while changed and time.time() < deadline - 2.0:
            changed = False
            new_kept = []
            for i in range(len(kept)):
                idx = kept[i]
                # Compute mask of all OTHER blocks
                others_mask = 0
                for j in range(len(kept)):
                    if j != i:
                        others_mask |= block_masks[kept[j]]

                # If all t-subsets covered by this block are also covered by others
                if (block_masks[idx] & ~others_mask) == 0:
                    changed = True
                else:
                    new_kept.append(idx)

            if changed:
                kept = new_kept

        return kept

    def _ilp_block_removal(selected_indices):
        """ILP-style: greedily remove blocks that don't leave any t-subset uncovered."""
        if not selected_indices:
            return selected_indices

        kept = list(selected_indices)
        changed = True

        while changed and time.time() < deadline - 2.0:
            changed = False
            best_block_idx = -1
            best_unique = float('inf')

            for i in range(len(kept)):
                idx = kept[i]
                # Count how many t-subsets are uniquely covered by this block
                others_mask = 0
                for j in range(len(kept)):
                    if j != i:
                        others_mask |= block_masks[kept[j]]

                unique_mask = block_masks[idx] & ~others_mask
                unique_count = unique_mask.bit_count()

                if unique_count < best_unique:
                    best_unique = unique_count
                    best_block_idx = i

                if unique_count == 0:
                    break

            if best_block_idx == -1 or best_unique > 0:
                break

            # Remove the block
            kept.pop(best_block_idx)
            changed = True

        return kept

    def _simulated_annealing(selected_indices, seed):
        """Simulated annealing: try swaps and drops to escape local minima."""
        if not selected_indices or len(selected_indices) <= 1:
            return selected_indices

        rng = random.Random(seed)
        blocks_list = list(selected_indices)
        current_len = len(blocks_list)
        best_blocks = list(blocks_list)
        best_len = current_len

        # Build current coverage mask
        cov_mask = 0
        for idx in blocks_list:
            cov_mask |= block_masks[idx]

        # Annealing parameters
        T_start = 5.0
        T_end = 0.1
        num_iters = 3000
        cooling_rate = (T_end / T_start) ** (1.0 / max(num_iters, 1))
        T = T_start

        for it in range(num_iters):
            if time.time() > deadline - 5.0:
                break

            move_type = rng.random()

            if move_type < 0.6 and len(blocks_list) > 1:
                # Swap: replace one block with a random new block
                i = rng.randint(0, len(blocks_list) - 1)
                old_block = blocks_list[i]
                new_block = rng.randint(0, num_k - 1)

                if new_block == old_block:
                    continue

                # Compute new coverage if we swap
                # Remove old block's contribution
                # We need to check if removing old_block leaves any gap
                # Compute mask without old_block
                without_old = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        without_old |= block_masks[blocks_list[j]]

                # Check if old_block covers anything not covered by others
                unique_to_old = block_masks[old_block] & ~without_old

                # If we swap, new coverage = without_old | block_masks[new_block]
                new_cov = without_old | block_masks[new_block]

                # Valid if new_cov covers everything old_cov covered
                # old_cov = without_old | block_masks[old_block]
                old_cov = without_old | block_masks[old_block]

                if (new_cov & ~old_cov) == 0:
                    # New block doesn't cover anything old didn't, but might miss some
                    # Actually we need new_cov to cover at least what old_cov covered
                    # i.e., old_cov & ~new_cov should be 0
                    if (old_cov & ~new_cov) == 0:
                        # Valid swap
                        blocks_list[i] = new_block
                        cov_mask = new_cov
                        if len(blocks_list) < best_len:
                            best_blocks = list(blocks_list)
                            best_len = len(blocks_list)
                    else:
                        # Invalid: new block misses some coverage
                        pass
                else:
                    # New block covers more than old, always accept
                    blocks_list[i] = new_block
                    cov_mask = new_cov
                    if len(blocks_list) < best_len:
                        best_blocks = list(blocks_list)
                        best_len = len(blocks_list)

            elif move_type < 0.85 and len(blocks_list) > 1:
                # Drop: try to remove a block
                i = rng.randint(0, len(blocks_list) - 1)
                old_block = blocks_list[i]

                # Compute mask without this block
                without_old = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        without_old |= block_masks[blocks_list[j]]

                # Check if removal is valid
                if (block_masks[old_block] & ~without_old) == 0:
                    # Valid drop
                    blocks_list.pop(i)
                    current_len -= 1
                    cov_mask = without_old
                    if current_len < best_len:
                        best_blocks = list(blocks_list)
                        best_len = current_len
                else:
                    # Invalid drop: could accept with probability exp(-delta/T)
                    # delta = number of uniquely covered t-subsets
                    unique_count = (block_masks[old_block] & ~without_old).bit_count()
                    delta = unique_count * 0.5
                    if rng.random() < pow(2.718281828, -delta / T):
                        # Accept invalid move (temporarily)
                        blocks_list.pop(i)
                        current_len -= 1
                        cov_mask = without_old
                        if current_len < best_len:
                            best_blocks = list(blocks_list)
                            best_len = current_len

            else:
                # Add: add a random block to escape local minima
                if len(blocks_list) < 450:
                    new_block = rng.randint(0, num_k - 1)
                    if new_block not in blocks_list:
                        blocks_list.append(new_block)
                        current_len += 1
                        cov_mask |= block_masks[new_block]

            # Cool down
            T *= cooling_rate

        return best_blocks

    # Main search: multiple restarts with different strategies
    best_blocks = None
    best_len = float('inf')

    # Strategy 1: Pure greedy (no randomization)
    if time.time() < deadline - 20.0:
        selected = _greedy_pass(0, randomize_ties=False)
        if selected:
            selected = _remove_redundant(selected)
            selected = _ilp_block_removal(selected)
            if len(selected) < best_len:
                best_len = len(selected)
                best_blocks = list(selected)

    # Strategy 2: Randomized greedy + SA
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        seed = 100 + restart * 17
        selected = _greedy_pass(seed, randomize_ties=True)
        if not selected:
            continue

        selected = _remove_redundant(selected)
        selected = _ilp_block_removal(selected)

        # Apply SA if time allows
        if time.time() < deadline - 10.0:
            selected = _simulated_annealing(selected, seed + 500)
            selected = _remove_redundant(selected)
            selected = _ilp_block_removal(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = list(selected)

    # Strategy 3: More aggressive SA on the best solution
    if best_blocks and time.time() < deadline - 8.0:
        polished = _simulated_annealing(best_blocks, 999)
        polished = _remove_redundant(polished)
        polished = _ilp_block_removal(polished)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = list(polished)

    # Final cleanup pass
    if best_blocks and time.time() < deadline - 3.0:
        best_blocks = _remove_redundant(best_blocks)
        best_blocks = _ilp_block_removal(best_blocks)

    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_pass(0, randomize_ties=False)
        if best_blocks is None:
            best_blocks = []

    return [list(k_blocks[i]) for i in best_blocks]
# EVOLVE-BLOCK-END
