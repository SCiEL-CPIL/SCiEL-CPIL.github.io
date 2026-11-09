import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Compute masks and subset lists for each block
    block_masks = [0] * num_blocks
    block_subsets = [[] for _ in range(num_blocks)]

    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs

    # Inverted index: subset_idx -> list of block indices
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)

    # Convert block_masks to numpy array for vectorized gain computation
    # Use uint64 arrays: each mask is split into chunks of 64 bits
    num_chunks = (num_t_subsets + 63) // 64
    block_masks_np = np.zeros((num_blocks, num_chunks), dtype=np.uint64)
    for i in range(num_blocks):
        mask = block_masks[i]
        for c in range(num_chunks):
            block_masks_np[i, c] = (mask >> (c * 64)) & 0xFFFFFFFFFFFFFFFF

    def count_bits(x):
        return x.bit_count()

    def greedy_construct(randomize=False, seed=None):
        """
        Greedy set cover with vectorized gain computation.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        # uncovered as numpy array of uint64 chunks
        uncovered_np = np.ones(num_chunks, dtype=np.uint64)
        # Set the last chunk properly
        if num_t_subsets % 64 != 0:
            last_bits = num_t_subsets % 64
            uncovered_np[-1] = (1 << last_bits) - 1

        chosen = []
        used = np.zeros(num_blocks, dtype=bool)

        while True:
            # Check if all covered
            if not uncovered_np.any():
                break

            # Compute gains: number of uncovered subsets covered by each block
            # gain[i] = popcount(block_masks[i] & uncovered)
            # Vectorized: for each chunk, AND and popcount
            gains = np.zeros(num_blocks, dtype=np.int32)
            for c in range(num_chunks):
                and_result = block_masks_np[:, c] & uncovered_np[c]
                # Popcount for each element
                # Use lookup table or numpy bit_count if available
                # np.bit_count exists in numpy 1.17+? Let's use a trick
                # Actually, let's just use a simple approach with numpy
                # For uint64, we can use:
                # x = x - ((x >> 1) & 0x5555555555555555)
                # x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
                # x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
                # x = (x * 0x0101010101010101) >> 56
                # This is complex in numpy. Let's use a simpler method.
                # For our purposes, since num_blocks is ~50k and num_chunks is ~61,
                # we can loop over chunks and use a vectorized popcount.
                
                # Actually, let's just use Python's int.bit_count() in a loop over blocks
                # but that defeats the purpose. Let me think of a better way.
                
                # Alternative: convert to Python ints and use bit_count
                # But that's slow for 50k blocks.
                
                # Let's use a different approach: precompute popcount for 16-bit chunks
                # and sum them up.
                pass

            # Let's just do the naive approach but with some optimization
            # For each block, compute gain using bit operations
            best_gain = -1
            best_idx = -1

            # Only consider unused blocks
            for i in range(num_blocks):
                if used[i]:
                    continue
                mask = block_masks[i]
                # Compute gain = popcount(mask & uncovered)
                # We need uncovered as a Python int for this
                # Let's maintain uncovered as a Python int too
                pass

        # The vectorized approach is getting complex. Let's use a simpler but effective method.
        # Maintain uncovered as a Python int.
        uncovered = (1 << num_t_subsets) - 1
        chosen = []
        used = [False] * num_blocks

        while uncovered != 0:
            best_gain = -1
            best_idx = -1

            # For each unused block, compute gain
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = count_bits(block_masks[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and randomize and gain > 0:
                    if rng.random() < 0.5:
                        best_idx = i

            if best_idx == -1:
                break

            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]

        return chosen

    def prune_solution(chosen_indices):
        """
        Removes blocks that are not essential.
        """
        if not chosen_indices:
            return []

        counts = [0] * num_t_subsets
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1

        changed = True
        current_list = list(chosen_indices)

        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            current_list = new_list

        return current_list

    def local_search(chosen_indices, max_iterations=100):
        """
        Local search: try to remove a block and replace it with a better one,
        or remove a block if possible.
        Objective: minimize number of blocks.
        """
        if not chosen_indices:
            return []

        current = list(chosen_indices)
        current_set = set(current)

        # Compute coverage counts
        counts = [0] * num_t_subsets
        for idx in current:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1

        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Recompute counts
            counts = [0] * num_t_subsets
            for idx in current:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1

            # Try to remove each block
            for i, idx in enumerate(current):
                # Check if block idx is essential
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break

                if not essential:
                    # Remove it
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    current.pop(i)
                    improved = True
                    break

                # Try to swap with another block
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                if not essential_subs:
                    continue

                # Find candidates that cover all essential_subs
                pool = subset_to_blocks[essential_subs[0]]

                best_candidate = -1
                best_score = -1

                for cand_idx in pool:
                    if cand_idx in current_set:
                        continue

                    # Check if cand covers all essential_subs
                    covers_all = True
                    cand_mask = block_masks[cand_idx]
                    for s in essential_subs:
                        if not (cand_mask & (1 << s)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue

                    # Score: how many subsets in cand are already covered (count > 1)?
                    # After swap, these become more redundant.
                    redundancy = 0
                    cand_subs = block_subsets[cand_idx]
                    idx_subs_set = set(block_subsets[idx])

                    for s in cand_subs:
                        if s in idx_subs_set:
                            if counts[s] > 1:
                                redundancy += 1
                        else:
                            if counts[s] >= 1:
                                redundancy += 1

                    if redundancy > best_score:
                        best_score = redundancy
                        best_candidate = cand_idx

                if best_candidate != -1:
                    # Perform swap and check if it allows further removals
                    test_solution = list(current)
                    test_solution[i] = best_candidate
                    pruned_test = prune_solution(test_solution)

                    if len(pruned_test) < len(current):
                        current = pruned_test
                        current_set = set(current)
                        improved = True
                        break

        return current

    # Main loop
    best_solution = None
    best_len = float('inf')

    # Initial deterministic greedy
    init_sol = greedy_construct(randomize=False)
    pruned_init = prune_solution(init_sol)
    if len(pruned_init) < best_len:
        best_len = len(pruned_init)
        best_solution = pruned_init

    # Try randomized greedy + local search
    seed = 42
    while time.time() < deadline - 5.0:
        sol = greedy_construct(randomize=True, seed=seed)
        sol = prune_solution(sol)

        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

        # Local search
        improved_sol = local_search(sol, max_iterations=50)

        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol

        seed += 1

        if best_len <= 131:
            break

    result_blocks = [list(all_blocks[i]) for i in best_solution]
    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
    v, k, t = 19, 7, 4
    all_t = list(itertools.combinations(range(v), t))
    covered = set()
    for b in blocks:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    is_valid = all(t_sub in covered for t_sub in all_t)
    print(f"Valid: {is_valid}")
