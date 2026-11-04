import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)  # 11628

    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)  # 75582

    # Precompute t-subsets for each block
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # Inverse: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)

    # NumPy arrays for fast operations
    # block_tsubs_np: shape (num_blocks, 56), int32
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)
    # tsub_blocks_np: for each t-subset, list of block indices
    # We'll use this as a list of numpy arrays
    tsub_blocks_np = [np.array(tb, dtype=np.int32) for tb in tsub_blocks]

    # Deadline
    start_time = time.time()
    deadline = start_time + 145.0

    def get_elapsed():
        return time.time() - start_time

    def check_time(budget=10.0):
        return get_elapsed() > deadline - budget

    # Fast greedy using NumPy
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random

        # uncovered: boolean array of size num_tsubs
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        selected_blocks = []

        while uncovered_count > 0:
            if check_time(15.0):
                return None

            # Pick a random uncovered t-subset
            # For efficiency, if uncovered_count is small, build a list
            if uncovered_count < 500:
                uncov_indices = np.where(uncovered == 1)[0]
                if len(uncov_indices) == 0:
                    break
                s_idx = int(rng.choice(uncov_indices))
            else:
                while True:
                    s_idx = rng.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break

            # Get candidate blocks
            candidates = tsub_blocks_np[s_idx]

            # Compute gain for each candidate block
            # gain = number of uncovered t-subsets in the block
            # Use NumPy: for each candidate, count uncovered in its t-subsets
            best_block_idx = -1
            best_gain = -1

            # Batch process candidates
            # For each candidate block, get its t-subsets and count uncovered
            # This is a gather operation
            if len(candidates) > 0:
                # Get all t-subsets for all candidate blocks
                # candidates shape: (num_candidates,)
                # block_tsubs_np[candidates] shape: (num_candidates, 56)
                cand_tsubs = block_tsubs_np[candidates]  # (num_candidates, 56)
                # uncovered[cand_tsubs] shape: (num_candidates, 56)
                # Sum along axis 1
                gains = np.sum(uncovered[cand_tsubs], axis=1)

                # Find best
                best_local = int(np.argmax(gains))
                best_gain = int(gains[best_local])
                best_block_idx = int(candidates[best_local])

            if best_block_idx == -1:
                break

            selected_blocks.append(best_block_idx)

            # Update uncovered
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1

        if uncovered_count > 0:
            return None

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Reduce the number of blocks while maintaining coverage.
        Multiple passes:
        1. Remove redundant blocks.
        2. Try swaps that enable removals.
        """
        block_indices = list(block_indices)

        # Build cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs_np[b_idx]] += 1

        changed = True
        max_passes = 20
        for _ in range(max_passes):
            if check_time(10.0):
                break

            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                # Check if any t-subset in this block has cover_count == 1
                # i.e., this block is the only one covering some t-subset
                b_tsubs = block_tsubs_np[b_idx]
                if np.any(cover_count[b_tsubs] == 1):
                    new_block_indices.append(b_idx)
                else:
                    # Remove it
                    changed = True
                    cover_count[b_tsubs] -= 1

            block_indices = new_block_indices

            if not changed:
                break

        # Try swaps: replace a block with another that enables a removal
        # This is more complex. For now, do a few rounds of:
        # For each block, try replacing it with a block that covers more of the
        # "critical" t-subsets, then re-run reduction.
        # Given time constraints, we'll do a simple version.

        return block_indices

    # Main loop: multiple restarts
    best_len = float('inf')
    best_block_list = []

    # First, run a deterministic greedy to get a baseline
    initial = greedy_construct(random_seed=42)
    if initial is not None:
        reduced = reduce_blocks(initial)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

    # Now, run randomized restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if check_time(15.0):
            break

        seed = restart * 1000 + 7
        selected = greedy_construct(random_seed=seed)
        if selected is None:
            continue

        reduced = reduce_blocks(selected)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

            # Early exit if we hit a very good number
            if best_len <= 325:
                break

    # If we still have time, do more aggressive reduction on the best solution
    if best_block_list and not check_time(30.0):
        # Try to further reduce the best solution
        # Build cover count and do more aggressive removal
        best_indices = [all_blocks.index(b) for b in best_block_list]
        # Actually, this is slow. Let's just do a few more reduction passes.
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in best_indices:
            cover_count[block_tsubs_np[b_idx]] += 1

        for _ in range(50):
            if check_time(10.0):
                break
            changed = False
            new_indices = []
            for b_idx in best_indices:
                b_tsubs = block_tsubs_np[b_idx]
                if np.any(cover_count[b_tsubs] == 1):
                    new_indices.append(b_idx)
                else:
                    changed = True
                    cover_count[b_tsubs] -= 1
            best_indices = new_indices
            if not changed:
                break

        if len(best_indices) < best_len:
            best_len = len(best_indices)
            best_block_list = [all_blocks[i] for i in best_indices]

    # Return the best solution
    return [list(b) for b in best_block_list]
