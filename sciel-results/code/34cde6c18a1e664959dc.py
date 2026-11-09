"""
Covering design C(19,7,4) via randomized greedy with bitmask acceleration and local search.
"""
import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets and assign each an index
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {s: i for i, s in enumerate(all_tsubsets)}

    # Precompute for each point x, the list of t-subset indices that contain x
    # This is used to compute gains quickly
    # Actually, for bitmask approach, we represent uncovered as a bit vector
    # But num_tsubsets = C(19,4) = 3876, which fits in multiple 64-bit words
    # Let's use a Python int as a bit vector (arbitrary precision)

    # For each k-subset (block), we want to know which t-subsets it covers
    # Precompute: for each block, the bitmask of t-subsets it covers
    # There are C(19,7) = 50388 possible blocks, each with C(7,4)=35 t-subsets
    # We can precompute the bitmask for each possible block

    # First, map each t-subset to its bit position
    # We'll represent the uncovered set as a Python int where bit i is 1 if tsubset i is uncovered

    # Precompute block_mask: for each 7-element subset, the bitmask of t-subsets it covers
    # To save memory and time, we can compute on the fly, but let's precompute since 50388 is manageable

    # Actually, let's think about the greedy algorithm:
    # 1. Start with all t-subsets uncovered (all bits set)
    # 2. While uncovered:
    #    a. Pick a block that covers the most uncovered t-subsets
    #    b. Add it to the solution
    #    c. Clear those bits from uncovered

    # The key optimization: for a given partial block (set of points chosen so far),
    # we want to find the next point that maximizes the number of NEW t-subsets covered.

    # Alternative approach: randomized greedy with multiple restarts
    # For each restart:
    #   - Randomly order the t-subsets
    #   - Greedily add blocks that cover the first uncovered t-subset in the random order
    #   - Use local search to remove redundant blocks

    # Let's implement a fast bitmask-based greedy with randomization

    best_blocks = None
    best_count = float('inf')

    # Precompute: for each point x, the bitmask of t-subsets containing x
    # This helps in computing gains for adding point x to a partial block
    point_masks = {}
    for x in range(v):
        mask = 0
        for tsub in all_tsubsets:
            if x in tsub:
                idx = tsubset_to_idx[tsub]
                mask |= (1 << idx)
        point_masks[x] = mask

    # For a partial block P (list of points), the t-subsets covered by P are those
    # t-subsets that are subsets of P. We can compute this as:
    # For each t-subset s, s is covered by P iff all elements of s are in P.
    # With bitmask: we need to check if s's bits are a subset of P's "point set" bits.
    # This is tricky with the current representation.

    # Better approach: precompute for each possible block (k-subset) its coverage mask
    # C(19,7) = 50388 blocks, each mask is a Python int with up to 3876 bits
    # This is about 50388 * 485 bytes ≈ 24 MB, which is fine.

    block_to_idx = {}
    all_blocks = list(itertools.combinations(range(v), k))
    for i, block in enumerate(all_blocks):
        block_to_idx[block] = i
        num_blocks = len(all_blocks)

    # Precompute coverage masks for all blocks
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsubset_to_idx[tsub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Full mask: all t-subsets uncovered
    full_mask = (1 << num_tsubsets) - 1

    def time_check():
        return time.time() < deadline

    # Randomized greedy with multiple restarts
    rng = random.Random(42)

    for restart in range(200):  # up to 200 restarts, but stop at deadline
        if not time_check():
            break

        uncovered = full_mask
        blocks = []

        # Randomize the order in which we consider blocks
        # Instead of trying all blocks each step (slow), use a heuristic:
        # Pick a random uncovered t-subset, then find a good block containing it.

        while uncovered:
            if not time_check():
                break

            # Pick a random uncovered t-subset
            # Find a random bit that is set in uncovered
            # We can do this by picking a random index and checking if it's set,
            # or by finding the lowest set bit
            # For speed, let's pick a random position among the set bits
            # Actually, let's just pick the lowest set bit for simplicity, or randomize

            # To randomize, we can XOR with a random value and find the lowest set bit
            # Or simply: pick a random t-subset index and check if it's uncovered
            # This might take multiple tries, but on average it's fast

            # Better: maintain a list of uncovered t-subset indices, but that's expensive to update
            # Let's just pick the lowest set bit for now, and rely on restarts for diversity

            # Find lowest set bit
            # In Python, we can do:
            # lowbit = uncovered & (-uncovered)
            # idx = lowbit.bit_length() - 1
            # But this always picks the same one. Let's randomize by rotating.

            # Alternative: pick a random block and see if it covers any uncovered t-subsets
            # This is inefficient.

            # Let's use a different strategy: at each step, evaluate a random sample of blocks
            # and pick the one with the highest gain.

            # For speed, let's limit the candidate blocks to those that contain a specific
            # uncovered t-subset. Pick a random uncovered t-subset, then consider all blocks
            # that contain it.

            # Find a random uncovered t-subset:
            # We'll pick a random index and check if it's uncovered. If not, try again.
            # To make this faster, we can maintain a count of uncovered and use rejection sampling.

            # For now, let's just pick the lowest set bit for the first version.
            # We can improve this later.

            # Actually, let's randomize by picking a random offset and finding the next set bit.
            # This is complex. Let's just use the lowest set bit and rely on multiple restarts
            # with different random seeds to get diversity.

            # Hmm, but the lowest set bit will always be the same for a given uncovered set.
            # So all restarts will be identical. We need to randomize.

            # Solution: before each restart, shuffle the t-subset indices by permuting the bits.
            # This is expensive. Alternative: use a different random strategy.

            # Let's use a simpler approach: at each step, randomly sample N candidate blocks
            # and pick the one with the best gain.

            N_SAMPLES = 50
            best_gain = -1
            best_block = None

            # Sample random blocks
            for _ in range(N_SAMPLES):
                if not time_check():
                    break
                b_idx = rng.randint(0, num_blocks - 1)
                gain = bin(uncovered & block_masks[b_idx]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                if gain == 0:
                    continue

            if best_block is None or best_gain <= 0:
                # Fallback: find any block with positive gain
                # This should be rare
                for b_idx in range(num_blocks):
                    gain = bin(uncovered & block_masks[b_idx]).count('1')
                    if gain > 0:
                        best_block = b_idx
                        best_gain = gain
                        break

            if best_block is None:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~block_masks[best_block]

        if uncovered == 0 and len(blocks) < best_count:
            best_blocks = blocks[:]
            best_count = len(blocks)
            # Local search: try to remove redundant blocks
            if time_check():
                # Try removing each block and check if coverage is still complete
                for i in range(len(best_blocks)):
                    if not time_check():
                        break
                    # Temporarily remove block i
                    remaining = best_blocks[:i] + best_blocks[i+1:]
                    # Check if remaining covers all t-subsets
                    # Compute union of masks
                    union_mask = 0
                    for b in remaining:
                        b_idx = block_to_idx[b]
                        union_mask |= block_masks[b_idx]
                    if union_mask == full_mask:
                        # Block i is redundant, remove it
                        best_blocks = remaining
                        best_count = len(best_blocks)

    # Final local search on best_blocks
    if best_blocks and time_check():
        improved = True
        while improved and time_check():
            improved = False
            for i in range(len(best_blocks)):
                if not time_check():
                    break
                remaining = best_blocks[:i] + best_blocks[i+1:]
                union_mask = 0
                for b in remaining:
                    b_idx = block_to_idx[b]
                    union_mask |= block_masks[b_idx]
                if union_mask == full_mask:
                    best_blocks = remaining
                    best_count = len(best_blocks)
                    improved = True
                    break

    if best_blocks is None:
        # Fallback: return a valid (but possibly large) covering
        # Use a simple deterministic greedy
        uncovered = full_mask
        blocks = []
        while uncovered:
            best_gain = -1
            best_block = None
            for b_idx in range(num_blocks):
                gain = bin(uncovered & block_masks[b_idx]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block is None:
                break
            blocks.append(all_blocks[best_block])
            uncovered &= ~block_masks[best_block]
        return blocks

    # Convert blocks to lists of ints (they're already tuples from itertools)
    return [list(b) for b in best_blocks]
