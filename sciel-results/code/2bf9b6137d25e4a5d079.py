import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all C(v, t) triples and assign each an index 0..N-1
    triples = list(itertools.combinations(range(v), t))
    N = len(triples)
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}

    # Precompute, for each point x, the list of triple-indices that contain x
    point_triples = [[] for _ in range(v)]
    for idx, tri in enumerate(triples):
        for x in tri:
            point_triples[x].append(idx)

    # Precompute, for each pair (a,b) with a<b, the list of triple-indices containing both
    pair_triples = {}
    for idx, tri in enumerate(triples):
        for i in range(t):
            for j in range(i + 1, t):
                key = (tri[i], tri[j])
                pair_triples.setdefault(key, []).append(idx)

    # Precompute, for each k-subset (block), the list of triple-indices it covers
    # We'll compute block triple-lists on the fly using point_triples + pair_triples for speed
    def block_triple_indices(block):
        """Return list of triple indices covered by a k-element block."""
        # For t=3: a triple (a,b,c) is in the block iff all three points are in the block
        # Use pair_triples: for each pair in block, get triples containing that pair,
        # then filter to those whose third point is also in the block.
        block_set = set(block)
        seen = set()
        result = []
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                key = (block[i], block[j])
                lst = pair_triples.get(key)
                if lst is None:
                    continue
                for idx in lst:
                    if idx in seen:
                        continue
                    # Check the third point
                    tri = triples[idx]
                    # The triple contains block[i] and block[j]; find the third element
                    third = tri[0] + tri[1] + tri[2] - block[i] - block[j]
                    if third in block_set:
                        seen.add(idx)
                        result.append(idx)
        return result

    # Initialize with a greedy covering (fast, using bitsets via numpy)
    def greedy_cover():
        uncovered_mask = np.ones(N, dtype=np.uint8)
        blocks = []
        while uncovered_mask.any():
            # Pick a random uncovered triple
            uncov_indices = np.nonzero(uncovered_mask)[0]
            start_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
            start_tri = triples[start_idx]
            block = list(start_tri)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                best_gain = -1
                best_cands = []
                for x in candidates:
                    gain = 0
                    # For t=3, adding x to a block of size m covers C(m,2) new triples
                    # involving x and pairs from the current block
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            key = (block[i], block[j])
                            lst = pair_triples.get(key)
                            if lst:
                                for idx in lst:
                                    tri = triples[idx]
                                    third = tri[0] + tri[1] + tri[2] - block[i] - block[j]
                                    if third == x and uncovered_mask[idx]:
                                        gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cands = [x]
                    elif gain == best_gain:
                        best_cands.append(x)
                if best_cands:
                    block.append(random.choice(best_cands))
                else:
                    block.append(random.choice(candidates))
            block.sort()
            blocks.append(block)
            # Update uncovered mask
            for idx in block_triple_indices(block):
                uncovered_mask[idx] = 0
        return blocks

    def coverage_counts(blocks):
        """Return numpy array of length N with coverage counts for each triple."""
        counts = np.zeros(N, dtype=np.int32)
        for b in blocks:
            for idx in block_triple_indices(b):
                counts[idx] += 1
        return counts

    def is_valid(counts):
        return counts.min() >= 1

    def remove_redundant(blocks, counts):
        """Remove blocks that don't uniquely cover any triple."""
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                b = blocks[i]
                can_remove = True
                for idx in block_triple_indices(b):
                    if counts[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for idx in block_triple_indices(b):
                        counts[idx] -= 1
        return [blocks[i] for i in range(len(blocks)) if active[i]], counts

    # Main: start with greedy, then SA to reduce block count
    best_blocks = None
    best_count = None

    # Initial greedy
    init_blocks = greedy_cover()
    init_counts = coverage_counts(init_blocks)
    if is_valid(init_counts):
        # Remove redundant
        init_blocks, init_counts = remove_redundant(init_blocks, init_counts)
        best_blocks = list(init_blocks)
        best_count = len(best_blocks)

    # SA phase: work on a fixed set of blocks, try to reduce count by finding
    # redundant blocks and swapping in better blocks.
    # Strategy: keep a pool of blocks. Try to replace one block with another that
    # allows removal of another block (net -1).
    if best_blocks is not None:
        blocks = list(best_blocks)
        counts = coverage_counts(blocks)
        n_blocks = len(blocks)

        # SA parameters
        T = 50.0  # initial temperature (in units of "uncovered triples" penalty)
        T_min = 0.1
        cooling = 0.995

        # Precompute triple lists for all current blocks
        block_triples_list = [block_triple_indices(b) for b in blocks]

        sa_start = time.time()
        while time.time() < deadline - 5:
            if not is_valid(counts):
                # Shouldn't happen if we start valid, but safeguard
                break

            # Try a move: replace block i with a new block, then try to remove a redundant block
            # Or: remove a block, see how many triples become uncovered, try to fix with one new block
            move_type = random.random()

            if move_type < 0.5 and n_blocks > 1:
                # Try removing block i and replacing with a better block
                i = random.randint(0, n_blocks - 1)
                old_block = blocks[i]
                old_tris = block_triples_list[i]

                # Remove old block
                for idx in old_tris:
                    counts[idx] -= 1
                uncovered_after_removal = np.nonzero(counts == 0)[0]

                if len(uncovered_after_removal) == 0:
                    # Block was redundant! Remove it.
                    blocks.pop(i)
                    block_triples_list.pop(i)
                    n_blocks -= 1
                    if n_blocks < best_count:
                        best_count = n_blocks
                        best_blocks = list(blocks)
                    continue

                # Try to find a new block that covers all uncovered triples
                # The new block must contain all points that appear in uncovered triples
                must_points = set()
                for idx in uncovered_after_removal:
                    must_points.update(triples[idx])

                if len(must_points) > k:
                    # Can't cover all with one block; restore
                    for idx in old_tris:
                        counts[idx] += 1
                    continue

                remaining = k - len(must_points)
                available = [x for x in range(v) if x not in must_points]
                if remaining > len(available):
                    for idx in old_tris:
                        counts[idx] += 1
                    continue

                # Try several random completions
                best_new_block = None
                best_gain = -1
                for attempt in range(30):
                    new_block_set = set(must_points)
                    rand_avail = available[:]
                    random.shuffle(rand_avail)
                    for x in rand_avail[:remaining]:
                        new_block_set.add(x)
                    new_block = sorted(new_block_set)
                    new_tris = block_triple_indices(new_block)
                    # Check it covers all uncovered
                    new_tris_set = set(new_tris)
                    if all(idx in new_tris_set for idx in uncovered_after_removal):
                        # Compute how many extra triples it covers (beyond uncovered)
                        gain = len(new_tris) - len(uncovered_after_removal)
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = new_block

                if best_new_block is not None:
                    new_tris = block_triple_indices(best_new_block)
                    for idx in new_tris:
                        counts[idx] += 1
                    blocks[i] = best_new_block
                    block_triples_list[i] = new_tris
                else:
                    # Restore
                    for idx in old_tris:
                        counts[idx] += 1

            else:
                # Try swapping two blocks: replace block i with a new random block
                # that covers at least as many currently-undercovered or singly-covered triples
                i = random.randint(0, n_blocks - 1)
                old_block = blocks[i]
                old_tris = block_triples_list[i]

                # Remove old block temporarily
                for idx in old_tris:
                    counts[idx] -= 1

                # Find a new block that restores coverage and ideally improves
                # A good new block covers triples that are now uncovered (count==0)
                uncov = np.nonzero(counts == 0)[0]
                if len(uncov) == 0:
                    # Old block was redundant
                    blocks.pop(i)
                    block_triples_list.pop(i)
                    n_blocks -= 1
                    if n_blocks < best_count:
                        best_count = n_blocks
                        best_blocks = list(blocks)
                    continue

                # Must cover all currently-uncovered triples
                must_points = set()
                for idx in uncov:
                    must_points.update(triples[idx])

                if len(must_points) > k:
                    for idx in old_tris:
                        counts[idx] += 1
                    continue

                remaining = k - len(must_points)
                available = [x for x in range(v) if x not in must_points]
                if remaining > len(available):
                    for idx in old_tris:
                        counts[idx] += 1
                    continue

                best_new_block = None
                best_gain = -1
                for attempt in range(30):
                    new_block_set = set(must_points)
                    rand_avail = available[:]
                    random.shuffle(rand_avail)
                    for x in rand_avail[:remaining]:
                        new_block_set.add(x)
                    new_block = sorted(new_block_set)
                    new_tris = block_triple_indices(new_block)
                    new_tris_set = set(new_tris)
                    if all(idx in new_tris_set for idx in uncov):
                        gain = len(new_tris)
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = new_block

                if best_new_block is not None:
                    new_tris = block_triple_indices(best_new_block)
                    for idx in new_tris:
                        counts[idx] += 1
                    blocks[i] = best_new_block
                    block_triples_list[i] = new_tris
                else:
                    for idx in old_tris:
                        counts[idx] += 1

            # Periodic redundant removal
            if random.random() < 0.05:
                blocks, counts = remove_redundant(blocks, counts)
                block_triples_list = [block_triple_indices(b) for b in blocks]
                n_blocks = len(blocks)
                if n_blocks < best_count:
                    best_count = n_blocks
                    best_blocks = list(blocks)

            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min

    # Final redundant removal on best
    if best_blocks is not None:
        final_counts = coverage_counts(best_blocks)
        best_blocks, final_counts = remove_redundant(best_blocks, final_counts)

    # Validate
    if best_blocks is None:
        # Fallback: return greedy
        best_blocks = init_blocks
        final_counts = coverage_counts(best_blocks)
        best_blocks, final_counts = remove_redundant(best_blocks, final_counts)

    return best_blocks
