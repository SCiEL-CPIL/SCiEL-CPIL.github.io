"""Column-generation covering design C(v,k,t): iteratively add blocks that
maximally cover currently-uncovered t-subsets, then polish with local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _encode_key(combo, v, t):
    """Encode a sorted t-tuple as a unique integer index into combinations."""
    return sum(c * (v ** (t - 1 - i)) for i, c in enumerate(combo))


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin under 150s kill

    if t == 0:
        return []

    # --- Precomputation ---
    combos_list = list(itertools.combinations(range(v), t))
    n_subsets = len(combos_list)
    combo_arr = np.array(combos_list, dtype=np.int64)

    # key -> index mapping
    key_to_idx = {}
    for i, comb in enumerate(combos_list):
        key = _encode_key(comb, v, t)
        key_to_idx[key] = i

    # For each pair (a,b) with a<b, precompute the list of t-subset indices
    # that contain BOTH a and b.  For t=3 this is the key speedup: the gain
    # of adding point x to a block B is the sum over pairs (b1,b2) in B of
    # the number of uncovered triples containing {x, b1, b2}.
    # pair_to_triples[(a,b)] = list of triple-indices containing a and b
    pair_to_triples = {}
    for a in range(v):
        for b in range(a + 1, v):
            lst = []
            for i, row in enumerate(combo_arr):
                # Check if row contains both a and b
                # row is sorted, so use binary search or simple check
                if a in row and b in row:
                    lst.append(i)
            pair_to_triples[(a, b)] = np.array(lst, dtype=np.int64)

    # uncovered: True means the t-subset is still not covered
    uncovered = np.ones(n_subsets, dtype=bool)

    # Precompute for each pair (a,b) the numpy array of triple indices
    # so we can do fast sum(uncovered[idxs])
    # Already done above as pair_to_triples

    blocks = []
    block_sets = []  # parallel list of sets for fast membership

    def _compute_gain(x, block_set):
        """Compute how many currently-uncovered t-subsets would be covered
        by adding point x to a block represented by block_set."""
        if t == 1:
            return 1 if uncovered[x] else 0
        elif t == 2:
            total = 0
            for b in block_set:
                key = (min(x, b), max(x, b))
                if key in pair_to_triples:
                    idxs = pair_to_triples[key]
                    if len(idxs) > 0:
                        total += int(np.sum(uncovered[idxs]))
            return total
        elif t == 3:
            total = 0
            for b1, b2 in itertools.combinations(block_set, 2):
                key = (min(x, b1, b2), max(x, b1, b2))
                # Need the triple (x, b1, b2) sorted
                triple = tuple(sorted((x, b1, b2)))
                k3 = _encode_key(triple, v, t)
                if k3 in key_to_idx:
                    idx = key_to_idx[k3]
                    if uncovered[idx]:
                        total += 1
            return total
        else:
            # General case
            total = 0
            for rest in itertools.combinations(block_set, t - 1):
                combo = tuple(sorted(rest + (x,)))
                key = _encode_key(combo, v, t)
                if key in key_to_idx:
                    idx = key_to_idx[key]
                    if uncovered[idx]:
                        total += 1
            return total

    def _add_block(block):
        """Add a block, update uncovered, return number of newly covered subsets."""
        block.sort()
        block_set = set(block)
        newly = 0
        for comb in itertools.combinations(block, t):
            key = _encode_key(comb, v, t)
            if key in key_to_idx:
                idx = key_to_idx[key]
                if uncovered[idx]:
                    uncovered[idx] = False
                    newly += 1
        blocks.append(block)
        block_sets.append(block_set)
        return newly

    # --- Column Generation Loop ---
    while True:
        if time.time() > deadline - 10.0:
            break

        n_uncov = int(np.sum(uncovered))
        if n_uncov == 0:
            break

        # Pricing: find the 7-subset that maximally covers uncovered triples.
        # Strategy: greedy construction seeded from an uncovered triple.
        # Try multiple seeds and keep the best block.

        best_block = None
        best_gain = -1

        # Get indices of uncovered triples
        uncov_indices = np.where(uncovered)[0]

        # Try several seeds (up to a budget)
        n_seeds = min(20, len(uncov_indices))
        if n_seeds == 0:
            break
        seed_indices = random.sample(range(len(uncov_indices)), n_seeds)

        for si in seed_indices:
            if time.time() > deadline - 10.0:
                break

            seed_idx = uncov_indices[si]
            block = list(combo_arr[seed_idx])
            block_set = set(block)

            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                if time.time() > deadline - 10.0:
                    break

                best_x = -1
                best_g = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = _compute_gain(x, block_set)
                    if g > best_g:
                        best_g = g
                        best_x = x

                if best_x == -1 or best_g == 0:
                    # No point adds coverage; pick a random unused point
                    unused = [x for x in range(v) if x not in block_set]
                    if unused:
                        best_x = random.choice(unused)
                    else:
                        break

                block.append(best_x)
                block_set.add(best_x)

            if len(block) < k:
                continue

            # Evaluate total gain of this complete block
            gain = 0
            for comb in itertools.combinations(block, t):
                key = _encode_key(comb, v, t)
                if key in key_to_idx:
                    idx = key_to_idx[key]
                    if uncovered[idx]:
                        gain += 1

            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None or best_gain <= 0:
            # Fallback: pick an uncovered triple and fill randomly
            seed_idx = uncov_indices[0]
            block = list(combo_arr[seed_idx])
            block_set = set(block)
            while len(block) < k:
                unused = [x for x in range(v) if x not in block_set]
                if not unused:
                    break
                x = random.choice(unused)
                block.append(x)
                block_set.add(x)
            if len(block) < k:
                break
            best_block = block

        _add_block(best_block)

    # --- Local Search Polish ---
    if time.time() < deadline - 5.0 and blocks:
        # Recompute uncovered from scratch
        uncovered2 = np.ones(n_subsets, dtype=bool)
        for block in blocks:
            for comb in itertools.combinations(block, t):
                key = _encode_key(comb, v, t)
                if key in key_to_idx:
                    idx = key_to_idx[key]
                    uncovered2[idx] = False

        # Build coverage count: how many blocks cover each triple
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for block in blocks:
            for comb in itertools.combinations(block, t):
                key = _encode_key(comb, v, t)
                if key in key_to_idx:
                    idx = key_to_idx[key]
                    cover_count[idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for bi in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                block = blocks[bi]
                block_set = set(block)

                # Try each position and each candidate swap
                for pos in range(k):
                    if time.time() > deadline - 5.0:
                        break
                    old_pt = block[pos]
                    for x in range(v):
                        if x in block_set:
                            continue
                        # Compute gain/loss of swapping old_pt -> x
                        new_block = list(block)
                        new_block[pos] = x
                        new_block.sort()
                        new_set = set(new_block)

                        # Triples lost: those containing old_pt but not x
                        # Triples gained: those containing x but not old_pt
                        gain = 0
                        loss = 0

                        # Triples involving x and (t-1) points from new_set \ {x}
                        for rest in itertools.combinations(new_set - {x}, t - 1):
                            combo = tuple(sorted(rest + (x,)))
                            key = _encode_key(combo, v, t)
                            if key in key_to_idx:
                                idx = key_to_idx[key]
                                if uncovered2[idx]:
                                    gain += 1
                                elif cover_count[idx] == 1:
                                    # Only covered by this block; check if still covered
                                    # after swap: does new_block still contain this combo?
                                    # It does if all elements of combo are in new_block
                                    if all(c in new_set for c in combo):
                                        pass  # still covered
                                    else:
                                        loss += 1

                        # Triples involving old_pt that are lost
                        # (those that contained old_pt and (t-1) other points from block,
                        #  and do NOT contain x)
                        for rest in itertools.combinations(block_set - {old_pt}, t - 1):
                            combo = tuple(sorted(rest + (old_pt,)))
                            # Check if x is in combo
                            if x in combo:
                                continue  # not lost, x replaces old_pt
                            key = _encode_key(combo, v, t)
                            if key in key_to_idx:
                                idx = key_to_idx[key]
                                if cover_count[idx] == 1:
                                    # Only covered by this block, and x is not in combo
                                    # so it's lost
                                    loss += 1

                        if gain > loss:
                            # Apply swap
                            # Update cover counts for lost triples
                            for rest in itertools.combinations(block_set - {old_pt}, t - 1):
                                combo = tuple(sorted(rest + (old_pt,)))
                                if x in combo:
                                    continue
                                key = _encode_key(combo, v, t)
                                if key in key_to_idx:
                                    idx = key_to_idx[key]
                                    cover_count[idx] -= 1
                                    if cover_count[idx] == 0:
                                        uncovered2[idx] = True

                            # Update cover counts for gained triples
                            for rest in itertools.combinations(new_set - {x}, t - 1):
                                combo = tuple(sorted(rest + (x,)))
                                key = _encode_key(combo, v, t)
                                if key in key_to_idx:
                                    idx = key_to_idx[key]
                                    cover_count[idx] += 1
                                    if cover_count[idx] >= 1:
                                        uncovered2[idx] = False

                            blocks[bi] = new_block
                            block_sets[bi] = new_set
                            improved = True
                            break
                    if improved:
                        break

    return blocks
# EVOLVE-BLOCK-END
