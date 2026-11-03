"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math


def _precompute_triples(v, t):
    """Precompute all t-subsets and map them to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triple_to_idx[combo] = idx
        triples.append(combo)
    return triples, triple_to_idx


def _block_triple_indices(block, triple_to_idx):
    """Get the set of triple indices covered by a block."""
    idxs = set()
    for combo in itertools.combinations(block, 3):
        if combo in triple_to_idx:
            idxs.add(triple_to_idx[combo])
    return idxs


def _greedy_cover_weighted(v, k, t, rng, triples, triple_to_idx, max_triples):
    """
    Weighted greedy set cover: prioritize blocks covering rare/uncovered triples.
    Returns a list of blocks (tuples).
    """
    n_triples = len(triples)
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()

    # Precompute for each point, the list of triples it participates in
    point_triples = [[] for _ in range(v)]
    for tri_idx, tri in enumerate(triples):
        for p in tri:
            point_triples[p].append(tri_idx)

    while uncovered_count > 0:
        # Find the hardest uncovered triple (one with fewest potential block partners)
        # Approximate: pick first uncovered for speed, but weight by rarity
        seed_idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                seed_idx = i
                break

        if seed_idx < 0:
            break

        seed_triple = triples[seed_idx]
        block = list(seed_triple)
        block_set = set(block)

        # Greedily extend: pick point maximizing newly-covered uncovered triples
        while len(block) < k:
            best_x = None
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(new_triple, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            for combo in itertools.combinations(b_tuple, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1

    return blocks


def _try_block_merge(blocks, v, k, t, triples, triple_to_idx):
    """
    Try to replace two blocks with one (or remove one if subsumed).
    Returns (new_blocks, success) where success indicates count reduction.
    """
    n = len(blocks)
    if n < 2:
        return blocks, False

    # Precompute triple coverage counts
    counts = [0] * len(triples)
    for b in blocks:
        for combo in itertools.combinations(b, t):
            idx = triple_to_idx.get(combo, -1)
            if idx >= 0:
                counts[idx] += 1

    # For each pair of blocks, try to merge
    for i in range(n):
        for j in range(i + 1, n):
            bi = set(blocks[i])
            bj = set(blocks[j])

            # Try: can we find a single block that covers all triples
            # currently covered only by block i or block j (count==1 for those)?
            # First, identify triples that would become uncovered if we remove both
            critical = []
            for combo in itertools.combinations(blocks[i], t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and counts[idx] == 1:
                    critical.append(idx)
            for combo in itertools.combinations(blocks[j], t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and counts[idx] == 1:
                    critical.append(idx)

            if not critical:
                # Both blocks are redundant — remove one
                new_blocks = blocks[:i] + blocks[i+1:]
                return new_blocks, True

            # The merged block must cover all critical triples
            # Critical triples are 3-element subsets; the new block must contain
            # all points that appear in critical triples
            required = set()
            for idx in critical:
                for p in triples[idx]:
                    required.add(p)

            if len(required) > k:
                continue

            # Build a candidate block containing all required points, fill rest greedily
            candidate = list(required)
            cand_set = set(candidate)

            # Count how many critical triples are already covered
            # Fill remaining slots to cover more uncovered-after-removal triples
            # Simpler: just check if required fits, then verify full coverage
            while len(candidate) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in cand_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(candidate, t - 1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = triple_to_idx.get(new_triple, -1)
                        if idx >= 0 and counts[idx] <= 2:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                candidate.append(best_x)
                cand_set.add(best_x)

            candidate.sort()
            new_b = tuple(candidate)
            if new_b in blocks:
                continue

            # Verify: after removing i and j, adding new_b, all triples still covered
            # Quick check: for each critical triple, is it in new_b?
            ok = True
            for idx in critical:
                tri_set = set(triples[idx])
                if not tri_set.issubset(cand_set):
                    ok = False
                    break
            if not ok:
                continue

            # Full verification of new solution
            new_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:] + [new_b]
            # Check validity
            new_counts = [0] * len(triples)
            for b in new_blocks:
                for combo in itertools.combinations(b, t):
                    idx = triple_to_idx.get(combo, -1)
                    if idx >= 0:
                        new_counts[idx] += 1
            if all(c >= 1 for c in new_counts):
                return new_blocks, True

    return blocks, False


def _local_search_sa(blocks, v, k, t, triples, triple_to_idx, time_limit=100.0):
    """
    Simulated annealing local search with remove, swap, add, and merge moves.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(42)

    best_blocks = list(blocks)
    best_n = len(best_blocks)
    current_blocks = list(blocks)
    current_n = len(current_blocks)

    def compute_counts(blk_list):
        counts = [0] * len(triples)
        for b in blk_list:
            for combo in itertools.combinations(b, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def is_valid_counts(counts):
        return all(c >= 1 for c in counts)

    current_counts = compute_counts(current_blocks)

    T_start = 1.0
    T_end = 0.01
    cooling = 0.9995

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        T = T_start * (cooling ** iteration)
        if T < T_end:
            T = T_end

        move_type = rng.random()
        if move_type < 0.35:
            # Try to remove a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            b = current_blocks[i]
            removable = True
            for combo in itertools.combinations(b, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and current_counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                new_blocks = current_blocks[:i] + current_blocks[i+1:]
                current_blocks = new_blocks
                current_counts = compute_counts(new_blocks)
                current_n = len(current_blocks)
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
                continue
        elif move_type < 0.65:
            # Try to swap a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            old_b = current_blocks[i]
            old_set = set(old_b)

            new_set = set()
            for _ in range(k):
                candidates = [x for x in range(v) if x not in new_set]
                old_cands = [x for x in old_set if x not in new_set]
                if old_cands and rng.random() < 0.6:
                    x = rng.choice(old_cands)
                else:
                    x = rng.choice(candidates)
                new_set.add(x)

            new_b = tuple(sorted(new_set))
            if new_b == old_b:
                continue
            if new_b in current_blocks:
                continue

            new_blocks = current_blocks[:]
            new_blocks[i] = new_b
            new_counts = compute_counts(new_blocks)

            if is_valid_counts(new_counts):
                current_blocks = new_blocks
                current_counts = new_counts
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
        elif move_type < 0.80:
            # Try block merge (replace 2 with 1 or remove redundant)
            if len(current_blocks) < 2:
                continue
            new_blocks, success = _try_block_merge(
                current_blocks, v, k, t, triples, triple_to_idx)
            if success:
                current_blocks = new_blocks
                current_counts = compute_counts(new_blocks)
                current_n = len(current_blocks)
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
        else:
            # Try to add a block (to escape local minima)
            if current_n > 90:
                continue
            new_b = tuple(sorted(rng.sample(range(v), k)))
            if new_b in current_blocks:
                continue
            new_blocks = current_blocks + [new_b]
            current_blocks = new_blocks
            current_counts = compute_counts(new_blocks)
            current_n = len(current_blocks)

        if iteration % 5000 == 0 and current_n > best_n + 5:
            current_blocks = list(best_blocks)
            current_counts = compute_counts(current_blocks)
            current_n = best_n

    return best_blocks, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)
    max_triples = len(triples)

    # Step 1: Multiple greedy restarts
    best_greedy = None
    best_greedy_n = float('inf')

    greedy_time = 15.0
    greedy_deadline = start_time + greedy_time

    for restart in range(20):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover_weighted(v, k, t, r, triples, triple_to_idx, max_triples)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover_weighted(v, k, t, rng, triples, triple_to_idx, max_triples)

    # Step 2: Simulated annealing with merge moves
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        best_blocks, best_n = _local_search_sa(
            best_greedy, v, k, t, triples, triple_to_idx,
            time_limit=remaining_time - 5.0)
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    # Step 3: Final polish — try more merges on the best solution
    if time.time() < start_time + time_limit - 2.0:
        for _ in range(50):
            if time.time() >= start_time + time_limit - 2.0:
                break
            new_blocks, success = _try_block_merge(
                best_blocks, v, k, t, triples, triple_to_idx)
            if success:
                best_blocks = new_blocks
                best_n = len(best_blocks)
            else:
                break

    return [list(b) for b in best_blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
