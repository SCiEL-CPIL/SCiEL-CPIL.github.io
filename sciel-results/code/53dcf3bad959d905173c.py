"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math
import numpy as np


def _precompute_triples(v, t):
    """Precompute all t-subsets and map them to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triple_to_idx[combo] = idx
        triples.append(combo)
    return triples, triple_to_idx


def _build_candidate_blocks(v, k, triples, triple_to_idx):
    """
    Build a set of candidate blocks and their triple coverage bitsets.
    Returns: list of (block_tuple, set_of_triple_indices)
    """
    candidates = []
    seen = set()
    for combo in itertools.combinations(range(v), k):
        b = tuple(combo)
        if b in seen:
            continue
        seen.add(b)
        idxs = set()
        for c3 in itertools.combinations(b, 3):
            idx = triple_to_idx.get(c3, -1)
            if idx >= 0:
                idxs.add(idx)
        candidates.append((b, idxs))
    return candidates


def _greedy_cover_fast(v, k, t, rng, triples, triple_to_idx, candidates, time_budget=None, start_time=None):
    """
    Fast greedy using precomputed candidate blocks and demand-priority.
    """
    n_triples = len(triples)
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()

    # Precompute for each triple, which candidate blocks cover it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for bi, (b, idxs) in enumerate(candidates):
        for idx in idxs:
            triple_to_blocks[idx].append(bi)

    # Demand: number of candidate blocks covering each triple
    demand = np.array([len(triple_to_blocks[i]) for i in range(n_triples)], dtype=np.int32)

    while uncovered_count > 0:
        # Pick uncovered triple with minimum demand (hardest to cover)
        min_demand = float('inf')
        seed_idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                d = demand[i]
                if d < min_demand:
                    min_demand = d
                    seed_idx = i
                elif d == min_demand and rng.random() < 0.3:
                    seed_idx = i  # random tie-break

        if seed_idx < 0:
            break

        # Among blocks covering this triple, pick the one with max new coverage
        best_bi = -1
        best_gain = -1
        for bi in triple_to_blocks[seed_idx]:
            b, idxs = candidates[bi]
            if b in seen_blocks:
                continue
            gain = 0
            for idx in idxs:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_bi = bi

        if best_bi < 0 or best_gain <= 0:
            # Fallback: random block covering the seed triple
            avail = [bi for bi in triple_to_blocks[seed_idx] if candidates[bi][0] not in seen_blocks]
            if not avail:
                # Should not happen for valid covering
                break
            best_bi = rng.choice(avail)

        b = candidates[best_bi][0]
        seen_blocks.add(b)
        blocks.append(b)

        # Update uncovered
        for idx in candidates[best_bi][1]:
            if uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1

        # Update demand (decrement for triples that are now covered)
        # This is approximate but helps guide future choices

    return blocks


def _greedy_cover_direct(v, k, t, rng, triples, triple_to_idx):
    """
    Direct greedy: pick seed triple, then greedily extend block.
    Uses demand-priority for seed selection.
    """
    n_triples = len(triples)
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()

    # For each triple, count how many k-subsets of v contain it
    # A triple {a,b,c} is contained in C(v-3, k-3) blocks
    # But we want "demand" = how hard it is to cover, which relates to
    # how many blocks in our current solution cover it (0 for uncovered)
    # Use a simpler heuristic: prefer triples whose elements are "rare" in uncovered context

    while uncovered_count > 0:
        # Pick an uncovered triple - use demand-based priority
        # A triple has lower "demand" if fewer of its elements appear in many uncovered triples
        # Simplified: pick first uncovered, or use random with bias
        seed_idx = -1
        # Try to find a triple with low "support" (elements that appear in few uncovered triples)
        # For speed, just pick randomly among uncovered
        uncovered_indices = [i for i in range(n_triples) if uncovered[i]]
        if not uncovered_indices:
            break
        seed_idx = rng.choice(uncovered_indices)

        seed_triple = triples[seed_idx]
        block = list(seed_triple)
        block_set = set(block)

        # Greedily extend the block
        while len(block) < k:
            best_x = None
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                # Calculate gain: how many new uncovered triples would be covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(new_triple, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and best_x is not None and gain > 0 and rng.random() < 0.1:
                    best_x = x  # slight randomization on ties

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


def _local_search(blocks, v, k, t, triples, triple_to_idx, time_limit=100.0):
    """
    Local search to minimize block count.
    Uses precomputed incidence for fast validity checks.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(42)

    n_triples = len(triples)
    best_blocks = list(blocks)
    best_n = len(best_blocks)
    current_blocks = list(blocks)
    current_n = len(current_blocks)

    # Precompute block -> set of triple indices
    def block_triples(b):
        return [triple_to_idx[c] for c in itertools.combinations(b, t) if c in triple_to_idx]

    # Coverage counts
    def compute_counts(blk_list):
        counts = np.zeros(n_triples, dtype=np.int32)
        for b in blk_list:
            for idx in block_triples(b):
                counts[idx] += 1
        return counts

    current_counts = compute_counts(current_blocks)
    block_triple_cache = [block_triples(b) for b in current_blocks]

    def is_valid(counts):
        return np.all(counts >= 1)

    no_improve = 0
    iteration = 0

    while time.time() < deadline:
        iteration += 1
        move_type = rng.random()

        if move_type < 0.5:
            # Try to remove a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            # Check if removable
            removable = True
            for idx in block_triple_cache[i]:
                if current_counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                # Remove block i
                current_counts -= np.array([1 if idx == j else 0 for j in range(n_triples)] for idx in block_triple_cache[i]).sum(axis=0)
                # Simpler: recompute
                current_blocks.pop(i)
                block_triple_cache.pop(i)
                current_counts = compute_counts(current_blocks)
                current_n = len(current_blocks)
                if is_valid(current_counts):
                    if current_n < best_n:
                        best_n = current_n
                        best_blocks = list(current_blocks)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    current_blocks.insert(i, blocks[i] if i < len(blocks) else current_blocks[i])
                    # This is getting complex, let's just revert
                    current_blocks = list(best_blocks)
                    block_triple_cache = [block_triples(b) for b in current_blocks]
                    current_counts = compute_counts(current_blocks)
                    current_n = len(current_blocks)
                    no_improve += 1
        elif move_type < 0.85:
            # Try to swap a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            old_b = current_blocks[i]
            old_set = set(old_b)

            # Generate perturbed block
            new_set = set()
            for _ in range(k):
                old_cands = [x for x in old_set if x not in new_set]
                all_cands = [x for x in range(v) if x not in new_set]
                if old_cands and rng.random() < 0.5:
                    x = rng.choice(old_cands)
                else:
                    x = rng.choice(all_cands)
                new_set.add(x)

            new_b = tuple(sorted(new_set))
            if new_b == old_b:
                continue
            if new_b in current_blocks:
                continue

            old_block = current_blocks[i]
            current_blocks[i] = new_b
            block_triple_cache[i] = block_triples(new_b)
            current_counts = compute_counts(current_blocks)

            if is_valid(current_counts):
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_blocks[i] = old_block
                block_triple_cache[i] = block_triples(old_block)
                current_counts = compute_counts(current_blocks)
                no_improve += 1
        else:
            # Try to add a block (escape local minima)
            if current_n > 90:
                continue
            new_b = tuple(sorted(rng.sample(range(v), k)))
            if new_b in current_blocks:
                continue

            current_blocks.append(new_b)
            block_triple_cache.append(block_triples(new_b))
            current_counts = compute_counts(current_blocks)
            current_n = len(current_blocks)
            no_improve += 1

        # Restart if stuck
        if no_improve > 300:
            current_blocks = list(best_blocks)
            block_triple_cache = [block_triples(b) for b in current_blocks]
            current_counts = compute_counts(current_blocks)
            current_n = len(current_blocks)
            no_improve = 0

    return best_blocks, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 150s limit

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)
    n_triples = len(triples)

    # Step 1: Build candidate blocks (all C(24,7) = 346104 blocks)
    # This is feasible: 346104 * C(7,3) = 346104 * 35 = ~12M operations
    candidates = _build_candidate_blocks(v, k, triples, triple_to_idx)

    # Step 2: Multiple greedy restarts using candidate-based approach
    best_greedy = None
    best_greedy_n = float('inf')

    greedy_deadline = start_time + 15.0  # 15s for greedy

    for restart in range(50):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover_fast(v, k, t, r, triples, triple_to_idx, candidates)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    # Also try direct greedy as fallback/diversity
    if time.time() < greedy_deadline:
        for restart in range(10):
            if time.time() > greedy_deadline:
                break
            r = random.Random(99999 + restart * 31337)
            blocks = _greedy_cover_direct(v, k, t, r, triples, triple_to_idx)
            if len(blocks) < best_greedy_n:
                best_greedy_n = len(blocks)
                best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover_direct(v, k, t, rng, triples, triple_to_idx)
        best_greedy_n = len(best_greedy)

    # Step 3: Local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        best_blocks, best_n = _local_search(best_greedy, v, k, t, triples, triple_to_idx, time_limit=remaining_time - 5.0)
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    return [list(b) for b in best_blocks]


if __name__ == "__main__":
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
