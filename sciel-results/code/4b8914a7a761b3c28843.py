import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin

    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of triple-indices it covers
    block_triples = []
    for block in all_blocks:
        subs = [subset_index[sub] for sub in itertools.combinations(block, t)]
        block_triples.append(subs)

    # Precompute block masks as Python ints for fast intersection
    block_masks = []
    for subs in block_triples:
        m = 0
        for s in subs:
            m |= (1 << s)
        block_masks.append(m)

    # Build inverse index: for each triple, which blocks cover it
    # triple_to_blocks[t] = list of block indices that cover triple t
    triple_to_blocks = [[] for _ in range(n_subsets)]
    for bi, subs in enumerate(block_triples):
        for t_idx in subs:
            triple_to_blocks[t_idx].append(bi)

    def validate(blocks_list):
        """Check every t-subset is covered by at least one block."""
        covered = np.zeros(n_subsets, dtype=np.bool_)
        for b in blocks_list:
            for t_idx in block_triples[b]:
                covered[t_idx] = True
        return covered.all()

    def abfkp_derandomize(sample_indices, seed):
        """
        ABFKP: Given a random sample of block indices (with possible repeats,
        but we deduplicate), iteratively remove redundant blocks.
        A block is redundant if removing it still leaves every triple covered.
        We process in an order that prioritizes blocks with fewest unique contributions.
        """
        rng = random.Random(seed)
        # Deduplicate
        unique_blocks = list(dict.fromkeys(sample_indices))  # preserve order, dedup
        n = len(unique_blocks)
        if n == 0:
            return []

        # Compute coverage counts
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b in unique_blocks:
            for t_idx in block_triples[b]:
                counts[t_idx] += 1

        # A block is "critical" for triple t if counts[t] == 1 and b covers t.
        # We want to remove blocks that are NOT critical for any triple.
        # Strategy: iteratively find blocks with no critical triples and remove them.
        # For efficiency, maintain for each block the set of triples it is the sole cover of.

        # Initial: for each block, count how many triples it uniquely covers
        # unique_cover_count[b] = number of triples t where b is in unique_blocks and counts[t]==1 and b covers t
        # Actually, let's just do iterative pruning.

        # Represent the current set as a boolean array
        active = np.ones(n, dtype=np.bool_)
        active_set = set(unique_blocks)

        # For fast "is this block redundant?" check:
        # A block b is redundant iff for every triple t in block_triples[b],
        # there exists another active block b' != b that covers t.
        # Equivalently: for every t in block_triples[b], counts[t] > 1 OR
        # there's another active block covering t.
        # Since we maintain counts, a block is redundant iff
        # for all t in block_triples[b], counts[t] > 1.
        # Wait, that's only if all blocks in active_set are counted. Yes.
        # So: block b is redundant iff min(counts[t] for t in block_triples[b]) > 1.

        # Iterative removal:
        # Use a queue. Initially, any block with min count > 1 is removable.
        # After removing a block, update counts and re-check neighbors.

        # For efficiency, let's do rounds. In each round, scan all active blocks
        # and remove those that are redundant. Repeat until no more can be removed.
        # To avoid O(n * n_blocks) per round, we can be smarter, but for our sizes it's OK.

        changed = True
        while changed:
            changed = False
            # Build list of active block indices
            active_indices = [i for i in range(n) if active[i]]
            to_remove = []
            for i in active_indices:
                b = unique_blocks[i]
                # Check if redundant: all triples in b have count > 1
                redundant = True
                for t_idx in block_triples[b]:
                    if counts[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            if to_remove:
                for i in to_remove:
                    b = unique_blocks[i]
                    active[i] = False
                    active_set.discard(b)
                    for t_idx in block_triples[b]:
                        counts[t_idx] -= 1
                changed = True

        result = [unique_blocks[i] for i in range(n) if active[i]]
        return result

    def greedy_cover(seed):
        """Standard max-gain greedy with random tie-breaking."""
        rng = random.Random(seed)
        uncovered_mask = (1 << n_subsets) - 1
        selected = []
        used = set()

        while uncovered_mask:
            best_gain = -1
            best_candidates = []
            for i in range(n_blocks):
                if i in used:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)

            if not best_candidates:
                break

            chosen = rng.choice(best_candidates)
            selected.append(chosen)
            used.add(chosen)
            uncovered_mask &= ~block_masks[chosen]

        return selected

    def local_search_refine(blocks, seed, time_limit):
        """
        Local search: try to remove blocks (if redundant) or swap to reduce size.
        Focus on deletion first, then 1-for-1 swaps to escape local minima.
        """
        rng = random.Random(seed)
        current = list(blocks)
        current_set = set(current)

        # Compute coverage counts
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b in current:
            for t_idx in block_triples[b]:
                counts[t_idx] += 1

        best_size = len(current)
        best_solution = list(current)

        # Phase 1: Aggressive deletion of redundant blocks
        def try_delete_all():
            nonlocal counts, current, current_set
            changed = True
            while changed:
                changed = False
                for idx in range(len(current)):
                    b = current[idx]
                    # Check if b is redundant
                    redundant = True
                    for t_idx in block_triples[b]:
                        if counts[t_idx] <= 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove b
                        current.pop(idx)
                        current_set.discard(b)
                        for t_idx in block_triples[b]:
                            counts[t_idx] -= 1
                        changed = True
                        break  # restart scan
            return len(current) < best_size

        try_delete_all()
        if len(current) < best_size:
            best_size = len(current)
            best_solution = list(current)

        # Phase 2: Swap-based local search
        T = 30.0
        T_min = 0.5
        cooling = 0.995
        no_improve_count = 0

        while T > T_min and time.time() < time_limit:
            # Pick a random block to potentially remove
            if not current:
                break
            b_out_idx = rng.randrange(len(current))
            b_out = current[b_out_idx]
            out_triples = block_triples[b_out]

            # Find critical triples (count == 1) in b_out
            critical = [t_idx for t_idx in out_triples if counts[t_idx] == 1]

            if not critical:
                # Can just delete
                current.pop(b_out_idx)
                current_set.discard(b_out)
                for t_idx in out_triples:
                    counts[t_idx] -= 1
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = list(current)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Need to find a replacement block that covers all critical triples
                # Strategy: pick the rarest critical triple, find blocks covering it,
                # and check if they cover all criticals.
                # For efficiency, pick a random critical triple and sample blocks.
                target_triple = rng.choice(critical)
                candidate_blocks = triple_to_blocks[target_triple]

                if not candidate_blocks:
                    T *= cooling
                    continue

                # Try a few random candidates from those covering target_triple
                tried = 0
                max_tries = 20
                found_valid = False
                while tried < max_tries and not found_valid:
                    b_in = rng.choice(candidate_blocks)
                    if b_in in current_set:
                        tried += 1
                        continue

                    in_triples = block_triples[b_in]
                    in_set = set(in_triples)

                    # Check: b_in covers all critical triples
                    if not all(t in in_set for t in critical):
                        tried += 1
                        continue

                    # Check: all non-critical triples in b_out that are not in b_in have count > 1
                    valid = True
                    for t_idx in out_triples:
                        if t_idx not in in_set:
                            if counts[t_idx] <= 1:
                                valid = False
                                break

                    if not valid:
                        tried += 1
                        continue

                    # Valid swap!
                    # Remove b_out
                    current.pop(b_out_idx)
                    current_set.discard(b_out)
                    for t_idx in out_triples:
                        counts[t_idx] -= 1

                    # Add b_in
                    current.append(b_in)
                    current_set.add(b_in)
                    for t_idx in in_triples:
                        counts[t_idx] += 1

                    # Accept with probability based on SA (size is same, so delta=0)
                    if rng.random() < 0.7:  # accept most swaps to explore
                        found_valid = True
                        no_improve_count += 1
                    else:
                        # Revert
                        current.pop()  # remove b_in
                        current_set.discard(b_in)
                        for t_idx in in_triples:
                            counts[t_idx] -= 1
                        current.insert(b_out_idx, b_out)
                        current_set.add(b_out)
                        for t_idx in out_triples:
                            counts[t_idx] += 1
                        found_valid = True  # we did a move, just reverted

            # After any move, try to delete redundant blocks
            if time.time() < time_limit:
                changed = True
                while changed and time.time() < time_limit:
                    changed = False
                    for idx in range(len(current)):
                        b = current[idx]
                        redundant = True
                        for t_idx in block_triples[b]:
                            if counts[t_idx] <= 1:
                                redundant = False
                                break
                        if redundant:
                            current.pop(idx)
                            current_set.discard(b)
                            for t_idx in block_triples[b]:
                                counts[t_idx] -= 1
                            changed = True
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = list(current)
                            break

            T *= cooling

        return best_solution

    # Main search loop
    best_blocks = None
    best_count = float('inf')

    # First, get a good baseline via greedy
    init = greedy_cover(0)
    if len(init) < best_count:
        best_count = len(init)
        best_blocks = [all_blocks[i] for i in init]

    # Now run ABFKP + local search with multiple random samples
    restart_id = 0
    # Determine sample size: probabilistic method suggests m = O(n * log n / p)
    # where p = probability a random block covers a given triple = C(k-1, t-1)/C(v-1, t-1)
    # For v=24, k=7, t=3: p = C(6,2)/C(23,2) = 15/253 ≈ 0.0593
    # n = 2024 triples. m ≈ 2024 * ln(2024) / 0.0593 ≈ 2024 * 7.61 / 0.0593 ≈ 259,000
    # That's too many. Let's use a smaller sample and rely on local search.
    # Actually, for ABFKP to work well, we need m large enough that the random
    # sample covers everything. Let's use m = 300-500 blocks.

    while time.time() < deadline - 3.0:
        seed = restart_id
        rng = random.Random(seed)

        # Sample m random distinct blocks
        # Use a size that gives good coverage probability
        # P(a specific triple is uncovered) = (1-p)^m where p ≈ 0.0593
        # For m=200: (1-0.0593)^200 ≈ e^(-11.86) ≈ 7e-6. Good.
        # For m=150: (1-0.0593)^150 ≈ e^(-8.89) ≈ 1.4e-4. Still good.
        # Let's vary m between 120 and 250.
        m_sample = rng.randint(120, 250)

        # Sample without replacement
        sample = rng.sample(range(n_blocks), min(m_sample, n_blocks))

        # ABFKP derandomization
        pruned = abfkp_derandomize(sample, seed)

        # If ABFKP didn't produce a valid cover, fall back to greedy
        if not validate(pruned):
            # Try to complete with greedy
            uncovered_mask = (1 << n_subsets) - 1
            for b in pruned:
                uncovered_mask &= ~block_masks[b]
            if uncovered_mask:
                # Greedy completion
                used = set(pruned)
                while uncovered_mask:
                    best_gain = -1
                    best_candidates = []
                    for i in range(n_blocks):
                        if i in used:
                            continue
                        gain = (uncovered_mask & block_masks[i]).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_candidates = [i]
                        elif gain == best_gain and gain > 0:
                            best_candidates.append(i)
                    if not best_candidates:
                        break
                    chosen = rng.choice(best_candidates)
                    pruned.append(chosen)
                    used.add(chosen)
                    uncovered_mask &= ~block_masks[chosen]

        # Local search refinement
        ls_limit = min(time.time() + 2.5, deadline - 2.0)
        refined = local_search_refine(pruned, seed, ls_limit)

        if len(refined) < best_count:
            best_count = len(refined)
            best_blocks = [all_blocks[i] for i in refined]

        restart_id += 1

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return greedy solution
        init = greedy_cover(0)
        return [list(all_blocks[i]) for i in init]
