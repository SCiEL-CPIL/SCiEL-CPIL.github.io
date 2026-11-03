import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # c(k, t) = number of t-subsets in a k-set
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    # block_triples[i] is a sorted array of triple indices covered by block i
    block_triples = np.empty((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # For fast membership checks, create a set per block (for local search)
    block_triple_sets = [set(block_triples[i].tolist()) for i in range(n_blocks)]

    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)

    # Entropy-based weight: blocks covering rare triples get higher weight
    # Use log-frequency as a proxy for information content
    triple_info = np.log(np.maximum(triple_freq, 1.0))  # higher freq = less info

    def greedy_cover(use_entropy=False, random_tiebreak=False, rng=None):
        """Run greedy set cover with optional entropy-based weighting."""
        uncovered = np.ones(n_triples, dtype=bool)
        uncovered_count = n_triples
        selected = []

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Calculate gains
            if use_entropy:
                # Entropy-based: weight by information content of uncovered triples
                gains = np.sum(uncovered[block_triples] * triple_info[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            if random_tiebreak and rng is not None:
                candidates = np.where(gains == max_gain)[0]
                best_idx = rng.choice(candidates)
            else:
                best_idx = np.argmax(gains)

            selected.append(best_idx)
            newly_covered = block_triples[best_idx]
            uncovered_count -= np.sum(uncovered[newly_covered])
            uncovered[newly_covered] = False

        return selected

    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution."""
        if not block_indices:
            return list(block_indices)

        n_sel = len(block_indices)
        # Count how many selected blocks cover each triple
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_triples[bidx]] += 1

        keep = []
        for bidx in block_indices:
            my_triples = block_triples[bidx]
            # Check if any triple is covered only by this block
            has_critical = False
            for tr in my_triples:
                if cover_count[tr] == 1:
                    has_critical = True
                    break
            if not has_critical:
                # Remove this block
                cover_count[my_triples] -= 1
            else:
                keep.append(bidx)

        return keep

    def local_search(block_indices, rng, max_iters=1000, temperature=0.05):
        """Local search with simulated annealing: try removals and swaps."""
        if not block_indices:
            return list(block_indices)

        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        n_sel = len(current_blocks)

        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in current_blocks:
            coverage_count[block_triples[bidx]] += 1

        no_improve = 0
        max_no_improve = 50
        iter_count = 0
        current_temp = temperature

        while time.time() < deadline - 3.0 and iter_count < max_iters:
            iter_count += 1

            # Try to remove redundant blocks
            removed = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            for i in indices:
                bidx = current_blocks[i]
                my_triples = block_triples[bidx]
                redundant = True
                for triple_idx in my_triples:
                    if coverage_count[triple_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    removed = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                        no_improve = 0
                    break

            if removed:
                continue

            # Try swaps: replace one block with another
            if len(current_blocks) == 0:
                break

            i = rng.randrange(len(current_blocks))
            old_bidx = current_blocks[i]
            old_triples = block_triples[old_bidx]

            # Find critical triples (covered only by this block)
            critical_triples = old_triples[coverage_count[old_triples] == 1]

            if len(critical_triples) == 0:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Perturbation: remove a random block and rebuild
                    if len(current_blocks) > 1:
                        j = rng.randrange(len(current_blocks))
                        removed_bidx = current_blocks[j]
                        coverage_count[block_triples[removed_bidx]] -= 1
                        current_blocks.pop(j)

                        # Rebuild
                        uncovered_mask = (coverage_count == 0)
                        while np.any(uncovered_mask):
                            if time.time() > deadline - 3.0:
                                break
                            gains = np.sum(uncovered_mask[block_triples], axis=1)
                            best_add = np.argmax(gains)
                            if gains[best_add] == 0:
                                break
                            coverage_count[block_triples[best_add]] += 1
                            current_blocks.append(best_add)
                            uncovered_mask = (coverage_count == 0)
                    no_improve = 0
                    current_temp = max(temperature * 0.5, 0.01)
                continue

            # Find candidate blocks that cover all critical triples
            # Sample candidates for speed
            n_candidates = min(300, n_blocks)
            cand_indices = rng.sample(range(n_blocks), n_candidates)

            best_candidate = -1
            best_gain = -1

            for cand_idx in cand_indices:
                if cand_idx == old_bidx:
                    continue
                cand_set = block_triple_sets[cand_idx]

                # Check if candidate covers all critical triples
                covers_all = all(ct in cand_set for ct in critical_triples)
                if not covers_all:
                    continue

                # Calculate gain: new triples covered
                new_coverage = 0
                for ct in cand_set:
                    if coverage_count[ct] == 0:
                        new_coverage += 1

                if new_coverage > best_gain:
                    best_gain = new_coverage
                    best_candidate = cand_idx

            if best_candidate >= 0:
                # Perform the swap
                old_set = block_triple_sets[old_bidx]
                cand_set = block_triple_sets[best_candidate]

                for tr in old_set:
                    coverage_count[tr] -= 1
                for tr in cand_set:
                    coverage_count[tr] += 1

                current_blocks[i] = best_candidate
                no_improve = 0
            else:
                no_improve += 1
                current_temp = max(current_temp * 0.95, 0.01)

        return best_blocks

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)

    # Strategy 1: Deterministic greedy (unweighted)
    sel = greedy_cover(use_entropy=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]

    # Strategy 2: Entropy-based greedy
    sel = greedy_cover(use_entropy=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]

    # Strategy 3+: Random restarts with entropy greedy + local search
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        sel = greedy_cover(use_entropy=True, random_tiebreak=True, rng=rng)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)

            # Apply local search
            if time.time() < deadline - 8.0:
                ls_result = local_search(peeled, rng, max_iters=400, temperature=0.03)
                ls_size = len(ls_result)
                if ls_size < size:
                    size = ls_size
                    peeled = ls_result

            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]

    # Final polish: try local search on the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = []
        for b in best_solution:
            t = tuple(b)
            if t in block_to_idx:
                current_indices.append(block_to_idx[t])

        if current_indices:
            ls_result = local_search(current_indices, rng, max_iters=300, temperature=0.02)
            if len(ls_result) < len(best_solution):
                best_solution = [list(all_blocks[i]) for i in ls_result]
                best_size = len(ls_result)

    if best_solution is None:
        best_solution = []

    return best_solution
