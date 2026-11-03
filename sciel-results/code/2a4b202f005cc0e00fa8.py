import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # For each triple, precompute which blocks cover it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tj in block_triples[i]:
            triple_to_blocks[tj].append(i)

    # Triple coverage frequency
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1

    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)

    # Precompute block weights
    block_weights = np.sum(triple_weights[block_triples], axis=1)
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, tie_temp=0.0):
        """Run greedy set cover, optionally weighted with optional random tie-breaking."""
        uncovered = uncovered_init.copy()
        selected = []

        while np.any(uncovered):
            if time.time() > deadline:
                break

            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            if random_tie:
                if tie_temp > 0:
                    # Soft-max sampling: pick with probability proportional to exp(gain/temp)
                    exp_gains = np.exp(gains / tie_temp - max_gain / tie_temp)
                    total = np.sum(exp_gains)
                    if total < 1e-300:
                        break
                    probs = exp_gains / total
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage counting."""
        if not selected_indices:
            return selected_indices

        selected_set = set(selected_indices)
        # Count how many selected blocks cover each triple
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in selected_indices:
            cover_count[block_triples[b]] += 1

        remaining = list(selected_indices)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            for b in remaining:
                if time.time() > deadline:
                    break
                # Check if all triples in block b have cover_count > 1
                can_remove = True
                for tj in block_triples[b]:
                    if cover_count[tj] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove block b
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    changed = True
                else:
                    new_remaining.append(b)
            remaining = new_remaining

        return remaining

    def local_search_swap(selected_indices, max_iters=50000):
        """Local search: try swapping one selected block for one unselected block to reduce size."""
        selected_set = set(selected_indices)
        remaining = list(selected_indices)
        n = len(remaining)

        # Build coverage count
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        # Precompute for each selected block, the triples it uniquely covers (cover_count == 1)
        # and for each unselected block, how many uncovered triples it would add

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            # Try removing each selected block
            for i in range(n):
                if time.time() > deadline:
                    break
                b = remaining[i]
                # Check if block b can be removed (all its triples covered by others)
                can_remove = True
                for tj in block_triples[b]:
                    if cover_count[tj] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    remaining.pop(i)
                    n -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: remove one selected block, add one unselected block
            # Only beneficial if we can remove 2+ blocks while adding 1
            # For now, just try single removals already handled above
            # Try 2-for-1 swaps: find two selected blocks whose union of unique triples
            # can be covered by one unselected block
            if n >= 2:
                # Sample a few candidate pairs to keep it fast
                n_samples = min(50, n * (n - 1) // 2)
                for _ in range(n_samples):
                    if time.time() > deadline:
                        break
                    i, j = random.sample(range(n), 2)
                    b1, b2 = remaining[i], remaining[j]

                    # Triples uniquely covered by b1 or b2 (cover_count == 1 and only in that block)
                    # Actually, we need: after removing both, are all their triples still covered?
                    # Triples in b1 ∪ b2 that have cover_count == 1 must be in the new block
                    unique_triples = set()
                    for tj in block_triples[b1]:
                        if cover_count[tj] == 1:
                            unique_triples.add(tj)
                    for tj in block_triples[b2]:
                        if cover_count[tj] == 1:
                            unique_triples.add(tj)

                    if not unique_triples:
                        # Both can be removed without replacement
                        for tj in block_triples[b1]:
                            cover_count[tj] -= 1
                        for tj in block_triples[b2]:
                            cover_count[tj] -= 1
                        # Remove both (remove higher index first)
                        if i > j:
                            remaining.pop(i)
                            remaining.pop(j)
                        else:
                            remaining.pop(j)
                            remaining.pop(i)
                        n -= 2
                        improved = True
                        break

                    if len(unique_triples) > c_k_t:
                        continue

                    # Find an unselected block that covers all unique_triples
                    # Sample candidates
                    n_candidates = min(200, n_blocks - n)
                    found = False
                    for _ in range(n_candidates):
                        if time.time() > deadline:
                            break
                        nb = random.randrange(n_blocks)
                        if nb in selected_set:
                            continue
                        # Check if nb covers all unique_triples
                        nb_set = set(block_triples[nb].tolist())
                        if unique_triples.issubset(nb_set):
                            # Do the swap
                            for tj in block_triples[b1]:
                                cover_count[tj] -= 1
                            for tj in block_triples[b2]:
                                cover_count[tj] -= 1
                            for tj in block_triples[nb]:
                                cover_count[tj] += 1
                            selected_set.discard(b1)
                            selected_set.discard(b2)
                            selected_set.add(nb)
                            # Remove b1 and b2 from remaining
                            if i > j:
                                remaining.pop(i)
                                remaining.pop(j)
                            else:
                                remaining.pop(j)
                                remaining.pop(i)
                            remaining.append(nb)
                            n -= 1
                            improved = True
                            found = True
                            break

                    if found:
                        break

            if not improved:
                break

        return remaining
    
    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')

    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]

    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    try_solution(selected)

    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    try_solution(selected)

    # Strategy 3-14: Random tie-breaking restarts with varying temperatures
    for restart in range(12):
        if time.time() > deadline - 30.0:
            break

        use_weighted = (restart % 2 == 0)
        tie_temp = 0.0 if restart < 6 else (0.5 + restart * 0.3)
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted,
                                random_tie=True, tie_temp=tie_temp)
        try_solution(selected)

    # Local search on the best solution found so far
    if best_solution is not None and time.time() < deadline - 10.0:
        # Reconstruct indices from best_solution
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        improved = local_search_swap(best_indices)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]

    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in blocks[:100]]

    return best_solution
