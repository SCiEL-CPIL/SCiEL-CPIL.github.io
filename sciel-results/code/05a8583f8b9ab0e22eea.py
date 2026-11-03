import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    def _remove_redundant(selected_idx_list):
        """Remove redundant blocks from a covering. A block is redundant if all
        its triples are covered by other selected blocks."""
        if len(selected_idx_list) <= 1:
            return list(selected_idx_list)
        n_sel = len(selected_idx_list)
        # For each selected block, count how many of its triples are also covered
        # by at least one other selected block. If all c_k_t triples are covered
        # by others, the block is redundant.
        # Build a coverage count array over triples.
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bi in selected_idx_list:
            coverage_count[block_triples[bi]] += 1

        result = []
        for bi in selected_idx_list:
            # Check if every triple in this block is covered by >= 1 other block
            # i.e., coverage_count > 1 for all triples in this block
            if np.all(coverage_count[block_triples[bi]] > 1):
                # Redundant: remove it and decrement coverage
                coverage_count[block_triples[bi]] -= 1
            else:
                result.append(bi)
        return result

    def _greedy_pass(use_weighted, random_ties):
        """Run one greedy pass. Returns list of block indices."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []

        # Weighted gain: weight each uncovered triple by 1/freq where freq is
        # how many blocks cover it. This prioritizes rare triples.
        # Precompute triple frequencies (number of blocks covering each triple)
        if use_weighted:
            # freq[j] = number of blocks that cover triple j
            freq = np.zeros(n_triples, dtype=np.float64)
            for bi in range(n_blocks):
                freq[block_triples[bi]] += 1
            # Avoid division by zero (shouldn't happen for valid covering params)
            freq = np.maximum(freq, 1.0)
            triple_weights = 1.0 / freq
        else:
            triple_weights = None

        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break

            if use_weighted and triple_weights is not None:
                # Weighted gain: sum of weights of uncovered triples in each block
                mask = uncovered[block_triples]  # (n_blocks, c_k_t)
                gains = np.sum(mask * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)

            if random_ties:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 1e-12:
                    break

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected

    best_solution = None
    best_size = float('inf')

    # Phase 1: Deterministic greedy
    sel = _greedy_pass(use_weighted=False, random_ties=False)
    sel = _remove_redundant(sel)
    if len(sel) < best_size:
        best_size = len(sel)
        best_solution = [list(blocks[i]) for i in sel]

    # Phase 2: Weighted greedy (prioritize rare triples)
    sel = _greedy_pass(use_weighted=True, random_ties=False)
    sel = _remove_redundant(sel)
    if len(sel) < best_size:
        best_size = len(sel)
        best_solution = [list(blocks[i]) for i in sel]

    # Phase 3: Multiple random restarts with weighted gains
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        # Alternate between weighted and unweighted with random ties
        use_w = (r % 2 == 0)
        sel = _greedy_pass(use_weighted=use_w, random_ties=True)
        sel = _remove_redundant(sel)
        if len(sel) < best_size:
            best_size = len(sel)
            best_solution = [list(blocks[i]) for i in sel]

    # Phase 4: Local improvement — try swapping blocks to reduce count
    if best_solution is not None and time.time() < deadline - 20.0:
        # Convert to index list for manipulation
        block_set = set(best_solution)
        for b in best_solution:
            block_set.add(tuple(b))

        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            # Try removing each block and see if the covering still holds
            # More efficient: for each block, check if all its triples are covered
            # by remaining blocks.
            for i in range(len(best_solution)):
                if time.time() > deadline - 15.0:
                    break
                # Temporarily remove block i
                test_solution = best_solution[:i] + best_solution[i+1:]
                # Check coverage: build set of covered triples
                covered = set()
                for b in test_solution:
                    for tr in itertools.combinations(b, t):
                        covered.add(tr)
                # Check if block i's triples are all in covered
                bi_triples = set(itertools.combinations(best_solution[i], t))
                if bi_triples.issubset(covered):
                    # Redundant! Remove it
                    best_solution = test_solution
                    best_size = len(best_solution)
                    improved = True
                    break  # Restart loop after modification

    return best_solution
