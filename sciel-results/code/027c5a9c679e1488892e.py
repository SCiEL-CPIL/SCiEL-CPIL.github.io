import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s wall-clock, 35s buffer)

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # C(k, t)
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    # block_triples[i] = array of triple indices covered by block i
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1

    # Precompute block point sets for fast overlap
    block_sets = [set(b) for b in blocks]

    # Triple coverage frequency
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)

    # triple_to_blocks: for each triple, list of block indices covering it
    triple_to_blocks = [np.empty(0, dtype=np.int32) for _ in range(n_triples)]
    for i in range(n_triples):
        triple_to_blocks[i] = np.empty(triple_counts[i], dtype=np.int32)
    cursors = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx][cursors[triple_idx]] = i
            cursors[triple_idx] += 1

    def greedy_cover(random_tiebreak=False, alpha=1.0, beta=0.0):
        """Weighted greedy set cover."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        selected_set = set()

        while np.any(uncovered):
            if time.time() > deadline:
                break

            raw_counts = np.sum(uncovered[block_triples].astype(np.int32), axis=1)
            weighted_gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)

            max_weighted = np.max(weighted_gains)
            normalized_weighted = weighted_gains / max_weighted if max_weighted > 0 else weighted_gains

            max_raw = np.max(raw_counts)
            normalized_raw = raw_counts.astype(np.float64) / max_raw if max_raw > 0 else raw_counts.astype(np.float64)

            combined_scores = alpha * normalized_weighted + (1 - alpha) * normalized_raw

            if beta > 0 and len(selected) > 0:
                # Fast overlap penalty using precomputed sets
                for i in range(n_blocks):
                    if combined_scores[i] == 0:
                        continue
                    overlap = 0
                    for s in selected_set:
                        overlap += len(block_sets[i] & block_sets[s])
                    combined_scores[i] -= beta * overlap / max(max_raw, 1)

            if random_tiebreak:
                max_score = np.max(combined_scores)
                if max_score == 0:
                    break
                threshold = max_score * 0.95
                candidates = np.where(combined_scores >= threshold)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(combined_scores)
                if combined_scores[best_idx] == 0:
                    break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected

    def remove_redundant(block_indices):
        """Remove blocks that are not needed."""
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        return result

    def local_search_conflict_density(block_indices, max_iterations=40):
        """
        Local search with conflict-density scoring:
        - Remove redundant blocks
        - Replace blocks using score = new_triples / (1 + conflict_size)
        where conflict_size measures how much the candidate overlaps with
        critical triples of other selected blocks.
        """
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False

            # Pass 1: Remove redundant blocks
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                bt = block_triples[bi]
                if np.all(coverage[bt] > 1):
                    coverage[bt] -= 1
                    current.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # Pass 2: Conflict-density replacement
            best_improvement = None
            best_i = -1
            best_cand = -1
            best_score = -1e9

            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                bt = block_triples[bi]

                critical_mask = coverage[bt] == 1
                if not np.any(critical_mask):
                    # Redundant
                    coverage[bt] -= 1
                    current.pop(i)
                    improved = True
                    break

                critical_triples = bt[critical_mask]
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                other_critical = critical_triples[critical_triples != min_ct]

                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) > 0 and not np.all(np.isin(other_critical, block_triples[cand])):
                        continue

                    # Conflict-density score
                    cand_triples = block_triples[cand]
                    new_count = np.sum(coverage[cand_triples] == 0)

                    # Conflict size: how many of this candidate's triples are
                    # already critical (coverage==1) for OTHER blocks
                    # i.e., swapping would risk uncovering those
                    conflict_mask = (coverage[cand_triples] == 1)
                    conflict_size = np.sum(conflict_mask)

                    # Also penalize point overlap with other selected blocks
                    point_overlap = 0
                    for j, other_bi in enumerate(current):
                        if j == i:
                            continue
                        point_overlap += len(block_sets[cand] & block_sets[other_bi])

                    score = float(new_count) / (1.0 + float(conflict_size)) - 0.01 * float(point_overlap)

                    if score > best_score:
                        best_score = score
                        best_improvement = True
                        best_i = i
                        best_cand = cand

            if best_improvement and best_cand >= 0:
                bi = current[best_i]
                coverage[block_triples[bi]] -= 1
                coverage[block_triples[best_cand]] += 1
                current[best_i] = best_cand
                improved = True

            if not improved:
                break

        return current

    # --- Main search ---
    best_solution = None
    best_size = float('inf')

    # Deterministic greedy passes with varying alpha
    alphas = [1.0, 0.8, 0.6, 0.5]
    for alpha in alphas:
        if time.time() > deadline - 40.0:
            break
        selected = greedy_cover(random_tiebreak=False, alpha=alpha)
        selected = remove_redundant(selected)
        selected = local_search_conflict_density(selected, max_iterations=25)
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]

    if best_solution is None:
        selected = greedy_cover(random_tiebreak=False, alpha=1.0)
        selected = remove_redundant(selected)
        best_solution = [list(blocks[i]) for i in selected]
        best_size = len(best_solution)

    # Random restarts
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        alpha = 0.5 + 0.5 * (r % 3) / 2
        beta = 0.05 * ((r % 4) / 3)
        selected = greedy_cover(random_tiebreak=True, alpha=alpha, beta=beta)
        selected = remove_redundant(selected)
        selected = local_search_conflict_density(selected, max_iterations=12)
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]

    # Final high-precision polish on best solution
    if time.time() < deadline - 5.0 and best_solution is not None:
        # Convert back to indices
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        polished = local_search_conflict_density(best_indices, max_iterations=50)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = [list(blocks[i]) for i in polished]

        # One more round of redundancy removal + polish
        if time.time() < deadline - 3.0:
            polished = remove_redundant(polished)
            polished = local_search_conflict_density(polished, max_iterations=30)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = [list(blocks[i]) for i in polished]

    return best_solution
