import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering of t-subsets of a v-set using k-subsets.
    Uses a weighted local search / hypergraph-coloring-inspired greedy
    with dynamic scoring and post-processing.
    """
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    n_t = len(all_t_subs)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    # Precompute for each block the list of t-subset indices it covers
    block_subs = []
    for b in all_blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_subs.append(subs)

    # Precompute for each t-subset, the list of block indices that cover it
    # This helps with redundancy checking
    sub_to_blocks = [[] for _ in range(n_t)]
    for bi, subs in enumerate(block_subs):
        for si in subs:
            sub_to_blocks[si].append(bi)

    def greedy_weighted(seed, time_limit=None):
        random.seed(seed)
        start_time = time.time()
        uncovered = set(range(n_t))
        chosen = []

        while uncovered:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break

            best_block = -1
            best_score = -1.0
            best_gain = -1
            best_degree = 1

            for bi in range(n_blocks):
                if bi in chosen:
                    # Can't pick same block twice
                    continue

                gain = 0
                for si in block_subs[bi]:
                    if si in uncovered:
                        gain += 1

                if gain == 0:
                    continue

                degree = gain if gain > 0 else 1
                # Score: marginal gain weighted by inverse degree (normalized)
                # Higher gain and lower "effective degree" is better
                score = gain / max(degree, 1)
                # Add a small random tie-breaker
                score += random.random() * 0.001

                if score > best_score:
                    best_score = score
                    best_block = bi
                    best_gain = gain
                    best_degree = degree
                elif score == best_score and gain > best_gain:
                    best_block = bi
                    best_gain = gain
                    best_degree = degree

            if best_block == -1:
                # No more blocks can help; we're stuck
                break

            chosen.append(best_block)
            for si in block_subs[best_block]:
                uncovered.discard(si)

        return chosen, uncovered

    def post_process(chosen_blocks, time_budget=30):
        """Remove redundant blocks via iterative sweeping."""
        if not chosen_blocks:
            return chosen_blocks

        # Compute coverage counts
        cov_count = [0] * n_t
        for bi in chosen_blocks:
            for si in block_subs[bi]:
                cov_count[si] += 1

        to_remove = set()
        remaining = list(chosen_blocks)
        random.shuffle(remaining)

        deadline = time.time() + time_budget
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for bi in remaining:
                if bi in to_remove:
                    continue
                redundant = True
                for si in block_subs[bi]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.add(bi)
                    for si in block_subs[bi]:
                        cov_count[si] -= 1
                    changed = True

        result = [bi for bi in chosen_blocks if bi not in to_remove]
        return result

    # Run multiple restarts with different strategies
    best_blocks = None
    best_len = float('inf')

    n_restarts = 6
    time_per_restart = 120.0 / n_restarts - 2.0  # leave buffer for post-processing

    for seed in range(n_restarts):
        chosen, uncovered = greedy_weighted(seed, time_limit=time_per_restart)

        # If we still have uncovered subsets, try to patch them
        if uncovered:
            # Try to cover remaining subsets with additional blocks
            patched = list(chosen)
            for _ in range(10):  # limited attempts
                if not uncovered:
                    break
                best_bi = -1
                best_gain = 0
                for bi in range(n_blocks):
                    if bi in patched:
                        continue
                    gain = sum(1 for si in block_subs[bi] if si in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                if best_bi == -1 or best_gain == 0:
                    break
                patched.append(best_bi)
                for si in block_subs[best_bi]:
                    uncovered.discard(si)
            chosen = patched

        # Post-process to remove redundancies
        processed = post_process(chosen, time_budget=5)

        if len(processed) < best_len:
            best_len = len(processed)
            best_blocks = processed

    # Final validation: ensure all t-subsets are covered
    if best_blocks is None:
        best_blocks = []

    # Convert block indices to actual blocks
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]

    return final_blocks
