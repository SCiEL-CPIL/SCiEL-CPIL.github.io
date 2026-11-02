import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers (only the first t_count entries are valid)
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Precompute, for each point x, the list of t-subset indices that contain x.
    point_ts = [[] for _ in range(v)]
    for i in range(n_ts):
        comb = all_ts[i]
        for x in comb:
            point_ts[x].append(i)
    point_ts = [np.array(lst, dtype=np.int32) for lst in point_ts]

    best_solution = None
    best_score = float('inf')

    def aggressive_deletion(current_blocks, coverage_count):
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 4.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def rebuild_coverage(block_list):
        cc = np.zeros(n_ts, dtype=np.int32)
        for b in block_list:
            cc[block_t_indices[b]] += 1
        return cc

    def try_point_swap(blocks, coverage_count, b_idx, old_x, new_x):
        """Try replacing old_x with new_x in block b_idx. Returns True if valid."""
        if new_x == old_x:
            return False
        block = all_ks[b_idx]
        if new_x in block:
            return False

        other = [p for p in block if p != old_x]
        lost = []
        gained = []
        for rest in itertools.combinations(other, t - 1):
            lost.append(idx_map[tuple(sorted(rest + (old_x,)))])
            gained.append(idx_map[tuple(sorted(rest + (new_x,)))])
        lost = np.array(lost, dtype=np.int32)
        gained = np.array(gained, dtype=np.int32)

        # After swap, coverage for 'lost' drops by 1; must stay >= 1.
        if np.any(coverage_count[lost] <= 1):
            return False

        coverage_count[lost] -= 1
        coverage_count[gained] += 1
        return True

    def local_search(blocks, coverage_count, max_iters):
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False

            # Pass 1: try to delete any redundant blocks
            for b_idx in list(blocks):
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    coverage_count[subs_in_block] -= 1
                    blocks.remove(b_idx)
                    improved = True

            # Pass 2: point-swap perturbations
            if len(blocks) == 0:
                break
            for b_idx in blocks:
                if time.time() - start_time > time_limit - 5.0:
                    break
                block = all_ks[b_idx]
                # Try swapping each point in the block with a random outside point
                for old_x in block:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    for new_x in candidates[:5]:
                        if try_point_swap(blocks, coverage_count, b_idx, old_x, new_x):
                            improved = True
                            break
                    if improved:
                        break

            # After any swap, try deleting again
            if improved:
                for b_idx in list(blocks):
                    subs_in_block = block_t_indices[b_idx]
                    if np.all(coverage_count[subs_in_block] > 1):
                        coverage_count[subs_in_block] -= 1
                        blocks.remove(b_idx)

            if not improved:
                break

        return blocks

    def run_greedy(seed, use_random_weights):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)

        # Random weights per point for randomized weighted greedy
        if use_random_weights:
            point_weights = np.random.uniform(0.5, 2.0, size=v).astype(np.float32)
        else:
            point_weights = None

        while remaining > 0:
            if time.time() - start_time > time_limit - 6.0:
                return None, None

            # Compute gain: number of new t-subsets covered
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks

            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    if point_weights is not None:
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                    else:
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)

        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        # Local search with point swaps (only if time permits)
        if time.time() - start_time < time_limit - 10.0:
            current_blocks = local_search(current_blocks, coverage_count, max_iters=200)
            coverage_count = rebuild_coverage(current_blocks)

        # Final deletion pass
        current_blocks = aggressive_deletion(current_blocks, coverage_count)

        return current_blocks, len(current_blocks)

    # Phase 1: Standard greedy with multiple seeds
    restart_count = 0
    while time.time() - start_time < time_limit - 25.0:
        seed = restart_count
        blocks, score = run_greedy(seed, use_random_weights=False)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    # Phase 2: Randomized weighted greedy (different structure to escape minima)
    if time.time() - start_time < time_limit - 15.0:
        for seed in range(5):
            if time.time() - start_time > time_limit - 15.0:
                break
            blocks, score = run_greedy(1000 + seed, use_random_weights=True)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 3: Final high-precision polish on the best solution found
    if best_solution is not None and time.time() - start_time < time_limit - 8.0:
        coverage_count = rebuild_coverage(best_solution)
        # Tighter deletion: iterate until no more deletions
        best_solution = aggressive_deletion(best_solution, coverage_count)
        coverage_count = rebuild_coverage(best_solution)

        # Final point-swap polish with more iterations
        if time.time() - start_time < time_limit - 5.0:
            best_solution = local_search(best_solution, coverage_count, max_iters=500)
            coverage_count = rebuild_coverage(best_solution)
            best_solution = aggressive_deletion(best_solution, coverage_count)

    if best_solution is None:
        blocks, score = run_greedy(0, use_random_weights=False)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    return final_blocks
