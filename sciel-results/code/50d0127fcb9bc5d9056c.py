import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 140.0

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)

    # Precompute block_subsets as lists of subset indices
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)

    # Precompute blocks_covering_subset
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)

    def time_exceeded():
        return time.time() - start_time > timeout

    def run_greedy(randomize_ties=False, seed=None):
        if time_exceeded():
            return None

        if seed is not None:
            random.seed(seed)

        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = np.array([len(s) for s in block_subsets], dtype=np.int32)

        selected_blocks = []
        uncovered_count = N_subsets

        while uncovered_count > 0:
            if time_exceeded():
                return None

            max_pot = np.max(potential)
            if max_pot == 0:
                break

            if randomize_ties:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = np.argmax(potential)

            selected_blocks.append(best_idx)

            for s in block_subsets[best_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    uncovered_count -= 1
                    for b in blocks_covering_subset[s]:
                        potential[b] -= 1

        return selected_blocks

    def local_search(selected_blocks):
        if time_exceeded():
            return selected_blocks

        cover_count = np.zeros(N_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            for s in block_subsets[b_idx]:
                cover_count[s] += 1

        selected_set = set(selected_blocks)

        def do_reduction():
            nonlocal cover_count, selected_blocks, selected_set
            changed = True
            while changed and not time_exceeded():
                changed = False
                for i in range(len(selected_blocks) - 1, -1, -1):
                    b_idx = selected_blocks[i]
                    is_redundant = True
                    for s in block_subsets[b_idx]:
                        if cover_count[s] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s in block_subsets[b_idx]:
                            cover_count[s] -= 1
                        selected_blocks.pop(i)
                        selected_set.remove(b_idx)
                        changed = True

        def do_swap_search(max_rounds=30):
            nonlocal cover_count, selected_blocks, selected_set
            for _ in range(max_rounds):
                if time_exceeded():
                    break

                improved = False
                current_list = list(selected_blocks)
                random.shuffle(current_list)

                for b_idx in current_list:
                    if b_idx not in selected_set:
                        continue

                    uncovered_subs = []
                    for s in block_subsets[b_idx]:
                        if cover_count[s] == 1:
                            uncovered_subs.append(s)

                    if not uncovered_subs:
                        for s in block_subsets[b_idx]:
                            cover_count[s] -= 1
                        selected_blocks.remove(b_idx)
                        selected_set.remove(b_idx)
                        improved = True
                        break

                    # Find candidate blocks that cover all uncovered_subs
                    if len(uncovered_subs) <= 4:
                        candidate_blocks = set(blocks_covering_subset[uncovered_subs[0]])
                        for s in uncovered_subs[1:]:
                            candidate_blocks &= set(blocks_covering_subset[s])
                        candidate_blocks -= selected_set

                        if candidate_blocks:
                            # Heuristic: pick block that covers most additional subsets
                            best_j = None
                            best_score = -1
                            for j_idx in list(candidate_blocks)[:20]:  # Limit candidates
                                # Count how many new subsets this block covers
                                new_subs = 0
                                for s in block_subsets[j_idx]:
                                    if cover_count[s] == 0:
                                        new_subs += 1
                                if new_subs > best_score:
                                    best_score = new_subs
                                    best_j = j_idx

                            if best_j is not None:
                                # Perform swap
                                for s in block_subsets[b_idx]:
                                    cover_count[s] -= 1
                                selected_blocks.remove(b_idx)
                                selected_set.remove(b_idx)

                                for s in block_subsets[best_j]:
                                    cover_count[s] += 1
                                selected_blocks.append(best_j)
                                selected_set.add(best_j)

                                improved = True
                                break

                if not improved:
                    break

                do_reduction()

        do_reduction()
        do_swap_search()

        return selected_blocks

    best_blocks = None
    best_len = float('inf')

    # Deterministic greedy
    res = run_greedy(randomize_ties=False)
    if res:
        res = local_search(res)
        if res:
            best_blocks = res
            best_len = len(res)

    # Randomized restarts with different seeds
    t_elapsed = time.time() - start_time
    if t_elapsed < timeout - 5:
        for i in range(30):
            if time.time() - start_time > timeout - 2:
                break
            res = run_greedy(randomize_ties=True, seed=42 + i)
            if res:
                res = local_search(res)
                if res and len(res) < best_len:
                    best_len = len(res)
                    best_blocks = res

    if best_blocks is None:
        return []

    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
