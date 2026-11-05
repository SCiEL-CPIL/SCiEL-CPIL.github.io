import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)

    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)

    # Helper: Greedy construction with weighted scoring
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=np.int8)
        total_uncovered = n_t_subs
        solution_set = set()
        solution_list = []

        # Count of uncovered subsets each block covers
        uncovered_count_per_block = np.array([len(subs) for subs in block_to_tsubs], dtype=np.int32)

        while total_uncovered > 0:
            if time.time() > deadline:
                break

            # Weighted greedy: score = uncovered_count / (1 + degree_factor)
            # degree_factor: how many other blocks also cover these subsets (redundancy penalty)
            # We approximate: prefer blocks that cover many uncovered subsets
            # and are not too "popular" (i.e., not many blocks overlap on those subsets)

            # Simple heuristic: just pick max uncovered count (classic greedy)
            # But add a small random tie-breaker to diversify
            best_b_idx = np.argmax(uncovered_count_per_block)
            if uncovered_count_per_block[best_b_idx] == 0:
                break

            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)

            # Update uncovered status and counts
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] =0
                    total_uncovered -= 1
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1

        return solution_list, solution_set

    # Helper: Local Search (Deletion and Swap)
    def local_search(solution_list, solution_set):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False

            # Pass 1: Deletion
            for i in range(len(solution_list)):
                b_idx = solution_list[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break

            # Pass 2: Swap - try to replace a block with one that covers its critical subsets
            if not improved:
                for i in range(len(solution_list)):
                    b_idx = solution_list[i]
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)

                    if not critical_subsets:
                        # Should have been deleted
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break

                    # Find candidate block that covers all critical subsets
                    if len(critical_subsets) == 1:
                        candidates = tsub_to_blocks[critical_subsets[0]]
                    else:
                        min_idx = min(range(len(critical_subsets)), key=lambda j: len(tsub_to_blocks[critical_subsets[j]]))
                        candidates = set(tsub_to_blocks[critical_subsets[min_idx]])
                        for s in critical_subsets:
                            if s == critical_subsets[min_idx]:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break

                    if candidates:
                        best_candidate = -1
                        for c in candidates:
                            if c not in solution_set:
                                best_candidate = c
                                break

                        if best_candidate != -1:
                            # Perform Swap
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1

                            solution_list[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break

        return solution_list, solution_set

    # Helper: Perturbation and Repair
    def perturb_and_repair(solution_list, solution_set):
        n_remove = random.randint(3, 7)
        if len(solution_list) <= n_remove:
            return greedy_init()

        indices_to_remove = random.sample(range(len(solution_list)), n_remove)
        indices_to_remove.sort(reverse=True)

        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)

        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove]

        # Calculate uncovered subsets
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        uncovered_mask = (cover_count == 0)
        uncovered_indices = np.where(uncovered_mask)[0]

        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set

        # Greedy Repair
        to_cover = set(uncovered_indices.tolist())

        while to_cover:
            if time.time() > deadline:
                break
            best_b_idx = -1
            max_cov = 0

            if to_cover:
                first_uncovered = next(iter(to_cover))
                candidates = tsub_to_blocks[first_uncovered]

                for b_idx in candidates:
                    if b_idx in new_solution_set:
                        continue
                    cnt = 0
                    for s in block_to_tsubs[b_idx]:
                        if s in to_cover:
                            cnt += 1
                            if cnt > max_cov:
                                max_cov = cnt
                                best_b_idx = b_idx
                                if cnt == len(to_cover):
                                    break
                    if max_cov == len(to_cover):
                        break
                if best_b_idx != -1:
                    new_solution_set.add(best_b_idx)
                    new_solution_list.append(best_b_idx)
                    for s in block_to_tsubs[best_b_idx]:
                        if s in to_cover:
                            to_cover.remove(s)
                else:
                    break

        return new_solution_list, new_solution_set

    # Main ILS Loop
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set

    random.seed(42)

    while time.time() < deadline - 5.0:
        # Perturb
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)

        # Local Search on perturbed
        opt_list, opt_set = local_search(pert_list, pert_set)

        # Update
        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
        else:
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set

    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
