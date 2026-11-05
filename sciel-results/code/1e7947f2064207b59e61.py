import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # For each block, the set of t-subset indices it covers
    # Use a compact representation: for each block, a sorted list of t-idx
    block_t_idxs = []
    for block in all_k_blocks:
        t_idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_idxs.append(t_idxs)

    # Build reverse: for each t-subset, list of blocks covering it
    t_sub_to_blocks = [[] for _ in range(n_t)]
    for b_idx, t_idxs in enumerate(block_t_idxs):
        for t_idx in t_idxs:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Convert to numpy arrays
    t_sub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_sub_to_blocks]

    def greedy_construct(order):
        """Greedy set cover with a given block ordering preference."""
        gains = np.array([len(t_idxs) for t_idxs in block_t_idxs], dtype=np.int32)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        selected_set = set()

        # Use a priority: process blocks in 'order' but pick max gain each time
        # Actually, standard greedy: each step pick block with max uncovered gain
        while np.any(uncovered):
            if time.time() > deadline - 35:
                break
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            newly = []
            for t_idx in block_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    newly.append(t_idx)
                    uncovered[t_idx] = False
            if newly:
                for t_idx in newly:
                    if len(t_sub_to_blocks_np[t_idx]) > 0:
                        gains[t_sub_to_blocks_np[t_idx]] -= 1
        return selected, selected_set

    def deletion_pass(selected):
        """Remove redundant blocks."""
        if not selected:
            return selected
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_sel = []
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                essential = False
                for t_idx in block_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential = True
                        break
                if essential:
                    new_sel.append(b_idx)
                else:
                    changed = True
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            selected = new_sel
        return selected

    def swap_local_search(selected, selected_set):
        """Randomized 1-swap local search: try replacing one block with another."""
        if not selected:
            return selected, selected_set
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        improved = True
        max_iters = 200
        iters = 0
        while improved and time.time() < deadline - 5 and iters < max_iters:
            improved = False
            iters += 1

            # Sample candidate unselected blocks
            unselected = [b for b in range(n_k) if b not in selected_set]
            if len(unselected) > 3000:
                random.shuffle(unselected)
                unselected = unselected[:3000]

            best_swap = None
            best_net = 0
            # For each selected block, try to remove it and see if we can re-cover
            # its uniquely-covered t-subsets by adding one unselected block
            sel_list = list(selected)
            for b_idx in sel_list:
                # t-subsets that would become uncovered if we remove b_idx
                unique = [t_idx for t_idx in block_t_idxs[b_idx] if cover_count[t_idx] == 1]
                if not unique:
                    # removing b_idx is free (no unique coverage) -> just remove
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    improved = True
                    break
                # Need a single block that covers all 'unique'
                # Find candidate blocks covering the rarest unique t-subset
                rarest = min(unique, key=lambda x: len(t_sub_to_blocks[x]))
                for cand in t_sub_to_blocks[rarest]:
                    if cand in selected_set:
                        continue
                    # check cand covers all unique
                    cand_set = set(block_t_idxs[cand])
                    if all(u in cand_set for u in unique):
                        # Apply: remove b_idx, add cand
                        for t_idx in block_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_t_idxs[cand]:
                            cover_count[t_idx] += 1
                        selected.remove(b_idx)
                        selected_set.discard(b_idx)
                        selected.append(cand)
                        selected_set.add(cand)
                        improved = True
                        best_swap = (b_idx, cand)
                        break
                if improved:
                    break
        return selected, selected_set

    # Phase 1: initial greedy
    selected, selected_set = greedy_construct(None)
    selected = deletion_pass(selected)
    selected, selected_set = swap_local_search(selected, selected_set)
    best_len = len(selected)
    best_selected = list(selected)
    best_set = set(selected_set)

    # Phase 2: multiple randomized restarts of greedy, keep best
    restarts = 0
    while time.time() < deadline - 40 and restarts < 8:
        restarts += 1
        # Perturb: start greedy but with a random tie-breaking / random initial pick
        sel2, set2 = greedy_construct(None)
        sel2 = deletion_pass(sel2)
        sel2, set2 = swap_local_search(sel2, set2)
        if len(sel2) < best_len:
            best_len = len(sel2)
            best_selected = list(sel2)
            best_set = set(set2)

    # Final: one more swap pass on the best
    if time.time() < deadline - 3:
        best_selected, best_set = swap_local_search(list(best_selected), set(best_set))
        best_selected = deletion_pass(best_selected)

    final_blocks = [list(all_k_blocks[i]) for i in best_selected]
    return final_blocks
