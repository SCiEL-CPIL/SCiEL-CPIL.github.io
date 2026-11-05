"""Covering design C(19,8,5): randomized greedy (NumPy gain tracking) + local-search polish."""
import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # ---- Precomputation ----
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each block, the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)

    # Reverse mapping: for each t-subset, which blocks cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # ---- Greedy set cover with NumPy gain tracking + optional noise ----
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        selected = []

        for _ in range(n_k):  # safety bound
            if not uncovered.any():
                break

            scores = gains.astype(np.float64)
            if seed > 0:
                # small noise to break ties and diversify
                scores = scores + rng.random() * 0.01
            best_idx = int(np.argmax(scores))
            if scores[best_idx] < 0.5:  # gain is 0
                break

            selected.append(best_idx)
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1

        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative passes."""
        if not blocks_idx:
            return blocks_idx
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks
        return blocks_idx

    def local_search(blocks_idx, max_time, rng):
        """
        Local search: try to remove redundant blocks and swap blocks to reduce count.
        State: list of block indices. Objective: minimize len(blocks) with full coverage.
        """
        if not blocks_idx:
            return blocks_idx

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        current_blocks = list(blocks_idx)
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)

        # Precompute, for each block, the set of blocks sharing at least one t-subset
        # (for swap moves). Build lazily per block to save memory.
        def blocks_sharing(b_idx):
            s = set()
            for t_idx in k_sub_t_idxs[b_idx]:
                for c in t_sub_to_k_idxs[t_idx]:
                    if c != b_idx:
                        s.add(c)
            return s

        T = 10.0
        cooling = 0.995
        min_T = 0.5

        while time.time() < max_time and T > min_T:
            # --- Move 1: try to remove a block ---
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                # valid if every t-subset it covers is covered by >= 1 other block
                valid = True
                new_uncovered = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        valid = False
                        new_uncovered += 1
                if valid:
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    delta_E = 1.0 + 1000.0 * new_uncovered
                    if rng.random() < math.exp(-delta_E / T):
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
            else:
                # --- Move 2: swap a block for another covering its unique subsets ---
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                # unique subsets of b_idx (covered only by b_idx)
                unique = [t_idx for t_idx in k_sub_t_idxs[b_idx] if cover_count[t_idx] == 1]
                if unique:
                    # candidate replacement blocks that cover all unique subsets
                    cands = blocks_sharing(b_idx)
                    best_cand = None
                    for c in cands:
                        if c in current_blocks:
                            continue
                        ok = True
                        for t_idx in unique:
                            if t_idx not in k_sub_t_idxs[c]:
                                ok = False
                                break
                        if ok:
                            best_cand = c
                            break
                    if best_cand is not None:
                        # remove b_idx, add best_cand
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        for t_idx in k_sub_t_idxs[best_cand]:
                            cover_count[t_idx] += 1
                        current_blocks[i] = best_cand
                        # count may be same; no size improvement but may enable future removals
            T *= cooling

        return best_blocks

    # ---- Main search ----
    best = None
    best_len = 10**9

    # Deterministic greedy baseline
    base = greedy_cover(0)
    if base:
        base = reduce_blocks(base)
        if len(base) < best_len:
            best_len = len(base)
            best = base

    # Randomized greedy restarts, polishing each with local search
    seed = 1
    while time.time() < deadline - 10.0:
        cand = greedy_cover(seed)
        if cand:
            cand = reduce_blocks(cand)
            # polish with local search using remaining time
            ls_budget = min(8.0, max(1.0, deadline - 6.0 - time.time()))
            if ls_budget > 1.0:
                cand = local_search(cand, time.time() + ls_budget, random.Random(seed * 1000 + 7))
            if len(cand) < best_len:
                best_len = len(cand)
                best = cand
        seed += 1

    if best is None:
        best = []

    return [list(all_k_subs[b]) for b in best]
