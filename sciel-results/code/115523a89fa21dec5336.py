"""
Combined: Lagrangian-guided WalkSAT + vectorized greedy + pair-replacement refinement.
Uses dynamic penalty weights on uncovered t-subsets to guide stochastic local search,
with pair-replacement moves for structural improvements beyond single-block flips.
"""
import time
import random
import itertools
import numpy as np

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # Inverse: t-subset -> list of blocks covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # Convert to numpy for speed where possible
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # Precompute block_covers as sets for fast membership in local search
    block_covers_set = [set(c) for c in block_covers]

    # ---------------- Vectorized Greedy with random tie-breaking ----------------
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for bi in range(n_k):
            gain[bi] = np.sum(uncovered[block_covers_np[bi]])
        gain = gain.copy()
        blocks_used = []
        while np.any(uncovered):
            if time.time() > deadline - 40.0:
                break
            max_gain = gain.max()
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(chosen)
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in t_to_blocks_np[ts_idx]:
                        gain[bi] -= 1
            gain[chosen] = -1
        return blocks_used

    # Run multiple greedy restarts, keep best
    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(8):
        if time.time() > deadline - 50.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

    selected = set(best_greedy)

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Deletion reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 15.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Pair replacement: try to replace 2 blocks with 1 ----------------
    def pair_replace(selected_set, cc, max_trials=500):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 30.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            union_t = block_covers_set[bi] | block_covers_set[bj]
            if len(union_t) > 35:
                continue
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                cc[block_covers_np[bi]] -= 1
                cc[block_covers_np[bj]] -= 1
                cc[block_covers_np[cand]] += 1
                if np.all(cc >= 1):
                    selected_set.discard(bi)
                    selected_set.discard(bj)
                    selected_set.add(cand)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    break
                else:
                    cc[block_covers_np[bi]] += 1
                    cc[block_covers_np[bj]] += 1
                    cc[block_covers_np[cand]] -= 1
        return selected_set, cc

    selected, cover_count = pair_replace(selected, cover_count)

    # ---------------- Triple replacement: 3 for 1 or 3 for 2 ----------------
    def triple_replace(selected_set, cc, max_trials=300):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 3:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 30.0:
                break
            i, j, l = rng.sample(range(n_sel), 3)
            bi, bj, bl = sel_list[i], sel_list[j], sel_list[l]
            union_t = block_covers_set[bi] | block_covers_set[bj] | block_covers_set[bl]
            if len(union_t) > 70:
                continue
            
            # Find rarest ti in union
            min_count = 1000000
            min_ti = -1
            for ti in union_t:
                c = len(t_to_blocks[ti])
                if c < min_count:
                    min_count = c
                    min_ti = ti
            
            if min_ti == -1:
                continue
            
            # Try 3-for-1 first
            for c1 in t_to_blocks[min_ti]:
                if c1 in (bi, bj, bl):
                    continue
                if union_t <= block_covers_set[c1]:
                    cc[block_covers_np[bi]] -= 1
                    cc[block_covers_np[bj]] -= 1
                    cc[block_covers_np[bl]] -= 1
                    cc[block_covers_np[c1]] += 1
                    if np.all(cc >= 1):
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        selected_set.discard(bl)
                        selected_set.add(c1)
                        selected_set, cc = reduce_blocks(selected_set, cc)
                        return selected_set, cc
                    else:
                        cc[block_covers_np[bi]] += 1
                        cc[block_covers_np[bj]] += 1
                        cc[block_covers_np[bl]] += 1
                        cc[block_covers_np[c1]] -= 1
                    continue
                
                # Try 3-for-2
                remaining = union_t - block_covers_set[c1]
                if not remaining:
                    continue
                min_count2 = 1000000
                min_ti2 = -1
                for ti in remaining:
                    c = len(t_to_blocks[ti])
                    if c < min_count2:
                        min_count2 = c
                        min_ti2 = ti
                if min_ti2 == -1:
                    continue
                for c2 in t_to_blocks[min_ti2]:
                    if c2 in (bi, bj, bl, c1):
                        continue
                    if union_t <= (block_covers_set[c1] | block_covers_set[c2]):
                        cc[block_covers_np[bi]] -= 1
                        cc[block_covers_np[bj]] -= 1
                        cc[block_covers_np[bl]] -= 1
                        cc[block_covers_np[c1]] += 1
                        cc[block_covers_np[c2]] += 1
                        if np.all(cc >= 1):
                            selected_set.discard(bi)
                            selected_set.discard(bj)
                            selected_set.discard(bl)
                            selected_set.add(c1)
                            selected_set.add(c2)
                            selected_set, cc = reduce_blocks(selected_set, cc)
                            return selected_set, cc
                        else:
                            cc[block_covers_np[bi]] += 1
                            cc[block_covers_np[bj]] += 1
                            cc[block_covers_np[bl]] += 1
                            cc[block_covers_np[c1]] -= 1
                            cc[block_covers_np[c2]] -= 1
        return selected_set, cc

    selected, cover_count = triple_replace(selected, cover_count)
    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Lagrangian-guided WalkSAT ----------------
    # State: boolean array
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    n_true = int(x.sum())

    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1

    # Penalty weights for Lagrangian relaxation
    penalty = np.ones(n_t, dtype=np.float64)

    best_x = x.copy()
    best_n_true = n_true

    remaining = deadline - time.time() - 20.0
    if remaining > 10.0:
        max_steps = int(min(remaining * 8000, 400000))
        max_steps = max(max_steps, 5000)

        unsat = np.where(clause_sat == 0)[0].tolist()
        unsat_count = len(unsat)

        for step in range(max_steps):
            if time.time() > deadline - 20.0:
                break

            # Update penalty dynamically based on unsat clauses
            if step % 500 == 0:
                # Increase penalty for unsatisfied clauses, decrease for satisfied
                for ti in unsat:
                    penalty[ti] = min(penalty[ti] * 1.1, 10.0)
                # Decay penalty for satisfied clauses
                sat_mask = clause_sat >= 1
                penalty[sat_mask] = np.maximum(penalty[sat_mask] * 0.95, 0.1)

            if unsat_count > 0:
                # Pick an unsatisfied clause, weighted by penalty
                weights = penalty[unsat]
                total_w = weights.sum()
                if total_w == 0:
                    chosen_ti = unsat[_RNG.randint(0, len(unsat) - 1)]
                else:
                    r = _RNG.random() * total_w
                    cumsum = 0
                    chosen_ti = unsat[0]
                    for idx, ti in enumerate(unsat):
                        cumsum += penalty[ti]
                        if r <= cumsum:
                            chosen_ti = ti
                            break

                # Find blocks that cover chosen_ti
                covering_blocks = t_to_blocks[chosen_ti]
                if not covering_blocks:
                    continue

                # With some probability, flip an already-true block to false
                if _RNG.random() < 0.3:
                    # Try to remove a block that covers chosen_ti
                    true_covering = [bi for bi in covering_blocks if x[bi]]
                    if true_covering:
                        bi = true_covering[_RNG.randint(0, len(true_covering) - 1)]
                        # Check if removing bi would break other clauses
                        removable = True
                        for ti in block_covers_np[bi]:
                            if clause_sat[ti] == 1:
                                removable = False
                                break
                        if removable:
                            x[bi] = False
                            clause_sat[block_covers_np[bi]] -= 1
                            n_true -= 1
                            # Update unsat
                            new_unsat = np.where(clause_sat == 0)[0].tolist()
                            unsat = new_unsat
                            unsat_count = len(unsat)
                else:
                    # Try to add a block that covers chosen_ti
                    candidates = [bi for bi in covering_blocks if not x[bi]]
                    if not candidates:
                        continue
                    # Choose candidate that maximizes gain (covers most unsat clauses)
                    best_cand = None
                    best_gain = -1
                    for bi in candidates:
                        gain = sum(1 for ti in block_covers_np[bi] if clause_sat[ti] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = bi
                    bi = best_cand
                    x[bi] = True
                    clause_sat[block_covers_np[bi]] += 1
                    n_true += 1
                    # Update unsat
                    new_unsat = np.where(clause_sat == 0)[0].tolist()
                    unsat = new_unsat
                    unsat_count = len(unsat)

            # If no unsat clauses, try to reduce
            if unsat_count == 0:
                # Try to remove a random block
                true_blocks = np.where(x)[0]
                if len(true_blocks) > 0:
                    bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                    removable = True
                    for ti in block_covers_np[bi]:
                        if clause_sat[ti] == 1:
                            removable = False
                            break
                    if removable:
                        x[bi] = False
                        clause_sat[block_covers_np[bi]] -= 1
                        n_true -= 1
                        new_unsat = np.where(clause_sat == 0)[0].tolist()
                        unsat = new_unsat
                        unsat_count = len(unsat)

            # Track best
            if n_true < best_n_true:
                best_n_true = n_true
                best_x = x.copy()

    # Restore best solution
    x = best_x
    selected = set(np.where(x)[0].tolist())
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Final reduction passes ----------------
    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = pair_replace(selected, cover_count, max_trials=500)
    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = triple_replace(selected, cover_count, max_trials=300)
    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = pair_replace(selected, cover_count, max_trials=300)
    selected, cover_count = reduce_blocks(selected, cover_count)

    # Build output
    blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            selected = set(best_greedy)
            blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks
