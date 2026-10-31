"""
Strategy: Randomized Multi-Start Batch Construction + Aggressive Post-Processing.
Instead of sequential greedy, we:
1. Generate diverse initial coverings via batch random selection with coverage feedback.
2. Aggressively reduce via: (a) iterative redundancy removal, (b) 2-for-1 block exchanges
   (replace two blocks with one if their unique coverage is redundant given others).
3. Keep the best (smallest) covering across all restarts within the time budget.
"""

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    cache_key = (v, k, t)
    if not hasattr(construct, '_cache') or construct._cache_key != cache_key:
        all_t_subsets = list(itertools.combinations(range(v), t))
        n_tsub = len(all_t_subsets)
        tsub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

        all_k_subsets = list(itertools.combinations(range(v), k))
        n_ksub = len(all_k_subsets)

        # Build incidence structure: for each k-subset, list of t-subset indices it covers
        flat_idx = []
        offsets = [0]
        for comb in all_k_subsets:
            for sub in itertools.combinations(comb, t):
                flat_idx.append(tsub_to_idx[sub])
            offsets.append(len(flat_idx))
        flat_idx = np.array(flat_idx, dtype=np.int32)
        offsets = np.array(offsets, dtype=np.int32)

        # Incidence matrix for fast gain computation
        A = np.zeros((n_ksub, n_tsub), dtype=np.int8)
        rows = np.repeat(np.arange(n_ksub), np.diff(offsets))
        A[rows, flat_idx] = 1

        # Inverse: for each t-subset, which k-subsets cover it
        inv = [[] for _ in range(n_tsub)]
        for i in range(n_ksub):
            s = offsets[i]; e = offsets[i+1]
            for j in flat_idx[s:e]:
                inv[j].append(i)
        inv_offsets = [0]
        for lst in inv:
            inv_offsets.append(inv_offsets[-1] + len(lst))
        inv_flat = np.array([x for lst in inv for x in lst], dtype=np.int32)
        inv_offsets = np.array(inv_offsets, dtype=np.int32)

        construct._cache = dict(
            all_t_subsets=all_t_subsets, n_tsub=n_tsub,
            all_k_subsets=all_k_subsets, n_ksub=n_ksub,
            A=A, offsets=offsets, flat_idx=flat_idx,
            inv_flat=inv_flat, inv_offsets=inv_offsets,
        )
        construct._cache_key = cache_key

    C = construct._cache
    all_t_subsets = C['all_t_subsets']; n_tsub = C['n_tsub']
    all_k_subsets = C['all_k_subsets']; n_ksub = C['n_ksub']
    A = C['A']; offsets = C['offsets']; flat_idx = C['flat_idx']
    inv_flat = C['inv_flat']; inv_offsets = C['inv_offsets']

    best_count = float('inf')
    best_selected = None

    def try_reduce(selected, used):
        """Aggressively reduce: redundancy removal + 2-for-1 exchanges."""
        nonlocal best_count, best_selected
        if time.time() > deadline - 5.0:
            return

        cover = np.zeros(n_tsub, dtype=np.int16)
        for i in selected:
            cover[flat_idx[offsets[i]:offsets[i+1]]] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False

            # Pass A: Remove redundant blocks
            for pos in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    return
                i = selected[pos]
                seg = flat_idx[offsets[i]:offsets[i+1]]
                if cover[seg].min() > 1:
                    cover[seg] -= 1
                    selected.pop(pos)
                    used.discard(i)
                    changed = True
                    break  # restart

            if not changed:
                # Pass B: 2-for-1 exchange
                # Try to replace two blocks i, j with one block u if:
                #   u covers all t-subsets that are uniquely covered by i or j (given others)
                # Simplified safe rule: if removing i and j leaves no holes, and u covers
                # the union of what i and j uniquely covered, accept.
                # Faster heuristic: for each pair (i,j), check if there exists u such that
                # every t-subset in i∪j that has cover==1 (i.e., only covered by i or j)
                # is also in u. Then u can replace both.
                n_sel = len(selected)
                if n_sel >= 2:
                    # Limit pairs checked per round for speed
                    max_pairs = min(200, n_sel * (n_sel - 1) // 2)
                    rng_local = random.Random(hash(tuple(sorted(selected))) % 10000)
                    pairs_checked = 0
                    for pos_i in range(n_sel):
                        if time.time() > deadline - 5.0:
                            return
                        i = selected[pos_i]
                        i_seg = flat_idx[offsets[i]:offsets[i+1]]
                        for pos_j in range(pos_i + 1, n_sel):
                            if pairs_checked >= max_pairs:
                                break
                            if time.time() > deadline - 5.0:
                                return
                            j = selected[pos_j]
                            j_seg = flat_idx[offsets[j]:offsets[j+1]]

                            # t-subsets uniquely covered by i or j (cover==1 and in i or j)
                            crit_i = i_seg[cover[i_seg] == 1]
                            crit_j = j_seg[cover[j_seg] == 1]
                            crit_set = set(crit_i.tolist()) | set(crit_j.tolist())

                            if len(crit_set) == 0:
                                # Both are redundant together? Try removing both
                                cover[i_seg] -= 1
                                cover[j_seg] -= 1
                                if (cover[i_seg] >= 0).all() and (cover[j_seg] >= 0).all():
                                    # Actually need to check no negatives
                                    # Revert and do properly
                                    cover[i_seg] += 1
                                    cover[j_seg] += 1
                                # Skip this complex case for now
                                continue

                            # Find candidate u covering all in crit_set
                            cand = None
                            for jj in list(crit_set)[:10]:  # limit for speed
                                s = inv_offsets[jj]; e = inv_offsets[jj+1]
                                lst = inv_flat[s:e]
                                if cand is None:
                                    cand = set(lst.tolist())
                                else:
                                    cand &= set(lst.tolist())
                                if not cand:
                                    break
                            if not cand:
                                continue
                            cand.discard(i)
                            cand.discard(j)
                            unused_cand = [u for u in cand if u not in used]
                            if not unused_cand:
                                continue

                            # Check: if we remove i and j, do we create holes?
                            # Holes would be t-subsets in i∪j with cover==1 that u doesn't cover
                            # But we required u covers all crit (cover==1 in i or j), so safe.
                            # Also need: t-subsets in i∪j with cover==2+ are still covered after removal.
                            # Since cover>=2 for those, removing i and j (each contributes 1) leaves cover>=0.
                            # Actually cover[i_seg] includes i's contribution. After removing i, cover drops by 1 for i_seg.
                            # We need cover[i_seg] - 1 >= 0 AND for j_seg similarly, AND for overlap handled.
                            # Simpler: just do it and verify.
                            i_seg_copy = i_seg.copy()
                            j_seg_copy = j_seg.copy()
                            cover[i_seg_copy] -= 1
                            cover[j_seg_copy] -= 1

                            if (cover >= 0).all():
                                u = unused_cand[rng_local.randrange(len(unused_cand))]
                                u_seg = flat_idx[offsets[u]:offsets[u+1]]
                                cover[u_seg] += 1

                                # Verify no holes
                                if (cover >= 1).all():
                                    # Accept: remove i and j, add u
                                    selected.remove(i)
                                    selected.remove(j)
                                    selected.append(u)
                                    used.discard(i)
                                    used.discard(j)
                                    used.add(u)
                                    changed = True
                                    break
                            else:
                                # Revert
                                cover[i_seg_copy] += 1
                                cover[j_seg_copy] += 1
                                continue
                        if changed:
                            break
                        pairs_checked += 1

        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = list(selected)

    def batch_construct(seed):
        """Build a covering via batch random selection with coverage feedback."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=np.int8)
        selected = []
        used = set()

        # Batch size: pick multiple blocks per iteration
        batch_size = 3

        while uncovered.any() and time.time() < deadline - 10.0:
            # Compute gains for all unused blocks
            gains = A.dot(uncovered)
            gains[list(used)] = -1  # exclude used

            max_gain = gains.max()
            if max_gain <= 0:
                break

            # Pick top candidates and sample from them
            threshold = max_gain - 2  # allow near-optimal
            candidates = np.where((gains >= threshold) & (gains > 0))[0]
            if len(candidates) == 0:
                candidates = np.where(gains == max_gain)[0]

            # Sample batch_size from candidates
            n_pick = min(batch_size, len(candidates))
            picks = rng.sample(list(candidates), n_pick)

            for block_idx in picks:
                if block_idx in used:
                    continue
                selected.append(block_idx)
                used.add(block_idx)
                uncovered[flat_idx[offsets[block_idx]:offsets[block_idx+1]]] = 0

        # Check coverage
        cover = np.zeros(n_tsub, dtype=np.int16)
        for i in selected:
            cover[flat_idx[offsets[i]:offsets[i+1]]] += 1

        # Repair any holes
        holes = np.where(cover == 0)[0]
        for j in holes:
            if time.time() > deadline - 5.0:
                break
            tsub = all_t_subsets[j]
            tset = set(tsub)
            remaining = [p for p in range(v) if p not in tset]
            need = k - t
            if need > 0:
                extra = rng.sample(remaining, need)
            else:
                extra = []
            block = tuple(sorted(list(tsub) + extra))
            # Find index
            for idx, comb in enumerate(all_k_subsets):
                if comb == block:
                    if idx not in used:
                        selected.append(idx)
                        used.add(idx)
                        cover[flat_idx[offsets[idx]:offsets[idx+1]]] += 1
                    break

        return selected, used

    # Multiple restarts with different seeds
    n_restarts = 8
    seeds = list(range(42, 42 + n_restarts))
    rng_main = random.Random(123)
    rng_main.shuffle(seeds)

    for seed in seeds:
        if time.time() > deadline - 15.0:
            break

        selected, used = batch_construct(seed)

        # Aggressively reduce
        try_reduce(selected, used)

        if time.time() > deadline - 5.0:
            break

    # If no valid result found, fall back to simple greedy
    if best_selected is None:
        # Quick greedy fallback
        uncovered = np.ones(n_tsub, dtype=np.int8)
        selected = []
        used = set()
        while uncovered.any() and time.time() < deadline - 2.0:
            gains = A.dot(uncovered)
            gains[list(used)] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best = int(gains.argmax())
            selected.append(best)
            used.add(best)
            uncovered[flat_idx[offsets[best]:offsets[best+1]]] = 0
        best_selected = selected

    # Final safety: ensure full coverage
    cover = np.zeros(n_tsub, dtype=np.int16)
    for i in best_selected:
        cover[flat_idx[offsets[i]:offsets[i+1]]] += 1
    holes = np.where(cover == 0)[0]
    for j in holes:
        if time.time() > deadline - 1.0:
            break
        tsub = all_t_subsets[j]
        tset = set(tsub)
        remaining = [p for p in range(v) if p not in tset]
        need = k - t
        extra = random.Random(j).sample(remaining, need) if need > 0 else []
        block = tuple(sorted(list(tsub) + extra))
        for idx, comb in enumerate(all_k_subsets):
            if comb == block:
                if idx not in set(best_selected):
                    best_selected.append(idx)
                break

    # Deduplicate
    best_selected = list(dict.fromkeys(best_selected))

    result = [list(all_k_subsets[i]) for i in best_selected]
    return result
