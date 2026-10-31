"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: count-based fast greedy + aggressive local search (removal / add-then-remove / swap),
with multiple random restarts. Uses a coverage-count array for O(k) removal/swap checks.
"""
import time
import itertools
import random

_DEADLINE = 140.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    # For each k-subset, list of t-subset indices it covers
    coverage = []
    # Reverse: for each t-subset index, list of k-subset indices covering it
    rev_coverage = [[] for _ in range(n_t)]
    for ki, ks in enumerate(k_subsets):
        idxs = [t_index[ts] for ts in itertools.combinations(ks, t)]
        coverage.append(idxs)
        for ti in idxs:
            rev_coverage[ti].append(ki)
    return all_t, t_index, k_subsets, coverage, n_t, rev_coverage


def _greedy_construct(coverage, n_t, rng, n_k, deadline):
    """Greedy set cover using a coverage-count array."""
    cnt = [0] * n_t  # cnt[i] = number of selected blocks covering t-subset i
    blocks = []
    # We track uncovered count
    uncovered = n_t
    # Precompute nothing; iterate all k-subsets each step.
    while uncovered > 0 and _elapsed() < deadline:
        best_i = -1
        best_gain = 0
        for i in range(n_k):
            g = 0
            for idx in coverage[i]:
                if cnt[idx] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_i = i
                if best_gain == 0:
                    break
        if best_i < 0:
            break
        blocks.append(best_i)
        for idx in coverage[best_i]:
            if cnt[idx] == 0:
                uncovered -= 1
            cnt[idx] += 1
    return blocks, cnt, uncovered


def _local_search(blocks, coverage, rev_coverage, n_t, n_k, rng, deadline):
    """Local search minimizing |blocks| while keeping full coverage.
    Uses cnt array for O(k) removal/swap checks.
    """
    blocks = list(blocks)
    cnt = [0] * n_t
    for b in blocks:
        for idx in coverage[b]:
            cnt[idx] += 1
    # Verify full coverage
    full = all(c > 0 for c in cnt)
    if not full:
        return blocks, cnt

    in_blocks = set(blocks)

    improved = True
    while improved and _elapsed() < deadline:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < deadline:
            b = blocks[i]
            # Check if removing b keeps all cnt > 0
            removable = True
            for idx in coverage[b]:
                if cnt[idx] == 1:
                    removable = False
                    break
            if removable:
                for idx in coverage[b]:
                    cnt[idx] -= 1
                blocks.pop(i)
                in_blocks.discard(b)
                improved = True
                continue
            i += 1

        # --- Add-then-remove pass (uses rev_coverage for speed) ---
        for _ in range(50):
            if _elapsed() >= deadline:
                break
            # Find t-subsets with cnt==1 (vulnerable)
            vulnerable = [i for i in range(n_t) if cnt[i] == 1]
            if not vulnerable:
                break
            # Pick a random vulnerable t, find blocks covering it via rev_coverage
            vt = vulnerable[rng.randrange(len(vulnerable))]
            candidates = rev_coverage[vt]
            if not candidates:
                continue
            # Score candidates by how many vulnerable t's they cover
            best_c = -1
            best_score = 0
            for c in candidates:
                if c in in_blocks:
                    continue
                s = 0
                for idx in coverage[c]:
                    if cnt[idx] == 1:
                        s += 1
                if s > best_score:
                    best_score = s
                    best_c = c
            if best_c < 0:
                continue
            # Add best_c
            blocks.append(best_c)
            in_blocks.add(best_c)
            for idx in coverage[best_c]:
                cnt[idx] += 1
            # Now try to remove a block
            removed = False
            for idx in range(len(blocks)):
                b = blocks[idx]
                removable = True
                for tidx in coverage[b]:
                    if cnt[tidx] == 1:
                        removable = False
                        break
                if removable:
                    for tidx in coverage[b]:
                        cnt[tidx] -= 1
                    blocks.pop(idx)
                    in_blocks.discard(b)
                    removed = True
                    improved = True
                    break
            if not removed:
                # Revert
                blocks.pop()
                in_blocks.discard(best_c)
                for idx in coverage[best_c]:
                    cnt[idx] -= 1

        # --- Swap pass: replace one block with another to enable a removal ---
        for _ in range(50):
            if _elapsed() >= deadline:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_k)
            if j in in_blocks:
                continue
            old = blocks[i]
            # Tentatively swap
            for idx in coverage[old]:
                cnt[idx] -= 1
            for idx in coverage[j]:
                cnt[idx] += 1
            blocks[i] = j
            in_blocks.discard(old)
            in_blocks.add(j)
            # Try to remove a block
            removed = False
            for idx in range(len(blocks)):
                b = blocks[idx]
                removable = True
                for tidx in coverage[b]:
                    if cnt[tidx] == 1:
                        removable = False
                        break
                if removable:
                    for tidx in coverage[b]:
                        cnt[tidx] -= 1
                    blocks.pop(idx)
                    in_blocks.discard(b)
                    removed = True
                    improved = True
                    break
            if not removed:
                # Revert swap
                blocks[i] = old
                in_blocks.discard(j)
                in_blocks.add(old)
                for idx in coverage[j]:
                    cnt[idx] -= 1
                for idx in coverage[old]:
                    cnt[idx] += 1

    return blocks, cnt


def _greedy_with_bias(coverage, rev_coverage, n_t, n_k, bias, deadline):
    """Greedy set cover with a frequency-based bias for tie-breaking."""
    cnt = [0] * n_t
    blocks = []
    uncovered = n_t
    while uncovered > 0 and _elapsed() < deadline:
        best_i = -1
        best_gain = 0
        best_bias = -1
        for i in range(n_k):
            g = 0
            for idx in coverage[i]:
                if cnt[idx] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_i = i
                best_bias = bias.get(i, 0)
            elif g == best_gain and g > 0:
                b_val = bias.get(i, 0)
                if b_val > best_bias:
                    best_bias = b_val
                    best_i = i
        if best_i < 0:
            break
        blocks.append(best_i)
        for idx in coverage[best_i]:
            if cnt[idx] == 0:
                uncovered -= 1
            cnt[idx] += 1
    return blocks, cnt, uncovered


def construct(v, k, t):
    global _START
    _START = time.time()

    all_t, t_index, k_subsets, coverage, n_t, rev_coverage = _precompute(v, k, t)
    n_k = len(k_subsets)

    best_blocks = None
    best_size = float('inf')

    # Multiple restarts with different seeds
    seeds = [12345, 67890, 24680, 13579, 99999, 55555, 11111, 77777]
    for seed in seeds:
        if _elapsed() > _DEADLINE - 30:
            break
        rng = random.Random(seed)
        # Shuffle the order of k-subsets to diversify greedy
        order = list(range(n_k))
        rng.shuffle(order)
        # Build a permuted coverage
        cov_perm = [coverage[i] for i in order]

        blocks, cnt, uncovered = _greedy_construct(cov_perm, n_t, rng, n_k, _DEADLINE - 25)
        if uncovered > 0:
            # Repair: add blocks to cover remaining
            while uncovered > 0 and _elapsed() < _DEADLINE - 5:
                best_i = -1
                best_gain = 0
                for i in range(n_k):
                    g = 0
                    for idx in cov_perm[i]:
                        if cnt[idx] == 0:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_i = i
                    if best_gain == 0:
                        break
                if best_i < 0:
                    break
                blocks.append(best_i)
                for idx in cov_perm[best_i]:
                    if cnt[idx] == 0:
                        uncovered -= 1
                    cnt[idx] += 1
        if uncovered > 0:
            continue  # invalid, skip

        # Map back to original indices
        orig_blocks = [order[b] for b in blocks]

        # Local search (needs original coverage)
        # Rebuild cnt in original space
        cnt2 = [0] * n_t
        for b in orig_blocks:
            for idx in coverage[b]:
                cnt2[idx] += 1
        if any(c == 0 for c in cnt2):
            continue

        ls_blocks, ls_cnt = _local_search(orig_blocks, coverage, rev_coverage, n_t, n_k, rng, _DEADLINE - 8)
        if any(c == 0 for c in ls_cnt):
            # invalid, fallback to orig
            ls_blocks = orig_blocks
        if len(ls_blocks) < best_size:
            best_size = len(ls_blocks)
            best_blocks = list(ls_blocks)

    if best_blocks is None:
        # Fallback: pure greedy with seed 12345
        rng = random.Random(12345)
        blocks, cnt, uncovered = _greedy_construct(coverage, n_t, rng, n_k, _DEADLINE - 5)
        best_blocks = blocks

    # Ensure validity
    cnt = [0] * n_t
    for b in best_blocks:
        for idx in coverage[b]:
            cnt[idx] += 1
    if any(c == 0 for c in cnt):
        # Repair
        while any(c == 0 for c in cnt) and _elapsed() < _DEADLINE - 2:
            best_i = -1
            best_gain = 0
            for i in range(n_k):
                g = 0
                for idx in coverage[i]:
                    if cnt[idx] == 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_i = i
            if best_i < 0:
                break
            best_blocks.append(best_i)
            for idx in coverage[best_i]:
                cnt[idx] += 1

    # --- Final polish: rebuild using frequency bias from best solution ---
    if _elapsed() < _DEADLINE - 15 and best_blocks is not None:
        # Count block frequencies in best solution
        freq = {}
        for b in best_blocks:
            freq[b] = freq.get(b, 0) + 1
        # Use biased greedy to try to find a smaller covering
        bias_blocks, bias_cnt, bias_uncovered = _greedy_with_bias(
            coverage, rev_coverage, n_t, n_k, freq, _DEADLINE - 10)
        if bias_uncovered == 0:
            # Run local search on the biased result
            bias_ls, bias_ls_cnt = _local_search(
                bias_blocks, coverage, rev_coverage, n_t, n_k,
                random.Random(42), _DEADLINE - 5)
            if all(c > 0 for c in bias_ls_cnt) and len(bias_ls) < best_size:
                best_blocks = bias_ls
                best_size = len(bias_ls)

    # Deduplicate
    seen = set()
    dedup = []
    for b in best_blocks:
        key = k_subsets[b]
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
