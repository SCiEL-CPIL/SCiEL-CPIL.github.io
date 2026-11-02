import itertools
import time
import heapq
import random

_DEADLINE = 135.0  # safety margin under 150s wall-clock kill


def _nCr(n, r):
    if r > n or r < 0:
        return 0
    r = min(r, n - r)
    num = 1
    den = 1
    for i in range(r):
        num *= (n - i)
        den *= (i + 1)
    return num // den


def construct(v, k, t):
    t0 = time.time()

    def remaining():
        return _DEADLINE - (time.time() - t0)

    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # For each block, the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        block_ts.append([ts_index[ts] for ts in itertools.combinations(blk, t)])

    # For each t-subset index, which blocks contain it
    ts_to_blocks = [[] for _ in range(n_ts)]
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    def _run_greedy_full(seed, deadline):
        """Heap-based greedy with randomized tie-breaking. Returns (sorted block idx, uncovered)."""
        rng = random.Random(seed)
        gain = [len(x) for x in block_ts]
        selected = set()
        uncovered = [True] * n_ts
        uncovered_count = n_ts

        heap = []
        for i in range(n_blocks):
            jitter = rng.random() * 1e-6
            heapq.heappush(heap, (-(gain[i] + jitter), i, 0))
        version = [0] * n_blocks

        while uncovered_count > 0 and time.time() < deadline:
            entry = None
            while heap:
                neg, b, ver = heapq.heappop(heap)
                if b in selected:
                    continue
                if ver != version[b]:
                    continue
                if gain[b] <= 0:
                    continue
                entry = (neg, b, ver)
                break
            if entry is None:
                break
            neg, b, ver = entry
            selected.add(b)
            for s_idx in block_ts[b]:
                if not uncovered[s_idx]:
                    continue
                uncovered[s_idx] = False
                uncovered_count -= 1
                for ob in ts_to_blocks[s_idx]:
                    if ob in selected:
                        continue
                    gain[ob] -= 1
                    version[ob] += 1
                    if gain[ob] > 0:
                        heapq.heappush(heap, (-gain[ob], ob, version[ob]))
        return sorted(selected), uncovered_count

    def _local_search(sel_idx, deadline):
        """Remove redundant blocks, then 2-for-1 exchange. Preserves validity."""
        sel = list(sel_idx)
        cov = [0] * n_ts
        for b in sel:
            for s in block_ts[b]:
                cov[s] += 1

        # Pass 1: remove redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(sel) - 1, -1, -1):
                if time.time() > deadline:
                    break
                redundant = True
                for s in block_ts[sel[i]]:
                    if cov[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[sel[i]]:
                        cov[s] -= 1
                    del sel[i]
                    changed = True
                    break

        # Pass 2: 2-for-1 exchange
        if time.time() < deadline - 2.0 and len(sel) >= 2:
            attempts = 0
            improved = True
            while improved and time.time() < deadline - 2.0 and attempts < 3000:
                improved = False
                attempts += 1
                n = len(sel)
                for i in range(n):
                    if time.time() > deadline - 2.0:
                        break
                    for j in range(i + 1, n):
                        lost = []
                        lost_set = set()
                        for s in block_ts[sel[i]]:
                            if cov[s] == 1:
                                lost.append(s)
                                lost_set.add(s)
                        for s in block_ts[sel[j]]:
                            if cov[s] == 1 and s not in lost_set:
                                lost.append(s)
                                lost_set.add(s)
                        if not lost:
                            continue
                        found = -1
                        for b in range(n_blocks):
                            if b == sel[i] or b == sel[j]:
                                continue
                            ok = True
                            for s in lost:
                                if s not in block_ts[b]:
                                    ok = False
                                    break
                            if ok:
                                found = b
                                break
                        if found >= 0:
                            for s in block_ts[sel[i]]:
                                cov[s] -= 1
                            for s in block_ts[sel[j]]:
                                cov[s] -= 1
                            for s in block_ts[found]:
                                cov[s] += 1
                            sel = [x for x in sel if x != sel[i] and x != sel[j]]
                            sel.append(found)
                            improved = True
                            break
                    if improved:
                        break
        return sel

    # ---- Multi-restart: baseline greedy first for guaranteed validity ----
    sel, unc = _run_greedy_full(0, time.time() + 30.0)
    best_sel = sel
    best_len = len(sel)

    # Local search on baseline
    ls_deadline = time.time() + max(5.0, min(20.0, remaining() - 15.0))
    if time.time() < ls_deadline:
        best_sel = _local_search(best_sel, ls_deadline)
        best_len = len(best_sel)

    # Additional restarts if budget allows
    for seed in (1, 2, 3, 5, 7, 11, 13):
        if time.time() > _DEADLINE - 25.0:
            break
        g_deadline = time.time() + min(20.0, max(5.0, _DEADLINE - 25.0))
        sel, unc = _run_greedy_full(seed, g_deadline)
        if sel is None or len(sel) >= best_len:
            continue
        ls_deadline = time.time() + max(3.0, min(12.0, _DEADLINE - 15.0))
        sel = _local_search(sel, ls_deadline)
        if len(sel) < best_len:
            best_sel = sel
            best_len = len(sel)
        if time.time() > _DEADLINE - 15.0:
            break

    result = [list(all_blocks[b]) for b in best_sel]
    return result
