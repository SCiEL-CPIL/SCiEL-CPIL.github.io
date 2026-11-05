import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start = time.monotonic()
    deadline = start + 145.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # block -> array of tsub indices it covers
    block_tsubs = []
    for b in all_kblocks:
        tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(np.array(tsubs, dtype=np.int32))

    # tsub -> list of block indices covering it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for ts in tsubs:
            tsub_blocks[ts].append(b_idx)

    tsub_degree = np.array([len(c) for c in tsub_blocks], dtype=np.int32)
    rng = random.Random(12345)

    # ---- 1) Disjoint 4-subset packing (Steiner-like) ----
    # Each block covers C(k,t) tsubs; we want a set of blocks whose covered
    # tsub sets are pairwise disjoint, maximizing total distinct tsubs covered.
    def build_packing(seed):
        r = random.Random(seed)
        covered = np.zeros(num_tsubs, dtype=bool)
        chosen = []
        order = list(range(num_kblocks))
        r.shuffle(order)
        for b_idx in order:
            tsubs = block_tsubs[b_idx]
            if not np.any(covered[tsubs]):
                covered[tsubs] = True
                chosen.append(b_idx)
        return chosen, covered

    # ---- 2) Greedy set cover over the residual deficit ----
    def greedy_cover(covered, seed, budget_blocks=2000):
        r = random.Random(seed)
        uncovered = ~covered
        num_unc = int(np.sum(uncovered))
        chosen = []
        # candidate pool: blocks that cover at least one uncovered tsub
        # We iterate: pick uncovered tsub with fewest available blocks (MRV),
        # then pick the block among its candidates with max uncovered gain.
        for _ in range(budget_blocks):
            if num_unc == 0:
                break
            unc = np.where(uncovered)[0]
            if len(unc) == 0:
                break
            # MRV: fewest candidate blocks
            if len(unc) > 120:
            # sample
                samp = unc[r.choice(unc, size=120, replace=False)]
            else:
                samp = unc
            best_ts = samp[0]
            best_deg = tsub_degree[samp[0]]
            for ts in samp[1:]:
                d = tsub_degree[ts]
                if d < best_deg:
                    best_deg = d
                    best_ts = ts
            cands = tsub_blocks[best_ts]
            if not cands:
                break
            # choose max uncovered gain
            best_b = -1
            best_gain = 0
            for b_idx in cands:
                tsubs = block_tsubs[b_idx]
                g = int(np.sum(uncovered[tsubs]))
                if g > best_gain:
                    best_gain = g
                    best_b = b_idx
            if best_b < 0 or best_gain <= 0:
                break
            chosen.append(best_b)
            tsubs = block_tsubs[best_b]
            newly = int(np.sum(uncovered[tsubs]))
            uncovered[tsubs] = False
            num_unc -= newly
        return chosen, uncovered

    def full_solution(seed):
        packing, covered = build_packing(seed)
        cover, remaining = greedy_cover(covered, seed)
        blocks = packing + cover
        # redundancy removal
        blocks, _ = _remove_redundant(blocks)
        return blocks

    # ---- redundancy removal ----
    def _remove_redundant(blocks):
        if not blocks:
            return [], np.zeros(num_tsubs, dtype=np.int8)
        counts = np.zeros(num_tsubs, dtype=np.int8)
        for b in blocks:
            counts[block_tsubs[b]] += 1
        changed = True
        while changed:
            changed = False
            to_rm = set()
            for b in blocks:
                tsubs = block_tsubs[b]
                if np.all(counts[tsubs] > 1):
                    to_rm.add(b)
                    changed = True
            if to_rm:
                for b in to_rm:
                    counts[block_tsubs[b]] -= 1
                blocks = [b for b in blocks if b not in to_rm]
        return blocks, counts

    # ---- local search: 1-for-1 exchange to shrink ----
    def _local_search(blocks, counts, budget):
        current = list(blocks)
        cur_counts = counts.copy()
        cur_set = set(current)
        improved = True
        while improved:
            if time.monotonic() > budget:
                break
            improved = False
            # remove redundants
            to_rm = [b for b in current if np.all(cur_counts[block_tsubs[b]] > 1)]
            if to_rm:
                for b in to_rm:
                    cur_counts[block_tsubs[b]] -= 1
                    current.remove(b)
                    cur_set.discard(b)
                improved = True
                continue
            # 1-for-1
            for b in list(current):
                if time.monotonic() > budget:
                    break
                tsubs_rm = block_tsubs[b]
                would = tsubs_rm[cur_counts[tsubs_rm] == 1]
                if len(would) == 0:
                    cur_counts[tsubs_rm] -= 1
                    current.remove(b)
                    cur_set.discard(b)
                    improved = True
                    break
                # find single replacement covering all `would`
                cand = None
                for ts in would:
                    cs = set(tsub_blocks[ts])
                    cand = cs if cand is None else (cand & cs)
                    if not cand:
                        break
                if not cand:
                    continue
                cand.discard(b)
                cands = [c for c in cand if c not in cur_set]
                if not cands:
                    continue
                best_r = -1
                best_s = -1
                for c in cands:
                    s = int(np.sum(cur_counts[block_tsubs[c]] == 1))
                    if s > best_s:
                        best_s = s
                        best_r = c
                if best_r >= 0 and best_s > 0:
                    cur_counts[tsubs_rm] -= 1
                    cur_counts[block_tsubs[best_r]] += 1
                    current.remove(b)
                    cur_set.discard(b)
                    current.append(best_r)
                    cur_set.add(best_r)
                    improved = True
                    break
        return current, cur_counts

    def _verify(blocks):
        c = np.zeros(num_tsubs, dtype=np.int8)
        for b in blocks:
            c[block_tsubs[b]] += 1
        return c

    best_blocks = None
    best_len = float('inf')

    # Phase A: several packing+cover restarts
    for s in range(12):
        if time.monotonic() > deadline - 20.0:
            break
        bl = full_solution(s)
        if len(bl) < best_len:
            best_len = len(bl)
            best_blocks = bl[:]

    if best_blocks is None:
        # fallback full greedy
        counts = np.zeros(num_tsubs, dtype=np.int8)
        unc = np.ones(num_tsubs, dtype=bool)
        sel = []
        while np.any(unc):
            if time.monotonic() > deadline - 2.0:
                break
            u = np.where(unc)[0]
            ts = u[0]
            b = tsub_blocks[ts][0]
            sel.append(b)
            counts[block_tsubs[b]] += 1
            unc[block_tsubs[b]] = False
        best_blocks, best_counts = _remove_redundant(sel)
        best_len = len(best_blocks)
    else:
        best_counts = _verify(best_blocks)

    # Phase B: local search on best
    if time.monotonic() < deadline - 15.0:
        polished, pc = _local_search(best_blocks, best_counts, deadline - 5.0)
        if np.all(pc > 0) and len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished[:]
            best_counts = pc.copy()

    # Phase C: ILS perturbation on best
    if time.monotonic() < deadline - 25.0:
        cur = best_blocks[:]
        curc = best_counts.copy()
        il_deadline = deadline - 3.0
        while time.monotonic() < il_deadline:
            r = random.Random()
            # remove a few random blocks
            n_rm = min(3, len(cur))
            to_rm = r.sample(cur, n_rm)
            tmp = list(cur)
            tmpc = curc.copy()
            for b in to_rm:
                tmpc[block_tsubs[b]] -= 1
                tmp.remove(b)
            # greedily repair
            unc = tmpc == 0
            while np.any(unc):
                if time.monotonic() > il_deadline - 2.0:
                    break
                u = np.where(unc)[0]
                ts = u[r.randrange(len(u))]
                cands = tsub_blocks[ts]
                if not cands:
                    break
                best_b = -1
                best_g = 0
                for b in cands:
                    if b in set(tmp):
                        continue
                    g = int(np.sum(unc[block_tsubs[b]]))
                    if g > best_g:
                        best_g = g
                        best_b = b
                if best_b < 0 or best_g <= 0:
                    break
                tmp.append(best_b)
                tmpc[block_tsubs[best_b]] += 1
                unc[block_tsubs[best_b]] = False
            if np.any(tmpc == 0):
                continue
            polished, pc = _local_search(tmp, tmpc, il_deadline - 2.0)
            if np.all(pc > 0) and len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished[:]
                best_counts = pc.copy()
                cur = polished[:]
                curc = pc.copy()

    result = [list(all_kblocks[i]) for i in best_blocks]
    return result
