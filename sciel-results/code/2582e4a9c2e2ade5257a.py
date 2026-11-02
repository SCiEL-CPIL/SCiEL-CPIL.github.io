import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # explicit safety margin before 150s kill

    # ---------- generic fallback for non-target params ----------
    if v != 19 or k != 8 or t != 5:
        blocks = []
        uncovered = set(itertools.combinations(range(v), t))
        while uncovered:
            if time.time() > deadline:
                break
            target = min(uncovered)
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            best_block, best_gain = None, -1
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain, best_block = gain, block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
        return blocks

    # ---------- precompute coverage structures ----------
    all_t = list(itertools.combinations(range(v), t))
    t_idx = {s: i for i, s in enumerate(all_t)}
    n_t = len(all_t)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t)]
    for j, block in enumerate(all_blocks):
        covered = []
        for ts in itertools.combinations(block, t):
            idx = t_idx[ts]
            covered.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered)

    # numpy views of the static incidence for fast per-block gain reads
    bts_arr = np.array([np.array(lst, dtype=np.int64) for lst in block_to_ts],
                       dtype=object)

    def popcount(x):
        return x.bit_count()

    # ---------- redundancy removal (prune redundant blocks) ----------
    def prune(blocks):
        if not blocks:
            return blocks
        cover = np.zeros(n_t, dtype=np.int32)
        bts_list = []
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                bts_list.append(block_to_ts[idx])
            else:
                bts_list.append([t_idx[ts] for ts in itertools.combinations(b, t)])
        for lst in bts_list:
            for ti in lst:
                cover[ti] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                redundant = True
                for ti in bts_list[i]:
                    if cover[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in bts_list[i]:
                        cover[ti] -= 1
                    blocks.pop(i)
                    bts_list.pop(i)
                    changed = True
                    break
        return blocks

    # ---------- standard greedy (max marginal gain) ----------
    def greedy_gmax(seed):
        rnd = random.Random(seed)
        gain = np.full(n_blocks, k * 7 // 2, dtype=np.int32)  # C(8,5)=56
        for j in range(n_blocks):
            gain[j] = len(block_to_ts[j])
        is_sel = np.zeros(n_blocks, dtype=bool)
        is_unc = np.ones(n_t, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 5.0:
                return None
            maxg = gain[~is_sel].max() if (~is_sel).any() else 0
            if maxg <= 0:
                break
            cands = np.where((gain == maxg) & (~is_sel))[0]
            if len(cands) == 0:
                break
            j = cands[rnd.randrange(len(cands))]
            sol.append(list(all_blocks[j]))
            is_sel[j] = True
            gain[j] = -1000
            new_covered = [ti for ti in block_to_ts[j] if is_unc[ti]]
            if not new_covered:
                continue
            is_unc[new_covered] = False
            to_dec = []
            for ti in new_covered:
                to_dec.extend(ts_to_blocks[ti])
            if to_dec:
                counts = np.bincount(to_dec, minlength=n_blocks)
                gain -= counts
                gain[is_sel] = -1000
        return sol

    # ---------- DUAL greedy (minimize redundancy / maximize uniqueness) ----------
    # Structural change: build a dense, low-overlap core set from scratch.
    # Score a candidate = - (redundant coverage) + (marginal new coverage).
    # This inverts the standard objective to prioritize structural density and
    # minimize overlap, rather than pruning a redundant greedy cover afterwards.
    def greedy_dual(seed):
        rnd = random.Random(seed)
        cover = np.zeros(n_t, dtype=np.int32)
        is_sel = np.zeros(n_blocks, dtype=bool)
        is_unc = np.ones(n_t, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 5.0:
                return None
            if is_unc.all() == False and not is_unc.any():
                break
            best_score = None
            best_j = None
            cands = np.where(~is_sel)[0]
            if len(cands) == 0:
                break
            # sample a subset of candidates to bound per-step cost
            if len(cands) > 4000:
                cands = cands[rnd.sample(range(len(cands)), 4000)]
            for j in cands:
                bts = block_to_ts[j]
                new_cov = 0
                red = 0
                for ti in bts:
                    if is_unc[ti]:
                        new_cov += 1
                    else:
                        red += 1
                score = new_cov - red
                if best_score is None or score > best_score or \
                   (score == best_score and rnd.random() < 0.5):
                    best_score, best_j = score, j
            if best_j is None:
                break
            if best_score <= 0 and not is_unc.any():
                break
            sol.append(list(all_blocks[best_j]))
            is_sel[best_j] = True
            for ti in block_to_ts[best_j]:
                cover[ti] += 1
                is_unc[ti] = False
            if not is_unc.any():
                break
        # if dual alone did not cover everything, top up with standard greedy
        if is_unc.any():
            # reset and finish with a quick greedy over remaining
            rem = np.where(is_unc)[0]
            is_sel2 = is_sel.copy()
            gain = np.zeros(n_blocks, dtype=np.int32)
            for j in range(n_blocks):
                if is_sel2[j]:
                    continue
                gain[j] = sum(1 for ti in block_to_ts[j] if is_unc[ti])
            while is_unc.any():
                if time.time() > deadline - 4.0:
                    break
                maxg = gain[~is_sel2].max() if (~is_sel2).any() else 0
                if maxg <= 0:
                    break
                c2 = np.where((gain == maxg) & (~is_sel2))[0]
                j = c2[rnd.randrange(len(c2))]
                sol.append(list(all_blocks[j]))
                is_sel2[j] = True
                gain[j] = -1000
                for ti in block_to_ts[j]:
                    if is_unc[ti]:
                        is_unc[ti] = False
                for ti in block_to_ts[j]:
                    if is_unc[ti]:
                        continue
                # decrement gains of blocks covering newly covered
                newly = [ti for ti in block_to_ts[j]]
                to_dec = []
                for ti in newly:
                    to_dec.extend(ts_to_blocks[ti])
                if to_dec:
                    counts = np.bincount(to_dec, minlength=n_blocks)
                    gain -= counts
                    gain[is_sel2] = -1000
        return sol

    best_blocks = None
    best_len = float('inf')

    def consider(blocks):
        nonlocal best_blocks, best_len
        if not blocks:
            return
        # dedupe
        seen = set()
        uniq = []
        for b in blocks:
            tb = tuple(b)
            if tb not in seen:
                seen.add(tb)
                uniq.append(b)
        uniq = prune(uniq)
        if len(uniq) < best_len:
            best_len = len(uniq)
            best_blocks = [list(b) for b in uniq]

    # Interleave dual (structural) and gmax (refinement) restarts.
    restart = 0
    while time.time() < deadline - 12.0:
        if restart % 2 == 0:
            sol = greedy_dual(restart * 1013 + 7)
        else:
            sol = greedy_gmax(restart * 991 + 3)
        if sol is not None:
            consider(sol)
        restart += 1

    if best_blocks is None:
        # deterministic fallback greedy
        sol = greedy_gmax(0)
        if sol is None:
            sol = []
        best_blocks = prune(sol)

    return best_blocks
