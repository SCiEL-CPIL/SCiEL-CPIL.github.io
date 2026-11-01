import time
import random
import math

try:
    import numpy as np
except Exception:
    np = None


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before the 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    rng = random.Random(12345)

    # ---------------- evaluation helpers ----------------
    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                ri = r[i]
                for j in range(i + 1, row_len):
                    d = r[j] - ri
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for c in diff_counts.values():
            if c > 1:
                conflicts += c - 1
        return conflicts

    # ---------------- bitmask conflict machinery ----------------
    def build_mask(rows):
        if np is not None:
            n = len(rows)
            arr = np.array(rows, dtype=np.int64)
            idx = np.arange(row_len)
            diffs = arr[:, None, :] - arr[:, :, None]  # (n, row_len, row_len)
            ii = idx[None, None, :]
            jj = idx[None, :, None]
            m = np.triu(diffs, k=1) * (jj > ii)
            flat = m.ravel()
            flat = flat[flat > 0]
            if flat.size == 0:
                return 0
            if flat.max() < 2**53:
                return int(np.bitwise_or.reduce(flat.astype(np.uint64)))
            return int(np.bitwise_or.reduce(flat.astype(object)))
        # pure-python fallback
        mask = 0
        for r in rows:
            for i in range(row_len):
                ri = r[i]
                for j in range(i + 1, row_len):
                    mask |= (1 << (r[j] - ri))
        return mask

    def row_mask(row):
        if np is not None:
            arr = np.array(row, dtype=np.int64)
            idx = np.arange(row_len)
            diffs = arr[None, :] - arr[:, None]
            ii = idx[None, :]
            jj = idx[:, None]
            m = np.triu(diffs, k=1) * (jj > ii)
            flat = m.ravel()
            flat = flat[flat > 0]
            if flat.size == 0:
                return 0
            if flat.max() < 2**53:
                return int(np.bitwise_or.reduce(flat.astype(np.uint64)))
            return int(np.bitwise_or.reduce(flat.astype(object)))
        mask = 0
        for i in range(row_len):
            ri = row[i]
            for j in range(i + 1, row_len):
                mask |= (1 << (row[j] - ri))
        return mask

    # ---------------- greedy initialisation ----------------
    def greedy_init(rand_prob=0.1, window=20):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                cands = []
                x = last + 1
                limit = 400
                while x <= limit and len(cands) < window:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        cands.append(x)
                    x += 1
                if not cands:
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            cands.append(x)
                            break
                if rng.random() < rand_prob and len(cands) > 1:
                    x = rng.choice(cands)
                else:
                    x = cands[0]
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
            rows.append(row)
        return rows

    # ---------------- targeted descent (reduce global max) ----------------
    def local_search_targeted(rows, time_limit):
        start = time.time()
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        while time.time() < start + time_limit:
            max_val = -1
            mr = mc = -1
            for ri, r in enumerate(best):
                for ci, v in enumerate(r):
                    if v > max_val:
                        max_val, mr, mc = v, ri, ci
            if mc == 0:
                break
            row = best[mr]
            cur = max_val
            prev = row[mc - 1]
            allm = build_mask(best)
            rm = row_mask(row)
            for i in range(mc):
                rm &= ~(1 << (cur - row[i]))
            for j in range(mc + 1, row_len):
                rm &= ~(1 << (row[j] - cur))
            fixed = allm & ~rm
            found = False
            for nv in range(cur - 1, prev, -1):
                nm = 0
                for i in range(mc):
                    nm |= 1 << (nv - row[i])
                for j in range(mc + 1, row_len):
                    nm |= 1 << (row[j] - nv)
                if nm & fixed:
                    continue
                row[mc] = nv
                found = True
                break
            if not found:
                # random jolt of a non-max element
                ri = rng.randrange(n_rows)
                ci = rng.randrange(1, row_len)
                row = best[ri]
                cv = row[ci]
                pv = row[ci - 1]
                nv2 = row[ci + 1] if ci < row_len - 1 else cv + 100
                allm = build_mask(best)
                rm = row_mask(row)
                for i in range(ci):
                    rm &= ~(1 << (cv - row[i]))
                for j in range(ci + 1, row_len):
                    rm &= ~(1 << (row[j] - cv))
                fixed = allm & ~rm
                lo = pv + 1
                hi = nv2 - 1 if ci < row_len - 1 else cv + 50
                if lo > hi:
                    continue
                for _ in range(40):
                    cand = rng.randrange(lo, hi + 1)
                    if cand == cv:
                        continue
                    nm = 0
                    for i in range(ci):
                        nm |= 1 << (cand - row[i])
                    for j in range(ci + 1, row_len):
                        nm |= 1 << (row[j] - cand)
                    if nm & fixed:
                        continue
                    row[ci] = cand
                    break
        return best, get_scope(best)

    # ---------------- simulated annealing ----------------
    def local_search_sa(rows, time_limit):
        cur = [r[:] for r in rows]
        cur_mask = build_mask(cur)
        cur_scope = get_scope(cur)
        best = [r[:] for r in cur]
        best_scope = cur_scope
        T = 12.0
        T_min = 0.01
        cooling = 0.996
        start = time.time()
        while time.time() < start + time_limit and T >= T_min:
            ri = rng.randrange(n_rows)
            pos = rng.randrange(1, row_len)
            row = cur[ri]
            ov = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < row_len - 1 else cur_scope + 10
            if lo > hi:
                continue
            v = rng.randrange(lo, hi + 1)
            if v == ov:
                continue
            rm = row_mask(row)
            for i in range(pos):
                rm &= ~(1 << (ov - row[i]))
            for j in range(pos + 1, row_len):
                rm &= ~(1 << (row[j] - ov))
            fixed = cur_mask & ~rm
            nm = 0
            for i in range(pos):
                nm |= 1 << (v - row[i])
            for j in range(pos + 1, row_len):
                nm |= 1 << (row[j] - v)
            new_mask = fixed | nm
            new_conf = new_mask.bit_count() - total_diffs
            row[pos] = v
            new_scope = max(cur_scope, v)
            dconf = new_conf - (cur_mask.bit_count() - total_diffs)
            dscope = new_scope - cur_scope
            accept = False
            if new_conf < cur_mask.bit_count() - total_diffs:
                accept = True
            elif new_conf == cur_mask.bit_count() - total_diffs:
                if new_scope < cur_scope:
                    accept = True
                else:
                    if rng.random() < math.exp(-dscope / T):
                        accept = True
            else:
                if rng.random() < math.exp(-dconf / T):
                    accept = True
            if accept:
                cur_mask = new_mask
                cur_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in cur]
            else:
                row[pos] = ov
            T *= cooling
        return best, best_scope

    # ---------------- ILS kick: rebuild one row from scratch ----------------
    def rebuild_row(used_other, max_val, rand_prob=0.2):
        used = set(used_other)
        row = [0]
        for _ in range(row_len - 1):
            last = row[-1]
            cands = []
            x = last + 1
            limit = max_val + 1
            while x <= limit and len(cands) < 40:
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    cands.append(x)
                x += 1
            if not cands:
                return None
            if rand_prob > 0 and rng.random() < rand_prob and len(cands) > 1:
                x = rng.choice(cands)
            else:
                x = cands[0]
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
        return row

    def local_polish_il(rows, time_limit):
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        best = [r[:] for r in cur]
        best_scope = cur_scope
        start = time.time()
        while time.time() < start + time_limit:
            # rebuild a random row
            ri = rng.randrange(n_rows)
            others = cur[:ri] + cur[ri + 1:]
            om = build_mask(others)
            newrow = rebuild_row(om, cur_scope, rand_prob=0.25)
            if newrow is not None:
                newrow = newrow[:row_len]
                cand = cur[:ri] + [newrow] + cur[ri + 1:]
                cconf = build_mask(cand).bit_count() - total_diffs
                cscope = get_scope(cand)
                if cconf == 0 and cscope <= cur_scope:
                    cur = cand
                    cur_scope = cscope
                    if cscope < best_scope:
                        best_scope = cscope
                        best = [r[:] for r in cur]
                else:
                    # accept small worsening to escape, then repair via targeted
                    if cconf <= 1 and rng.random() < 0.5:
                        cur = cand
                        cur_scope = max(cur_scope, cscope)
            # short targeted descent to repair/trim
            if time.time() < start + time_limit - 0.05:
                cur, s = local_search_targeted(cur, 0.5)
                if s < best_scope:
                    best_scope = s
                    best = [r[:] for r in cur]
        return best, best_scope

    # ---------------- main multi-start loop ----------------
    best_rows = None
    best_scope = float('inf')
    best_conf = float('inf')

    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 12:
            break
        rand_prob = 0.05 + 0.04 * i
        window = 10 + 4 * i

        rows = greedy_init(rand_prob, window)
        conf = count_conflicts(rows)
        scope = get_scope(rows)

        # Phase 1: targeted descent
        t1, s1 = local_search_targeted(rows, 5.0)
        c1 = count_conflicts(t1)
        if c1 < best_conf or (c1 == best_conf and s1 < best_scope):
            best_conf, best_scope, best_rows = c1, s1, t1

        # Phase 2: ILS kick (structural escape)
        if time.time() > deadline - 12:
            break
        t2, s2 = local_polish_il(t1, 6.0)
        c2 = count_conflicts(t2)
        if c2 < best_conf or (c2 == best_conf and s2 < best_scope):
            best_conf, best_scope, best_rows = c2, s2, t2

        # Phase 3: SA polish
        if time.time() > deadline - 12:
            break
        t3, s3 = local_search_sa(t2, 4.0)
        c3 = count_conflicts(t3)
        if c3 < best_conf or (c3 == best_conf and s3 < best_scope):
            best_conf, best_scope, best_rows = c3, s3, t3

        if best_conf == 0 and best_scope < 120:
            break

    # ---------------- final polish with remaining time ----------------
    if best_rows is not None:
        remaining = deadline - time.time() - 2.0
        if remaining > 8:
            half = remaining / 2
            f1, fs1 = local_polish_il(best_rows, half)
            fc1 = count_conflicts(f1)
            if fc1 < best_conf or (fc1 == best_conf and fs1 < best_scope):
                best_conf, best_scope, best_rows = fc1, fs1, f1
            if time.time() < deadline - 2.0:
                f2, fs2 = local_search_sa(best_rows, deadline - time.time() - 2.0)
                fc2 = count_conflicts(f2)
                if fc2 < best_conf or (fc2 == best_conf and fs2 < best_scope):
                    best_conf, best_scope, best_rows = fc2, fs2, f2

    if best_rows is None:
        best_rows = greedy_init(0.1, 20)

    return best_rows
