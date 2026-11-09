import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    def scope(rows):
        return max(max(r) for r in rows)

    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def involved_diffs(row, pos, val):
        s = set()
        for i in range(pos):
            s.add(val - row[i])
        for j in range(pos + 1, len(row)):
            s.add(row[j] - val)
        return s

    def interleaved_greedy():
        used = set()
        rows = [[0] for _ in range(n_rows)]
        while any(len(r) < k + 1 for r in rows):
            for ri in range(n_rows):
                row = rows[ri]
                if len(row) >= k + 1:
                    continue
                x = row[-1] + 1
                while True:
                    nd = {x - a for a in row}
                    if not (nd & used):
                        row.append(x)
                        used |= nd
                        break
                    x += 1
        return rows, used

    def sequential_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    nd = {x - a for a in row}
                    if not (nd & used):
                        row.append(x)
                        used |= nd
                        break
                    x += 1
            rows.append(row)
        return rows, used

    # Try multiple initializations
    candidates = []
    rows_a, used_a = interleaved_greedy()
    candidates.append((scope(rows_a), [r[:] for r in rows_a], set(used_a)))
    rows_b, used_b = sequential_greedy()
    candidates.append((scope(rows_b), [r[:] for r in rows_b], set(used_b)))

    # A few random-perturbed interleaved starts
    for _ in range(4):
        random.seed(int(time.time() * 1000) % (2**31))
        rows_c, used_c = interleaved_greedy()
        # Perturb: for each row, try to shift some elements
        perturbed = [r[:] for r in rows_c]
        pused = set(used_c)
        for ri in range(n_rows):
            for pos in range(1, k + 1):
                if random.random() < 0.3:
                    old = perturbed[ri][pos]
                    lo = perturbed[ri][pos-1] + 1
                    hi = perturbed[ri][pos+1] - 1 if pos < k else old + 10
                    for nv in range(old, max(lo, old - 8), -1):
                        if nv < lo:
                            break
                        oi = involved_diffs(perturbed[ri], pos, old)
                        ni = involved_diffs(perturbed[ri], pos, nv)
                        if len(ni) != len(oi):
                            continue
                        pused -= oi
                        if not (ni & pused):
                            pused |= ni
                            perturbed[ri][pos] = nv
                            old = nv
                            break
                        pused |= oi
        candidates.append((scope(perturbed), perturbed, pused))

    candidates.sort(key=lambda x: x[0])
    best_scope, best_rows, best_used = candidates[0]
    current_scope = best_scope
    rows = best_rows
    used_diffs = best_used

    # Hill climbing: greedily reduce elements
    def hill_climb(rows, used, max_iters=30):
        cur_scope = scope(rows)
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 10:
            improved = False
            it += 1
            order = list(range(n_rows))
            if it > 1:
                random.shuffle(order)
            for ri in order:
                if time.time() > deadline - 10:
                    break
                row = rows[ri]
                for pos in range(1, k + 1):
                    ov = row[pos]
                    lo = row[pos - 1] + 1
                    hi = row[pos + 1] - 1 if pos < k else ov
                    if lo >= ov:
                        continue
                    for nv in range(lo, ov):
                        oi = involved_diffs(row, pos, ov)
                        ni = involved_diffs(row, pos, nv)
                        if len(ni) != len(oi):
                            continue
                        used -= oi
                        if not (ni & used):
                            used |= ni
                            row[pos] = nv
                            ns = scope(rows)
                            if ns < cur_scope:
                                cur_scope = ns
                                improved = True
                                break
                            else:
                                used -= ni
                                used |= oi
                                row[pos] = ov
                        else:
                            used |= oi
                    if improved:
                        break
                if improved:
                    break
        return rows, used, cur_scope

    rows, used_diffs, current_scope = hill_climb(rows, used_diffs, max_iters=20)
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [r[:] for r in rows]

    # Simulated annealing
    T = 8.0
    cooling = 0.9985
    min_T = 0.3
    rebuild_prob = 0.25

    def rebuild_row(ri, rows, used):
        old_row = rows[ri]
        old_d = row_diffs(old_row)
        trial_used = used - old_d
        new_row = [0]
        for pos in range(1, k + 1):
            x = new_row[-1] + 1
            found = False
            limit = old_row[-1] + 30
            while x <= limit:
                nd = {x - a for a in new_row}
                if not (nd & trial_used):
                    new_row.append(x)
                    trial_used |= nd
                    found = True
                    break
                x += 1
            if not found:
                return None
        return new_row, trial_used

    while time.time() < deadline - 3:
        if random.random() < rebuild_prob:
            ri = random.randint(0, n_rows - 1)
            result = rebuild_row(ri, rows, used_diffs)
            if result:
                new_row, new_used = result
                new_scope_val = scope([rows[i] if i != ri else new_row for i in range(n_rows)])
                if new_scope_val < current_scope:
                    rows[ri] = new_row
                    used_diffs = new_used
                    current_scope = new_scope_val
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    delta = new_scope_val - current_scope
                    if random.random() < math.exp(-delta / T):
                        rows[ri] = new_row
                        used_diffs = new_used
                        current_scope = new_scope_val
        else:
            ri = random.randint(0, n_rows - 1)
            row = rows[ri]
            pos = random.randint(1, k)
            ov = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < k else ov + 20
            if pos == k:
                if lo < ov:
                    nv = random.randint(lo, ov - 1)
                else:
                    T *= cooling
                    continue
            else:
                s_lo = max(lo, ov - 12)
                s_hi = min(hi, ov + 12)
                if s_lo > s_hi:
                    T *= cooling
                    continue
                nv = random.randint(s_lo, s_hi)
            if nv == ov:
                T *= cooling
                continue
            oi = involved_diffs(row, pos, ov)
            ni = involved_diffs(row, pos, nv)
            if len(ni) != len(oi):
                T *= cooling
                continue
            used_diffs -= oi
            if ni & used_diffs:
                used_diffs |= oi
                T *= cooling
                continue
            used_diffs |= ni
            row[pos] = nv
            ns = scope(rows)
            if ns < current_scope:
                current_scope = ns
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                delta = ns - current_scope
                if random.random() < math.exp(-delta / T):
                    current_scope = ns
                else:
                    used_diffs -= ni
                    used_diffs |= oi
                    row[pos] = ov
        T *= cooling
        if T < min_T:
            T = 8.0
            # Quick hill climb after restart
            rows, used_diffs, current_scope = hill_climb(rows, used_diffs, max_iters=10)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]

    # Final polish on best_rows
    final_used = set()
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                final_used.add(r[j] - r[i])
    fs = scope(best_rows)
    for _ in range(5):
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            for pos in range(1, k + 1):
                ov = row[pos]
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1 if pos < k else ov
                if lo >= ov:
                    continue
                for nv in range(lo, ov):
                    oi = involved_diffs(row, pos, ov)
                    ni = involved_diffs(row, pos, nv)
                    if len(ni) != len(oi):
                        continue
                    final_used -= oi
                    if not (ni & final_used):
                        final_used |= ni
                        row[pos] = nv
                        ns = scope(best_rows)
                        if ns < fs:
                            fs = ns
                            improved = True
                            break
                        else:
                            final_used -= ni
                            final_used |= oi
                            row[pos] = ov
                    else:
                        final_used |= oi
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_rows
