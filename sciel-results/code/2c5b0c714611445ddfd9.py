import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    total_diffs = k * (k + 1) // 2
    counting_lb = n_rows * total_diffs
    row_len = k + 1
    scope_bits = 3000  # bits for difference tracking
    mask = (1 << scope_bits) - 1

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def diff_bits(row):
        b = 0
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                b |= (1 << (row[j] - ri))
        return b

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def score_of(rows):
        return max(max(r) for r in rows)

    # --- Bitset greedy row builder (fast inner loop) ---
    def greedy_row_bits(forbidden_bits, max_hint):
        row = [0]
        local_bits = 0
        upper = max_hint if max_hint else 2000
        for _ in range(k):
            base = row[-1]
            found = False
            for v in range(base + 1, upper + 1):
                ok = True
                new_bits = 0
                for a in row:
                    d = v - a
                    bit = 1 << d
                    if (forbidden_bits & bit) or (local_bits & bit) or (new_bits & bit):
                        ok = False
                        break
                    new_bits |= bit
                if ok:
                    row.append(v)
                    local_bits |= new_bits
                    found = True
                    break
            if not found:
                return None
        return row

    # --- Initial construction via greedy ---
    def initial_construct(max_hint):
        rows = []
        used_bits = 0
        for i in range(n_rows):
            r = greedy_row_bits(used_bits, max_hint)
            if r is None:
                return None
            rows.append(r)
            used_bits |= diff_bits(r)
        return rows

    hint = max(counting_lb + 10, 130)
    init_rows = initial_construct(hint)
    if init_rows is None:
        init_rows = initial_construct(counting_lb + 60)
    if init_rows is None:
        init_rows = initial_construct(counting_lb + 200)
    if init_rows is None:
        rows = []
        used_bits = 0
        for i in range(n_rows):
            row = [0]
            local_bits = 0
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    new_bits = 0
                    for a in row:
                        d = v - a
                        bit = 1 << d
                        if (used_bits & bit) or (local_bits & bit) or (new_bits & bit):
                            ok = False
                            break
                        new_bits |= bit
                    if ok:
                        break
                    v += 1
                row.append(v)
                local_bits |= new_bits
            rows.append(row)
            used_bits |= diff_bits(row)
        init_rows = rows

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)
    cur_rows = [r[:] for r in init_rows]
    cur_row_bits = [diff_bits(r) for r in cur_rows]
    cur_score = best_score

    def forbidden_except(idx):
        fb = 0
        for i in range(n_rows):
            if i != idx:
                fb |= cur_row_bits[i]
        return fb

    # --- Bitset backtracking row builder with limited nodes ---
    def build_row_backtrack(forbidden_bits, scope, max_nodes=20000, rng=None):
        if rng is None:
            rng = random
        nodes = [0]
        target = row_len

        def dfs(row, local_bits, depth):
            if nodes[0] > max_nodes:
                return None
            if time.time() > deadline - 5:
                return None
            if len(row) == target:
                return list(row)
            base = row[-1]
            rem = target - len(row)
            max_next = scope - (rem - 1)
            if base + 1 > max_next:
                return None
            # Try candidates in order, with occasional randomization
            cands = list(range(base + 1, max_next + 1))
            if rng.random() < 0.3 and len(cands) > 6:
                rng.shuffle(cands)
            for v in cands:
                nodes[0] += 1
                if nodes[0] > max_nodes:
                    return None
                ok = True
                new_bits = 0
                for a in row:
                    d = v - a
                    bit = 1 << d
                    if (forbidden_bits & bit) or (local_bits & bit) or (new_bits & bit):
                        ok = False
                        break
                    new_bits |= bit
                if not ok:
                    continue
                row.append(v)
                res = dfs(row, local_bits | new_bits, depth + 1)
                row.pop()
                if res is not None:
                    return res
            return None

        return dfs([0], 0, 0)

    # --- Iterative row-replacement refinement (from inspiration) ---
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        order = sorted(range(n_rows), key=lambda i: cur_rows[i][-1], reverse=True)
        for i in order:
            if time.time() > deadline - 20:
                break
            cur_max_row = cur_rows[i][-1]
            target = cur_max_row - 1
            fb = forbidden_except(i)
            res = build_row_backtrack(fb, target, max_nodes=30000)
            if res is not None:
                new_rows = cur_rows[:]
                new_rows[i] = res
                if is_valid(new_rows):
                    cur_rows = new_rows
                    cur_row_bits[i] = diff_bits(res)
                    cur_score = score_of(cur_rows)
                    if cur_score < best_score:
                        best_score = cur_score
                        best_rows = [r[:] for r in cur_rows]
                    improved = True

    # --- Simulated annealing with bitset conflict checking ---
    T = 15.0
    cooling = 0.9995
    min_T = 0.05
    skip_choices = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 4]

    def sample_skips():
        return [random.choice(skip_choices) for _ in range(k)]

    def build_row_with_skips(forbidden_bits, skips, max_hint):
        row = [0]
        local_bits = 0
        for t in range(k):
            base = row[-1]
            cands = []
            for v in range(base + 1, min(base + 500, max_hint + 1)):
                ok = True
                new_bits = 0
                for a in row:
                    d = v - a
                    bit = 1 << d
                    if (forbidden_bits & bit) or (local_bits & bit) or (new_bits & bit):
                        ok = False
                        break
                    new_bits |= bit
                if ok:
                    cands.append(v)
                    if len(cands) >= 8:
                        break
            if not cands:
                # fallback: find first valid
                v = base + 1
                while True:
                    ok = True
                    new_bits = 0
                    for a in row:
                        d = v - a
                        bit = 1 << d
                        if (forbidden_bits & bit) or (local_bits & bit) or (new_bits & bit):
                            ok = False
                            break
                        new_bits |= bit
                    if ok:
                        break
                    v += 1
                cands = [v]
            idx = skips[t]
            if idx < 0:
                idx = 0
            if idx >= len(cands):
                idx = 0
            v = cands[idx]
            new_bits = 0
            for a in row:
                new_bits |= (1 << (v - a))
            row.append(v)
            local_bits |= new_bits
        if row[-1] > max_hint:
            return None
        return row

    while time.time() < deadline - 5:
        if T < min_T:
            T = 10.0

        idx = random.randrange(n_rows)
        fb = forbidden_except(idx)
        skips = sample_skips()
        max_hint = best_score + 5
        new_row = build_row_with_skips(fb, skips, max_hint)
        if new_row is None:
            T *= cooling
            continue

        new_row_b = diff_bits(new_row)
        if (new_row_b & fb) != 0:
            T *= cooling
            continue

        new_rows = cur_rows[:]
        new_rows[idx] = new_row
        ns = score_of(new_rows)

        delta = ns - cur_score
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.05:
                accept = True
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accept = True

        if accept:
            cur_rows[idx] = new_row
            cur_row_bits[idx] = new_row_b
            cur_score = ns
            if cur_score < best_score:
                best_score = cur_score
                best_rows = [r[:] for r in cur_rows]

        T *= cooling

    # Final validation and fallback
    if not is_valid(best_rows):
        rows = []
        used_bits = 0
        for i in range(n_rows):
            row = [0]
            local_bits = 0
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    new_bits = 0
                    for a in row:
                        d = v - a
                        bit = 1 << d
                        if (used_bits & bit) or (local_bits & bit) or (new_bits & bit):
                            ok = False
                            break
                        new_bits |= bit
                    if ok:
                        break
                    v += 1
                row.append(v)
                local_bits |= new_bits
            rows.append(row)
            used_bits |= diff_bits(row)
        return rows

    return best_rows

# EVOLVE-BLOCK-END
