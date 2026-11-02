import time
import random
import math

_DEADLINE = 145.0  # safety margin under the 150 s wall-clock kill


def construct(n_rows, k):
    start = time.time()
    L = k + 1
    total_diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)
    lb = n_rows * total_diffs_per_row  # counting lower bound on scope

    best_rows = None
    best_scope = float('inf')

    def elapsed():
        return time.time() - start

    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    used.add(row[j] - ri)
        return used

    def get_scope(rows):
        return max(r[-1] for r in rows)

    def can_place(row, pos, new_val, used_set):
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        v = new_val
        for i in range(L):
            if i == pos:
                continue
            if i < pos:
                d = v - row[i]
            else:
                d = row[i] - v
            if d in used_set:
                return False
        return True

    def apply_set(row, pos, new_val, used_set):
        old = row[pos]
        for i in range(L):
            if i == pos:
                continue
            if i < pos:
                used_set.discard(old - row[i])
            else:
                used_set.discard(row[i] - old)
        for i in range(L):
            if i == pos:
                continue
            if i < pos:
                used_set.add(new_val - row[i])
            else:
                used_set.add(row[i] - new_val)
        row[pos] = new_val

    def greedy_once(seed):
        """Randomized greedy ruler construction; returns rows and scope."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                # Adaptive window
                window = 15 + int(0.12 * len(used)) + rng.randint(0, 10)
                found = False
                start = last + 1
                end = min(start + window, lb + 300)
                candidates = list(range(start, end))
                rng.shuffle(candidates)
                for x in candidates:
                    nd = {x - a for a in row}
                    if not (nd & used):
                        row.append(x)
                        used |= nd
                        found = True
                        break
                if not found:
                    # Expand search
                    for x in range(start, lb + 500):
                        nd = {x - a for a in row}
                        if not (nd & used):
                            row.append(x)
                            used |= nd
                            found = True
                            break
                    if not found:
                        return None, None
            rows.append(row)
        scope = max(r[-1] for r in rows)
        return rows, scope

    # Phase 1: Establish a good initial upper bound with several randomized greedy tries
    for seed in range(50):
        if elapsed() > _DEADLINE - 20:
            break
        rows, scope = greedy_once(seed * 131 + 17)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Fallback: deterministic greedy
        rows, scope = greedy_once(0)
        if rows is not None:
            best_rows = [list(r) for r in rows]
            best_scope = scope
        else:
            # Very aggressive fallback
            for seed in range(100, 200):
                rows, scope = greedy_once(seed)
                if rows is not None:
                    best_rows = [list(r) for r in rows]
                    best_scope = scope
                    break
            if best_rows is None:
                return None

    def simulated_annealing(rows_init, time_limit):
        """SA to minimize scope by moving individual elements."""
        rows = [list(r) for r in rows_init]
        used = build_used(rows)
        current_scope = get_scope(rows)
        best_local = current_scope
        best_rows_local = [list(r) for r in rows]

        T = 50.0
        T_min = 0.1
        cooling = 0.997
        rng = random.Random(int(time.time() * 1000))

        while time.time() < time_limit and T > T_min:
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)  # skip pos 0
            row = rows[r_idx]
            old_val = row[pos]

            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < L - 1 else old_val + 30

            if lower >= upper:
                T *= cooling
                continue

            # Choose new value: biased toward reducing scope
            if pos == L - 1 and rng.random() < 0.6:
                new_val = rng.randint(lower, min(old_val, upper))
            else:
                new_val = rng.randint(lower, max(lower, upper))

            if new_val == old_val:
                T *= cooling
                continue

            if not can_place(row, pos, new_val, used):
                T *= cooling
                continue

            apply_set(row, pos, new_val, used)
            new_scope = get_scope(rows)

            delta = new_scope - current_scope
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                current_scope = new_scope
                if current_scope < best_local:
                    best_local = current_scope
                    best_rows_local = [list(r) for r in rows]
            else:
                apply_set(row, pos, old_val, used)

            T *= cooling

        return best_rows_local, best_local

    def hill_climb_compact(rows, time_limit):
        """Aggressive local compaction: repeatedly try to reduce each element."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = [(r, p) for r in range(n_rows) for p in range(L - 1, 0, -1)]
            random.shuffle(order)
            for r_idx, pos in order:
                if time.time() >= time_limit:
                    break
                row = work[r_idx]
                old_val = row[pos]
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < L - 1 else old_val
                if lower >= old_val:
                    continue
                for new_val in range(lower, min(old_val, upper + 1)):
                    if can_place(row, pos, new_val, used_set):
                        apply_set(row, pos, new_val, used_set)
                        improved = True
                        break
        return work

    # Phase 2: Simulated annealing on the best solution
    if elapsed() < _DEADLINE - 10:
        sa_rows, sa_scope = simulated_annealing(best_rows, time.time() + min(60, _DEADLINE - 10 - elapsed()))
        if sa_rows is not None and sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Phase 3: Hill-climb compaction
    if elapsed() < _DEADLINE - 5:
        hc_rows = hill_climb_compact(best_rows, time.time() + min(30, _DEADLINE - 5 - elapsed()))
        hc_scope = get_scope(hc_rows)
        if hc_scope < best_scope:
            best_scope = hc_scope
            best_rows = hc_rows

    # Phase 4: Another round of SA + hill-climb if time remains
    if elapsed() < _DEADLINE - 5:
        sa_rows, sa_scope = simulated_annealing(best_rows, time.time() + min(40, _DEADLINE - 5 - elapsed()))
        if sa_rows is not None and sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
        if elapsed() < _DEADLINE - 3:
            hc_rows = hill_climb_compact(best_rows, time.time() + min(15, _DEADLINE - 3 - elapsed()))
            hc_scope = get_scope(hc_rows)
            if hc_scope < best_scope:
                best_scope = hc_scope
                best_rows = hc_rows

    # Final validation
    all_diffs = set()
    valid = True
    for r in best_rows:
        for i in range(L):
            for j in range(i + 1, L):
                d = r[j] - r[i]
                if d in all_diffs:
                    valid = False
                    break
                all_diffs.add(d)
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to a valid greedy solution
        for seed in range(10):
            rows, scope = greedy_once(seed * 997 + 3)
            if rows is not None:
                # Validate
                ad = set()
                ok = True
                for r in rows:
                    for i in range(L):
                        for j in range(i + 1, L):
                            d = r[j] - r[i]
                            if d in ad:
                                ok = False
                                break
                            ad.add(d)
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    return rows
        return None

    return best_rows
