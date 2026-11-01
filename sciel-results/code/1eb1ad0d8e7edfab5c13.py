import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 135.0  # Safety margin (kill at 150s)

    d = k * (k + 1) // 2  # differences per row

    def greedy_init(rng):
        """Greedy construction: build rows one by one, picking smallest valid element."""
        rows = []
        global_diffs = set()
        for r in range(n_rows):
            row = [0]
            for _ in range(k):
                cand = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        diff = cand - a
                        if diff in global_diffs:
                            ok = False
                            break
                    if ok:
                        row.append(cand)
                        for a in row[:-1]:
                            global_diffs.add(cand - a)
                        break
                    cand += 1
                    if cand > 5000:
                        return None
            rows.append(row)
        return rows

    def compute_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check all positive differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return False
                    seen.add(diff)
        return True

    def anneal(rows, rng, time_limit):
        """Simulated annealing to minimize scope."""
        rows = [r[:] for r in rows]
        global_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    global_diffs.add(row[j] - row[i])

        scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = scope

        t_end = time.time() + time_limit
        temp = max(1.0, scope * 0.1)
        cooling = 0.9995

        while time.time() < t_end:
            if temp < 0.01:
                temp = 0.01

            # Choose a random row
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]

            # Choose a random index to move (1..k)
            i = rng.randint(1, k)
            old_val = row[i]

            # Compute old differences involving index i
            old_diffs = set()
            for j in range(k + 1):
                if j == i:
                    continue
                old_diffs.add(abs(row[i] - row[j]))

            # Remove old diffs
            for diff in old_diffs:
                global_diffs.discard(diff)

            # Try a new value: random in a range around current, biased toward smaller
            # Range: from row[i-1]+1 to some upper bound
            lo = row[i - 1] + 1
            # Upper bound: old_val + some slack, but not too large
            hi = max(old_val, lo + 20)
            if old_val > lo:
                # Bias toward smaller values
                if rng.random() < 0.7:
                    hi = old_val  # try to decrease
                    lo = max(lo, old_val - 30)
                else:
                    lo = old_val
                    hi = old_val + 30

            if hi <= lo:
                hi = lo + 1

            new_val = rng.randint(lo, hi)

            # Check validity
            new_diffs = set()
            valid = True
            for j in range(k + 1):
                if j == i:
                    continue
                diff = abs(new_val - row[j])
                if diff in global_diffs:
                    valid = False
                    break
                new_diffs.add(diff)

            if not valid:
                # Revert
                for diff in old_diffs:
                    global_diffs.add(diff)
                temp *= cooling
                continue

            # Compute new scope
            row[i] = new_val
            for diff in new_diffs:
                global_diffs.add(diff)
            new_scope = compute_scope(rows)

            delta = new_scope - scope
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 0.01)):
                scope = new_scope
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: revert
                row[i] = old_val
                for diff in new_diffs:
                    global_diffs.discard(diff)
                for diff in old_diffs:
                    global_diffs.add(diff)

            temp *= cooling

        return best_rows, best_scope

    def local_refine(rows, rng, time_limit):
        """Local search: try to decrease each element to its minimum valid value."""
        rows = [r[:] for r in rows]
        global_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    global_diffs.add(row[j] - row[i])

        scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = scope

        t_end = time.time() + time_limit
        no_improve = 0

        while time.time() < t_end and no_improve < 50:
            improved = False
            # Try each row, each index from end to start
            for r_idx in range(n_rows - 1, -1, -1):
                for i in range(k, 0, -1):
                    row = rows[r_idx]
                    old_val = row[i]
                    min_val = row[i - 1] + 1
                    if old_val <= min_val:
                        continue

                    old_diffs = set()
                    for j in range(k + 1):
                        if j == i:
                            continue
                        old_diffs.add(abs(row[i] - row[j]))

                    for diff in old_diffs:
                        global_diffs.discard(diff)

                    # Find smallest valid new_val < old_val
                    found = None
                    for new_val in range(min_val, old_val):
                        valid = True
                        new_diffs = set()
                        for j in range(k + 1):
                            if j == i:
                                continue
                            diff = abs(new_val - row[j])
                            if diff in global_diffs:
                                valid = False
                                break
                            new_diffs.add(diff)
                        if valid:
                            found = (new_val, new_diffs)
                            break

                    if found is not None:
                        new_val, new_diffs = found
                        row[i] = new_val
                        for diff in new_diffs:
                            global_diffs.add(diff)
                        improved = True
                        new_scope = compute_scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in rows]
                    else:
                        # Revert
                        for diff in old_diffs:
                            global_diffs.add(diff)

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

        return best_rows, best_scope

    # Main: multiple restarts
    best_overall_rows = None
    best_overall_scope = float('inf')

    # First: greedy init
    rng0 = random.Random(12345)
    init_rows = greedy_init(rng0)
    if init_rows is None:
        # Fallback
        init_rows = [[i * (j + 1) for j in range(k + 1)] for i in range(n_rows)]
    init_scope = compute_scope(init_rows)
    best_overall_rows = [r[:] for r in init_rows]
    best_overall_scope = init_scope

    # Local refine on greedy
    t_remain = deadline - time.time()
    if t_remain > 2:
        refined, ref_scope = local_refine(init_rows, random.Random(777), min(20, t_remain - 1))
        if ref_scope < best_overall_scope:
            best_overall_scope = ref_scope
            best_overall_rows = refined

    # Simulated annealing with multiple restarts
    t_remain = deadline - time.time()
    if t_remain > 5:
        # Allocate time among restarts
        n_restarts = min(5, max(1, int(t_remain / 15)))
        per_restart = (t_remain - 2) / n_restarts

        for restart in range(n_restarts):
            if time.time() > deadline - 2:
                break
            rng = random.Random(1000 + restart * 137)
            # Perturb the current best
            perturbed = [r[:] for r in best_overall_rows]
            # Randomly perturb some elements
            for _ in range(k * n_rows // 2):
                r_idx = rng.randint(0, n_rows - 1)
                i = rng.randint(1, k)
                # Small random shift
                shift = rng.randint(-10, 10)
                new_val = perturbed[r_idx][i] + shift
                min_val = perturbed[r_idx][i - 1] + 1
                if i < k:
                    max_val = perturbed[r_idx][i + 1] - 1
                    new_val = min(new_val, max_val)
                new_val = max(new_val, min_val)
                if new_val != perturbed[r_idx][i]:
                    perturbed[r_idx][i] = new_val
            # Ensure strictly increasing
            valid = True
            for r in perturbed:
                for j in range(1, len(r)):
                    if r[j] <= r[j - 1]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                # Rebuild with greedy from perturbed start
                perturbed = greedy_init(rng)
                if perturbed is None:
                    continue

            t_left = deadline - time.time() - 1
            if t_left < 3:
                break
            sa_rows, sa_scope = anneal(perturbed, rng, min(per_restart, t_left))
            if sa_scope < best_overall_scope:
                best_overall_scope = sa_scope
                best_overall_rows = sa_rows

            # Local refine after SA
            t_left = deadline - time.time() - 1
            if t_left > 3:
                lr_rows, lr_scope = local_refine(sa_rows, random.Random(555 + restart), min(10, t_left))
                if lr_scope < best_overall_scope:
                    best_overall_scope = lr_scope
                    best_overall_rows = lr_rows

    # Final validation
    if not validate(best_overall_rows):
        # Fallback to greedy
        fallback = greedy_init(random.Random(999))
        if fallback is not None:
            best_overall_rows = fallback

    return best_overall_rows
