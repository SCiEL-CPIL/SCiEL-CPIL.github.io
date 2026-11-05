"""Improved construction for the increasing-unique-consecutive-block-sums problem."""
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # safety margin before 60s (evaluator kills at 90s, but keep conservative)

    def compute_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                if total in sums:
                    return None  # duplicate found
                sums.add(total)
        return sums

    def is_valid(seq):
        """Check if sequence has all distinct consecutive block sums."""
        return compute_sums(seq) is not None

    def greedy_forward():
        """Greedy: pick smallest valid next value."""
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            if time.time() > start_time + time_limit:
                break
            found = False
            for x in range(last_val + 1, 257):
                total = x
                ok = True
                if total in sums:
                    ok = False
                else:
                    for a in reversed(values):
                        total += a
                        if total in sums:
                            ok = False
                            break
                if ok:
                    values.append(x)
                    last_val = x
                    total = x
                    sums.add(total)
                    for a in reversed(values[:-1]):
                        total += a
                        sums.add(total)
                    found = True
                    break
            if not found:
                break
        return values

    def greedy_reverse():
        """Greedy: pick largest valid next value (build from high end)."""
        values = []
        sums = set()
        next_val = 257
        while next_val > 1:
            if time.time() > start_time + time_limit:
                break
            found = False
            for x in range(next_val - 1, 0, -1):
                total = x
                ok = True
                if total in sums:
                    ok = False
                else:
                    for a in reversed(values):
                        total += a
                        if total in sums:
                            ok = False
                            break
                if ok:
                    values.append(x)
                    next_val = x
                    total = x
                    sums.add(total)
                    for a in reversed(values[:-1]):
                        total += a
                        sums.add(total)
                    found = True
                    break
            if not found:
                break
        return values[::-1]

    def randomized_trial(strategy, rand, deadline):
        """Randomized greedy with various strategies."""
        values = []
        sums = set()
        last_val = 0

        while last_val < 256:
            if time.time() > deadline:
                break

            candidates = []
            for x in range(last_val + 1, 257):
                total = x
                ok = True
                if total in sums:
                    ok = False
                else:
                    for a in reversed(values):
                        total += a
                        if total in sums:
                            ok = False
                            break
                if ok:
                    candidates.append(x)

            if not candidates:
                break

            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                k = min(10, len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                k = min(10, len(candidates))
                x = rand.choice(candidates[:k])
            elif strategy == 'spread':
                n = len(candidates)
                lo = n // 4
                hi = 3 * n // 4
                if hi > lo:
                    x = rand.choice(candidates[lo:hi])
                else:
                    x = rand.choice(candidates)
            elif strategy == 'skip':
                # Skip some candidates to explore non-greedy paths
                if len(candidates) > 3:
                    # Skip the first few, pick from the rest
                    skip = rand.randint(1, min(3, len(candidates) - 1))
                    x = rand.choice(candidates[skip:])
                else:
                    x = rand.choice(candidates)
            else:
                x = rand.choice(candidates)

            values.append(x)
            last_val = x
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)

        return values

    def backtracking_search(max_depth, deadline, max_branches=3):
        """
        Depth-first backtracking: at each position, try up to max_branches candidates.
        This explores non-greedy branches that greedy methods miss.
        """
        best_seq = []
        best_len = 0

        def dfs(values, sums, last_val, depth):
            nonlocal best_seq, best_len
            if time.time() > deadline:
                return

            if len(values) > best_len:
                best_len = len(values)
                best_seq = values[:]

            if last_val >= 256 or depth >= max_depth:
                return

            # Find valid candidates
            candidates = []
            for x in range(last_val + 1, 257):
                total = x
                ok = True
                if total in sums:
                    ok = False
                else:
                    for a in reversed(values):
                        total += a
                        if total in sums:
                            ok = False
                            break
                if ok:
                    candidates.append(x)
                    if len(candidates) >= max_branches * 2:
                        break

            if not candidates:
                return

            # Try up to max_branches candidates (smallest, middle, largest)
            indices_to_try = [0]
            if len(candidates) > 1:
                indices_to_try.append(len(candidates) // 2)
            if len(candidates) > 2:
                indices_to_try.append(len(candidates) - 1)

            for idx in indices_to_try:
                if time.time() > deadline:
                    return
                x = candidates[idx]
                new_values = values + [x]
                new_sums = set(sums)
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                dfs(new_values, new_sums, x, depth + 1)

        dfs([], set(), 0, 0)
        return best_seq

    def hill_climb(seq, deadline):
        """Try to extend sequence by appending, or by local modifications."""
        values = list(seq)
        best_seq = values[:]
        best_len = len(values)

        def recompute_sums(s):
            sums = set()
            n = len(s)
            for i in range(n):
                total = 0
                for j in range(i, n):
                    total += s[j]
                    if total in sums:
                        return None
                    sums.add(total)
            return sums

        # Try extending from the end
        while time.time() < deadline:
            improved = False
            last_val = values[-1] if values else 0
            for x in range(last_val + 1, 257):
                new_seq = values + [x]
                if recompute_sums(new_seq) is not None:
                    values = new_seq
                    if len(values) > best_len:
                        best_len = len(values)
                        best_seq = values[:]
                    improved = True
                    break
            if not improved:
                break

        # Try inserting elements at various positions
        while time.time() < deadline:
            improved = False
            for pos in range(len(values) + 1):
                if time.time() > deadline:
                    break
                if pos == 0:
                    lo = 1
                    hi = values[0] - 1 if values else 256
                elif pos == len(values):
                    lo = values[-1] + 1
                    hi = 256
                else:
                    lo = values[pos - 1] + 1
                    hi = values[pos] - 1

                if lo > hi:
                    continue

                for x in range(lo, hi + 1):
                    new_seq = values[:pos] + [x] + values[pos:]
                    if recompute_sums(new_seq) is not None:
                        values = new_seq
                        if len(values) > best_len:
                            best_len = len(values)
                            best_seq = values[:]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return best_seq

    # Phase 1: Baselines
    best_seq = greedy_forward()
    best_len = len(best_seq)

    dec_seq = greedy_reverse()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)

    # Phase 2: Backtracking search (key improvement)
    phase2_deadline = start_time + time_limit * 0.3
    bt_seq = backtracking_search(20, phase2_deadline, max_branches=3)
    if len(bt_seq) > best_len:
        best_seq = bt_seq
        best_len = len(bt_seq)

    # Phase 3: Randomized trials
    strategies = ['random', 'large', 'small', 'spread', 'skip', 'random', 'large', 'skip']
    phase3_deadline = start_time + time_limit * 0.6
    trial_count = 0
    while time.time() < phase3_deadline:
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count * 1000 + 42)
        seq = randomized_trial(strategy, rand, time.time() + 1.0)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        trial_count += 1

    # Phase 4: Hill climbing on best sequence
    phase4_deadline = start_time + time_limit * 0.8
    hc_seq = hill_climb(best_seq, phase4_deadline)
    if len(hc_seq) > best_len:
        best_seq = hc_seq
        best_len = len(best_seq)

    # Phase 5: More backtracking with different parameters if time permits
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        phase5_deadline = start_time + time_limit - 1.0
        bt2_seq = backtracking_search(15, phase5_deadline, max_branches=4)
        if len(bt2_seq) > best_len:
            best_seq = bt2_seq
            best_len = len(bt2_seq)

        # Final hill climb
        final_deadline = start_time + time_limit - 0.5
        final_hc = hill_climb(best_seq, final_deadline)
        if len(final_hc) > best_len:
            best_seq = final_hc
            best_len = len(best_seq)

    # Ensure validity before returning
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy_forward()

    return best_seq
