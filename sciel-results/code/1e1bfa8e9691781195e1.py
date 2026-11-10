"""Optimized program: greedy + targeted backtracking with strong pruning."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75

    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best = greedy()
    best_len = len(best)

    def check_add(values, sums, x):
        total = x
        new_sums = [total]
        for a in reversed(values):
            total += a
            new_sums.append(total)
        for s in new_sums:
            if s in sums:
                return None
        return new_sums

    # Compute all block sums for a sequence
    def all_block_sums(vals):
        s = set()
        for i in range(len(vals)):
            t = 0
            for j in range(i, len(vals)):
                t += vals[j]
                s.add(t)
        return s

    # Validate a sequence
    def valid(seq):
        if len(seq) != len(set(seq)):
            return False
        if any(v < 1 or v > 256 for v in seq):
            return False
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        bs = all_block_sums(seq)
        return len(bs) == len(seq) * (len(seq) + 1) // 2

    # Backtrack from a given prefix to target length
    def backtrack_from(prefix, target_len, time_limit):
        vals = list(prefix)
        sums = all_block_sums(vals) if vals else set()
        lo = vals[-1] + 1 if vals else 1

        def bt(vals, sums, lo, tlimit):
            if time.time() > tlimit:
                return None
            if len(vals) == target_len:
                return list(vals)

            # Pruning: not enough values left
            if 256 - lo + 1 < target_len - len(vals):
                return None

            for x in range(lo, 257):
                if time.time() > tlimit:
                    return None
                ns = check_add(vals, sums, x)
                if ns is None:
                    continue
                vals.append(x)
                sums2 = sums | set(ns)
                res = bt(vals, sums2, x + 1, tlimit)
                if res is not None:
                    return res
                vals.pop()
            return None

        return bt(vals, sums, lo, time_limit)

    # Strategy: try to extend beyond greedy
    # For each target length, try backtracking from the greedy prefix
    # with progressively shorter prefixes to allow more flexibility

    for target_len in range(best_len + 1, best_len + 12):
        if time.time() > deadline:
            break

        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break

        # Try with full greedy prefix first (fastest if it works)
        max_prefix = min(best_len, target_len - 1)
        found = None

        # Try prefixes from longest to shortest
        for plen in range(max_prefix, max(0, target_len - 8) - 1, -1):
            if time.time() > deadline:
                break
            prefix = best[:plen]
            # Allocate time proportional to how much freedom we have
            freedom = target_len - plen
            tlimit = min(deadline, time.time() + remaining_time / max(1, (max_prefix - plen + 1)) * 0.7)
            if time.time() >= tlimit:
                continue
            result = backtrack_from(prefix, target_len, tlimit)
            if result is not None and valid(result):
                found = result
                break

        if found is not None:
            best = found
            best_len = target_len

    return best
# EVOLVE-BLOCK-END
