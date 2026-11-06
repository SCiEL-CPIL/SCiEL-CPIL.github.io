import time
import random
import sys

_DEADLINE = 75.0  # safety margin under 90s wall-clock
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)


def _all_block_sums(seq):
    """Return set of sums of all nonempty consecutive blocks."""
    s = set()
    n = len(seq)
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += seq[j]
            s.add(acc)
    return s


def _is_valid(seq):
    if not seq:
        return True
    if len(set(seq)) != len(seq):
        return False
    for x in seq:
        if x < 1 or x > 256:
            return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    # consecutive block sums distinct
    n = len(seq)
    seen = set()
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += seq[j]
            if acc in seen:
                return False
            seen.add(acc)
    return True


def _greedy_ascending():
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


def _greedy_descending(start_high):
    """Try to build by considering larger values with a bias, but still
    enforce strictly increasing order in the final sequence by insertion.
    Simpler: build a candidate set by scanning from high to low and adding
    if it can be placed while keeping sequence sorted and sums unique.
    For simplicity we just scan ascending but prefer larger early picks via
    a different tie-break: actually we just return ascending greedy variant
    that skips to leave room. We'll implement a 'greedy with lookahead' that
    at each step picks the largest feasible next value."""
    values = []
    sums = set()
    used = set()
    for _ in range(256):
        best = None
        # try from high to low
        for x in range(256, 0, -1):
            if x in used:
                continue
            if values and x <= values[-1]:
                continue
            # compute new tails
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            best = x
            break
        if best is None:
            break
        values.append(best)
        used.add(best)
        tails = [best]
        total = best
        for a in reversed(values[:-1]):
            total += a
            tails.append(total)
        sums.update(tails)
    return values


def _greedy_randomized(seed):
    """Randomized greedy: at each step, among feasible next values, pick
    randomly with bias toward larger values."""
    rng = random.Random(seed)
    values = []
    sums = set()
    used = set()
    while True:
        feasible = []
        for x in range(1, 257):
            if x in used:
                continue
            if values and x <= values[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            feasible.append(x)
        if not feasible:
            break
        # bias toward larger: pick from top half weighted
        # simple: pick with probability proportional to (x+1)
        weights = [x + 1 for x in feasible]
        total_w = sum(weights)
        r = rng.random() * total_w
        acc = 0
        chosen = feasible[-1]
        for x, w in zip(feasible, weights):
            acc += w
            if r <= acc:
                chosen = x
                break
        values.append(chosen)
        used.add(chosen)
        tails = [chosen]
        total = chosen
        for a in reversed(values[:-1]):
            total += a
            tails.append(total)
        sums.update(tails)
    return values


def _try_grow(seq):
    """Given a valid strictly-increasing seq, try to extend it greedily
    with the largest feasible values. Returns the grown sequence."""
    seq = list(seq)
    sums = _all_block_sums(seq)
    used = set(seq)
    while True:
        best = None
        for x in range(256, 0, -1):
            if x in used:
                continue
            if seq and x <= seq[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            best = x
            break
        if best is None:
            break
        seq.append(best)
        used.add(best)
        tails = [best]
        total = best
        for a in reversed(seq[:-1]):
            total += a
            tails.append(total)
        sums.update(tails)
    return seq


def _sa_improve(seq, seed, max_iters, time_limit):
    """Simulated annealing on the set of selected values.
    State: a set of values forming a valid strictly-increasing sequence.
    Moves: add a random feasible value, remove a random value, or swap.
    Objective: maximize length (cardinality). We accept worse (shorter)
    states with annealing probability to escape local optima, but we
    always keep the best valid sequence seen.
    """
    rng = random.Random(seed)
    best = list(seq)
    best_len = len(best)
    cur = list(seq)
    cur_sums = _all_block_sums(cur)
    cur_used = set(cur)

    T = 1.0
    cool = 0.9999
    it = 0
    t0 = time.time()
    while it < max_iters and (time.time() - t0) < time_limit:
        it += 1
        # random move
        move = rng.random()
        new = cur
        new_sums = cur_sums
        new_used = cur_used

        if move < 0.4:
            # add a random unused value that keeps validity
            candidates = [x for x in range(1, 257) if x not in cur_used]
            if not candidates:
                continue
            x = rng.choice(candidates)
            # must be greater than last to keep increasing, OR we could
            # insert in sorted position but that changes block structure.
            # For simplicity only allow appending (x > cur[-1]) or if cur empty.
            if cur and x <= cur[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(cur):
                total += a
                tails.append(total)
            if cur_sums.intersection(tails):
                continue
            new = cur + [x]
            new_used = cur_used | {x}
            new_sums = cur_sums | set(tails)
        elif move < 0.7:
            # remove a random element
            if not cur:
                continue
            idx = rng.randrange(len(cur))
            new = cur[:idx] + cur[idx+1:]
            new_used = cur_used - {cur[idx]}
            new_sums = _all_block_sums(new)
        else:
            # swap: remove one, add another
            if not cur:
                continue
            idx = rng.randrange(len(cur))
            removed = cur[idx]
            base = cur[:idx] + cur[idx+1:]
            candidates = [x for x in range(1, 257) if x not in cur_used or x == removed]
            if not candidates:
                continue
            x = rng.choice(candidates)
            # need to insert x into base keeping strictly increasing
            # find position
            pos = None
            for i in range(len(base)):
                if base[i] > x:
                    pos = i
                    break
            if pos is None:
                pos = len(base)
            if pos < len(base) and base[pos] == x:
                continue
            new = base[:pos] + [x] + base[pos:]
            new_used = (cur_used - {removed}) | {x}
            new_sums = _all_block_sums(new)
            # validate increasing
            ok = True
            for i in range(1, len(new)):
                if new[i] <= new[i-1]:
                    ok = False
                    break
            if not ok:
                continue

        new_len = len(new)
        cur_len = len(cur)
        delta = new_len - cur_len
        if delta >= 0:
            accept = True
        else:
            # worse: accept with prob exp(delta/T)
            if T > 0:
                import math
                p = math.exp(delta / T)
                accept = rng.random() < p
            else:
                accept = False
        if accept:
            cur = new
            cur_sums = new_sums
            cur_used = new_used
            if new_len > best_len:
                best = list(new)
                best_len = new_len
        T *= cool
    return best


def construct():
    # Fast baseline
    base_asc = _greedy_ascending()
    base_desc = _greedy_descending(256)
    best = base_asc if len(base_asc) >= len(base_desc) else base_desc

    # Try a few randomized greedy seeds quickly
    for s in range(8):
        if _time_left() < 5:
            break
        cand = _greedy_randomized(1000 + s)
        cand = _try_grow(cand)
        if len(cand) > len(best):
            best = cand

    # Simulated annealing from best and from a couple of other starts
    starts = [best]
    starts.append(base_asc)
    starts.append(base_desc)
    # a couple of randomized greedy starts
    for s in range(3):
        if _time_left() < 8:
            break
        cand = _greedy_randomized(5000 + s)
        cand = _try_grow(cand)
        starts.append(cand)

    for i, st in enumerate(starts):
        if _time_left() < 10:
            break
        t_left = _time_left() - 2
        if t_left <= 0:
            break
        max_iters = 200000
        cand = _sa_improve(st, 7000 + i, max_iters, t_left)
        if len(cand) > len(best):
            best = cand

    # Final validation
    if not _is_valid(best):
        # fallback to ascending greedy which is known valid
        best = _greedy_ascending()

    return list(best)


if __name__ == "__main__":
    print(construct())
