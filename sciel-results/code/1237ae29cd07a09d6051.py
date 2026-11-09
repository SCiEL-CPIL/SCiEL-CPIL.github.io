import time
import random

def _compute_tails(values, x):
    """Compute the set of new consecutive-block sums if we append x to values."""
    tails = set()
    total = x
    tails.add(total)
    for a in reversed(values):
        total += a
        tails.add(total)
    return tails

def _is_valid_extension(values, sums, x):
    """Check if appending x to values creates no duplicate block sums."""
    tails = _compute_tails(values, x)
    return not (sums & tails)

def _all_sums(seq):
    """Compute the full set of nonempty consecutive-block sums of seq."""
    s = set()
    n = len(seq)
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += seq[j]
            s.add(acc)
    return s

def _greedy(values, sums, candidates, deadline):
    """Extend values by greedily taking valid candidates in order."""
    ext = list(values)
    ext_sums = set(sums)
    for x in candidates:
        if time.time() > deadline:
            break
        if _is_valid_extension(ext, ext_sums, x):
            ext.append(x)
            ext_sums |= _compute_tails(ext, x)
    return ext, ext_sums

def _beam_search(deadline, beam_width=25, max_steps=60):
    """Beam search over sequence prefixes, keeping longest prefixes."""
    beams = [([], set())]
    best = []
    for step in range(max_steps):
        if time.time() > deadline:
            break
        new_beams = []
        for values, sums in beams:
            start = (values[-1] + 1) if values else 1
            for x in range(start, 257):
                if _is_valid_extension(values, sums, x):
                    new_beams.append((values + [x], sums | _compute_tails(values, x)))
        if not new_beams:
            break
        new_beams.sort(key=lambda b: len(b[0]), reverse=True)
        beams = new_beams[:beam_width]
        for values, _ in beams:
            if len(values) > len(best):
                best = list(values)
    return best

def _dfs_search(deadline):
    """Depth-first search with 1-step lookahead heuristic, returning best prefix."""
    best_seq = []
    best_len = 0
    nodes = 0
    max_nodes = 200000

    def search(seq, sums, min_val):
        nonlocal best_seq, best_len, nodes
        nodes += 1
        if time.time() > deadline or nodes > max_nodes:
            return
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

        candidates = []
        for x in range(min_val, 257):
            if _is_valid_extension(seq, sums, x):
                candidates.append(x)
        if not candidates:
            return

        scored = []
        for x in candidates:
            ns = sums | _compute_tails(seq, x)
            nseq = seq + [x]
            # 1-step lookahead: count a few valid next values
            cnt = 0
            for y in range(x + 1, 257):
                if _is_valid_extension(nseq, ns, y):
                    cnt += 1
                    if cnt >= 5:
                        break
            scored.append((cnt, -x, x, ns))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        # Explore top branch
        _, _, x, ns = scored[0]
        search(seq + [x], ns, x + 1)

        # Explore second branch if time and budget remain
        if (time.time() < deadline - 8 and nodes < max_nodes * 0.7
                and len(scored) > 1 and scored[1][0] > 0):
            _, _, x2, ns2 = scored[1]
            search(seq + [x2], ns2, x2 + 1)

    search([], set(), 1)
    return best_seq

def _local_search(initial, deadline, iterations=4000):
    """Polish a sequence by inserting/removing/replacing elements to grow length."""
    if not initial:
        return initial
    best = list(initial)
    best_sums = _all_sums(best)
    best_len = len(best)
    rng = random.Random(12345)

    for _ in range(iterations):
        if time.time() > deadline:
            break
        improved = False

        # Try inserting a new element at a random position
        for _ in range(40):
            pos = rng.randint(0, len(best))
            low = best[pos - 1] + 1 if pos > 0 else 1
            high = best[pos] - 1 if pos < len(best) else 256
            if low > high:
                continue
            x = rng.randint(low, high)
            test = best[:pos] + [x] + best[pos:]
            ts = _all_sums(test)
            if len(ts) == len(test) * (len(test) + 1) // 2:
                # all block sums distinct -> valid (count check)
                best = test
                best_sums = ts
                best_len = len(best)
                improved = True
                break

        if improved:
            continue

        # Try removing one element and replacing with a different one
        if len(best) > 1:
            rem = rng.randint(0, len(best) - 1)
            without = best[:rem] + best[rem + 1:]
            low = without[rem - 1] + 1 if rem > 0 else 1
            high = without[rem] - 1 if rem < len(without) else 256
            if low <= high:
                for x in rng.sample(range(low, high + 1), min(8, high - low + 1)):
                    test = without[:rem] + [x] + without[rem:]
                    ts = _all_sums(test)
                    if len(ts) == len(test) * (len(test) + 1) // 2 and len(test) >= best_len:
                        best = test
                        best_sums = ts
                        best_len = len(best)
                        improved = True
                        break

    return best

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    best = []

    # 1) Fast greedy initializers
    for cand in (range(1, 257), range(256, 0, -1)):
        res, _ = _greedy([], set(), cand, deadline)
        if len(res) > len(best):
            best = res

    # 2) DFS with lookahead
    dfs_res = _dfs_search(deadline)
    if len(dfs_res) > len(best):
        best = dfs_res

    # 3) Beam search for diversity
    beam_res = _beam_search(deadline, beam_width=25, max_steps=60)
    if len(beam_res) > len(best):
        best = beam_res

    # 4) Local-search polish on the best so far
    if time.time() < deadline - 10:
        polished = _local_search(best, deadline, iterations=4000)
        if len(polished) > len(best):
            best = polished

    # Final validity guarantee
    if best:
        sums = _all_sums(best)
        if len(sums) != len(best) * (len(best) + 1) // 2:
            # Fallback: rebuild greedily to be safe
            res, _ = _greedy([], set(), range(1, 257), deadline)
            best = res

    return best
