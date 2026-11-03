import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = []
    for idx in range(total):
        s = {idx}
        if radius >= 1:
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        s.add(idx + (new_digit - digit) * pow_q_i)
        if radius >= 2:
            # distance-2 neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    pow_i, pow_j = q ** i, q ** j
                    di = (idx // pow_i) % q
                    dj = (idx // pow_j) % q
                    for a in range(q):
                        for b in range(q):
                            if a == di and b == dj:
                                continue
                            s.add(idx + (a - di) * pow_i + (b - dj) * pow_j)
        neigh.append(list(s))
    return neigh, total

# Cache
_cache = {}

def _get_data(q, n, radius):
    key = (q, n, radius)
    if key not in _cache:
        neigh, total = _precompute_neighbors(q, n, radius)
        # bitmask of neighbors for each codeword
        masks = [0] * total
        for i in range(total):
            m = 0
            for u in neigh[i]:
                m |= (1 << u)
            masks[i] = m
        _cache[key] = (neigh, masks, total)
    return _cache[key]

def construct(q, n, radius):
    deadline = time.monotonic() + 118.0
    neigh, masks, total = _get_data(q, n, radius)
    rng = random.Random(12345)

    def idx_to_word(idx):
        w = [0] * n
        t = idx
        for i in range(n):
            w[i] = t % q
            t //= q
        return w

    # Greedy initialization using bitmask coverage
    def greedy_init(seed):
        r = random.Random(seed)
        covered = 0
        full = (1 << total) - 1
        code = []
        while covered != full:
            if time.monotonic() > deadline:
                break
            best_c, best_gain = -1, -1
            # candidate ordering: try random subset to add diversity
            order = list(range(total))
            r.shuffle(order)
            for c in order:
                gain = bin(masks[c] & ~covered).count('1')
                if gain > best_gain:
                    best_gain, best_c = gain, c
            if best_c == -1 or best_gain <= 0:
                break
            code.append(best_c)
            covered |= masks[best_c]
        # ensure full coverage
        if covered != full:
            code_set = set(code)
            for u in range(total):
                if not (covered >> u) & 1:
                    for nc in neigh[u]:
                        if nc not in code_set:
                            code.append(nc)
                            code_set.add(nc)
                            covered |= masks[nc]
                            break
        return code

    # Iterative compression local search
    def local_search(code, time_budget):
        code_set = set(code)
        # coverage count per word
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        best_code = code[:]
        best_size = len(code)
        end = time.monotonic() + time_budget

        def removable(c):
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True

        # Phase 1: remove all removable
        changed = True
        while changed and time.monotonic() < end:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]

        # Phase 2: iterative compression with 1-swap (remove a, add b)
        # Precompute for speed: for each codeword a, the set of words it uniquely covers
        # We'll compute gain of a swap on the fly but cheaply.
        while time.monotonic() < end:
            improved = False
            # Try removing each codeword and adding the best replacement
            for ai in range(len(code)):
                if time.monotonic() > end:
                    break
                a = code[ai]
                # words uniquely covered by a
                unique = [u for u in neigh[a] if counts[u] == 1]
                if not unique:
                    # removable
                    code.pop(ai)
                    code_set.discard(a)
                    for u in neigh[a]:
                        counts[u] -= 1
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                    break
                # find best b not in code_set that covers all of unique
                # candidate b's: those that are neighbors of at least one unique word
                cand = set()
                for u in unique:
                    cand.update(neigh[u])
                cand -= code_set
                best_b, best_extra = None, -1
                for b in cand:
                    mb = masks[b]
                    # must cover all unique
                    ok = True
                    for u in unique:
                        if not (mb >> u) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    extra = bin(mb & ~(masks[a] if False else 0)).count('1')  # not used
                    # We prefer b that is not already redundant; just accept first valid
                    best_b = b
                    break
                if best_b is not None:
                    # perform swap
                    for u in neigh[a]:
                        counts[u] -= 1
                    for u in neigh[best_b]:
                        counts[u] += 1
                    code[ai] = best_b
                    code_set.discard(a)
                    code_set.add(best_b)
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                    break
            if not improved:
                # perturbation: random swap to escape local optimum
                if len(code) >= 2 and time.monotonic() < end - 0.5:
                    i = rng.randint(0, len(code) - 1)
                    a = code[i]
                    # pick random b
                    for _ in range(20):
                        b = rng.randint(0, total - 1)
                        if b in code_set:
                            continue
                        # perform swap even if it temporarily breaks coverage, then repair
                        for u in neigh[a]:
                            counts[u] -= 1
                        for u in neigh[b]:
                            counts[u] += 1
                        code[i] = b
                        code_set.discard(a)
                        code_set.add(b)
                        # repair: add codewords for uncovered words
                        for u in range(total):
                            if counts[u] == 0:
                                for nc in neigh[u]:
                                    if nc not in code_set:
                                        code.append(nc)
                                        code_set.add(nc)
                                        for v in neigh[nc]:
                                            counts[v] += 1
                                        break
                        # remove removable
                        for j in range(len(code) - 1, -1, -1):
                            if removable(code[j]):
                                c = code.pop(j)
                                code_set.discard(c)
                                for u in neigh[c]:
                                    counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        break
                else:
                    # stuck, no perturbation possible
                    break
            # re-remove removable after any change
            changed = True
            while changed and time.monotonic() < end:
                changed = False
                for i in range(len(code) - 1, -1, -1):
                    if removable(code[i]):
                        c = code.pop(i)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]

        return best_code, best_size

    # Multi-restart
    overall_best, overall_size = None, total + 1
    num_restarts = 10
    t0 = time.monotonic()
    for r in range(num_restarts):
        if time.monotonic() > deadline - 3:
            break
        code = greedy_init(1000 + r * 37)
        if not code:
            continue
        budget = max(0.5, (deadline - time.monotonic()) / (num_restarts - r) - 0.2)
        bc, bs = local_search(code, budget)
        if bs < overall_size:
            overall_size = bs
            overall_best = bc[:]
        # also keep the greedy itself as a candidate
        if len(code) < overall_size:
            overall_size = len(code)
            overall_best = code[:]

    if overall_best is None:
        overall_best = greedy_init(0)
        overall_size = len(overall_best)

    return [idx_to_word(c) for c in overall_best]
