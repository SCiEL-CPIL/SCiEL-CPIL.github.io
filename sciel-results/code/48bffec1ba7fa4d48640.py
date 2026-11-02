"""Covering code K_3(7,1): greedy + redundancy removal + local search."""
import time
import itertools
import random

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin under 150s kill


def _build_universe(q, n):
    total = q ** n
    # index <-> vector
    vecs = [list(itertools.product(range(q), repeat=n))]  # placeholder
    vecs = [list(v) for v in itertools.product(range(q), repeat=n)]
    # neighbors within radius 1 (including self)
    neigh = []
    for i in range(total):
        v = vecs[i]
        s = [i]
        for pos in range(n):
            orig = v[pos]
            for a in range(q):
                if a == orig:
                    continue
                nv = v[:pos] + [a] + v[pos + 1:]
                j = 0
                base = 1
                for k in range(n - 1, -1, -1):
                    j += nv[k] * base
                    base *= q
                s.append(j)
        neigh.append(tuple(s))
    return vecs, neigh, total


def _to_index(vec, q, n):
    idx = 0
    for k in range(n - 1, -1, -1):
        idx = idx * q + vec[k]
    return idx


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 135.0

    if radius == 1 and q == 3 and n == 7:
        return _solve_ternary_7_1()

    # Generic fallback: greedy set cover with redundancy removal
    vecs, neigh, total = _build_universe(q, n)
    uncovered = [True] * total
    code = []
    for _ in range(total):
        if time.time() > _DEADLINE:
            break
        best = -1
        best_gain = -1
        for cand in range(total):
            if not uncovered[cand]:
                continue
            gain = 0
            for nb in neigh[cand]:
                if uncovered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = cand
        if best < 0:
            break
        code.append(best)
        for nb in neigh[best]:
            uncovered[nb] = False
        if not any(uncovered):
            break
    code = _remove_redundant(code, neigh, total)
    return [list(vecs[c]) for c in code]


def _remove_redundant(code, neigh, total):
    covered_by = [set() for _ in range(total)]
    for ci, c in enumerate(code):
        for nb in neigh[c]:
            covered_by[nb].add(ci)
    changed = True
    while changed:
        changed = False
        new_code = []
        for ci, c in enumerate(code):
            redundant = True
            for nb in neigh[c]:
                others = covered_by[nb] - {ci}
                if not others:
                    redundant = False
                    break
            if redundant:
                # remove ci: update covered_by
                for nb in neigh[c]:
                    covered_by[nb].discard(ci)
                changed = True
            else:
                new_code.append(c)
        code = new_code
    return code


def _solve_ternary_7_1():
    q, n = 3, 7
    vecs, neigh, total = _build_universe(q, n)
    rng = random.Random(12345)

    # Greedy seed with a little randomization for diversity
    def greedy_seed(seed_rng):
        uncovered = [True] * total
        code = []
        for _ in range(total):
            if time.time() > _DEADLINE:
                break
            best = -1
            best_gain = -1
            for cand in range(total):
                if not uncovered[cand]:
                    continue
                gain = 0
                for nb in neigh[cand]:
                    if uncovered[nb]:
                        gain += 1
                if gain > best_gain or (gain == best_gain and best_gain >= 0 and seed_rng.random() < 0.01):
                    best_gain = gain
                    best = cand
            if best < 0:
                break
            code.append(best)
            for nb in neigh[best]:
                uncovered[nb] = False
            if not any(uncovered):
                break
        return code

    # Start with a good deterministic seed, then improve
    code = greedy_seed(rng)
    code = _remove_redundant(code, neigh, total)

    # Local search: try removing, then 1-swap
    def covered_union_mask(code_idx):
        mask = [0] * total
        for ci in code_idx:
            for nb in neigh[ci]:
                mask[nb] = 1
        return mask

    def is_covered(code_idx):
        for ci in code_idx:
            for nb in neigh[ci]:
                pass
        # check all covered
        seen = [False] * total
        for ci in code_idx:
            for nb in neigh[ci]:
                seen[nb] = True
        return all(seen)

    # Redundancy removal loop with swaps
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        # Removal pass
        code = _remove_redundant(code, neigh, total)
        # Swap pass: for each codeword, try replacing with another to allow a removal elsewhere
        # Simpler: try adding a word then removing two
        if time.time() > _DEADLINE:
            break
        n_code = len(code)
        tried = 0
        for i in range(n_code):
            if time.time() > _DEADLINE or tried > 200:
                break
            tried += 1
            # try removing i
            candidate = code[:i] + code[i + 1:]
            if is_covered(candidate):
                code = candidate
                improved = True
                break
    # Final redundancy removal
    code = _remove_redundant(code, neigh, total)
    return [list(vecs[c]) for c in code]
