import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 85.0  # safety margin before 90s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _sq_nonzero_count(c):
    """Return number of nonzero entries in the exact square of coefficient list c (ascending degree)."""
    n = len(c)
    # square via convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ai = c[i]
        if ai == 0:
            continue
        for j in range(n):
            aj = c[j]
            if aj != 0:
                sq[i + j] += ai * aj
    return sum(1 for v in sq if v != 0)

def _valid(c):
    if len(c) > 41:
        return False
    if len(c) == 0 or c[-1] == 0:
        return False
    nz = 0
    for v in c:
        if not isinstance(v, int):
            return False
        if v < -64 or v > 64:
            return False
        if v != 0:
            nz += 1
    if nz != 12:
        return False
    return True

def _trim(c):
    # remove trailing zeros
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c

def _random_poly(rng, max_deg=40, coeff_range=(-64, 64)):
    """Generate a random 12-term polynomial."""
    degs = sorted(rng.sample(range(max_deg + 1), 12))
    coeffs = [0] * (max_deg + 1)
    for d in degs:
        # avoid zero
        v = rng.randint(coeff_range[0], coeff_range[1])
        while v == 0:
            v = rng.randint(coeff_range[1] // 2, coeff_range[1])
        coeffs[d] = v
    return _trim(coeffs)

def _neighbor(c, rng, max_deg=40):
    """Perturb a polynomial: change a coefficient, move a term, or add/remove (keeping 12 terms)."""
    n = len(c)
    nz_indices = [i for i, v in enumerate(c) if v != 0]
    op = rng.random()
    if op < 0.4:
        # change a coefficient value
        idx = rng.choice(nz_indices)
        v = rng.randint(-64, 64)
        if v == 0:
            v = 1
        c[idx] = v
    elif op < 0.7:
        # move a term to a new degree
        idx = rng.choice(nz_indices)
        v = c[idx]
        new_deg = rng.randint(0, max_deg)
        if new_deg != idx:
            c[idx] = 0
            if new_deg >= len(c):
                c.extend([0] * (new_deg - len(c) + 1))
            c[new_deg] += v
            # if collision, may reduce term count; fix later
    else:
        # swap two coefficients' degrees
        if len(nz_indices) >= 2:
            i, j = rng.sample(nz_indices, 2)
            c[i], c[j] = c[j], c[i]
    # normalize: ensure exactly 12 nonzero, trim, pad
    _trim(c)
    # count nonzero
    nz = sum(1 for v in c if v != 0)
    if nz > 12:
        # remove extras (smallest magnitude first)
        items = [(abs(v), i) for i, v in enumerate(c) if v != 0]
        items.sort()
        for _, i in items[:nz - 12]:
            c[i] = 0
    elif nz < 12:
        # add terms at random empty degrees
        empty = [i for i in range(max_deg + 1) if i < len(c) and c[i] == 0]
        while len(empty) < 12 - nz:
            empty.append(len(c))
            c.append(0)
        rng.shuffle(empty)
        for i in empty[:12 - nz]:
            v = rng.randint(1, 64)
            if rng.random() < 0.5:
                v = -v
            c[i] = v
    _trim(c)
    # final check: if still not 12, rebuild
    if sum(1 for v in c if v != 0) != 12 or len(c) > 41:
        return None
    return c

def construct():
    rng = random.Random(12345)
    best_c = [1] * 12
    best_score = _sq_nonzero_count(best_c)

    # Start with a few structured seeds
    seeds = []
    # consecutive
    seeds.append([1] * 12)
    # geometric spacing
    for gap in [2, 3, 4, 5]:
        c = [0] * (1 + gap * 11)
        for k in range(12):
            c[gap * k] = 1
        seeds.append(_trim(c))
    # alternating signs
    c = [0] * 12
    for k in range(12):
        c[k] = 1 if k % 2 == 0 else -1
    seeds.append(c)
    # some random
    for _ in range(8):
        seeds.append(_random_poly(rng))

    # Evaluate seeds
    for s in seeds:
        if _valid(s):
            sc = _sq_nonzero_count(s)
            if sc < best_score:
                best_score = sc
                best_c = list(s)

    # Local search + random restarts within budget
    while _elapsed() < _DEADLINE:
        # local search from best
        cur = list(best_c)
        cur_score = best_score
        for step in range(200):
            if _elapsed() > _DEADLINE:
                break
            cand = _neighbor(cur, rng)
            if cand is None:
                break
            if not _valid(cand):
                continue
            sc = _sq_nonzero_count(cand)
            if sc < cur_score:
                cur_score = sc
                cur = cand
                if sc < best_score:
                    best_score = sc
                    best_c = list(cur)
            elif sc == cur_score and rng.random() < 0.3:
                # accept equal to escape plateaus
                cur = cand

        # random restart
        if _elapsed() > _DEADLINE:
            break
        for _ in range(3):
            if _elapsed() > _DEADLINE:
                break
            s = _random_poly(rng)
            if not _valid(s):
                continue
            sc = _sq_nonzero_count(s)
            if sc < best_score:
                best_score = sc
                best_c = list(s)

    # Final polish: try small coefficient tweaks on best
    if _elapsed() < _DEADLINE:
        for _ in range(50):
            if _elapsed() > _DEADLINE:
                break
            cand = list(best_c)
            nz = [i for i, v in enumerate(cand) if v != 0]
            idx = rng.choice(nz)
            cand[idx] += rng.choice([-1, 1])
            if cand[idx] < -64 or cand[idx] > 64:
                continue
            if cand[idx] == 0:
                # need to add another term
                empty = [i for i in range(len(cand)) if cand[i] == 0]
                if empty:
                    cand[rng.choice(empty)] = 1
            if not _valid(cand):
                continue
            sc = _sq_nonzero_count(cand)
            if sc < best_score:
                best_score = sc
                best_c = cand

    return best_c

if __name__ == "__main__":
    print(construct())
